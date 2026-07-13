#!/usr/bin/env python3
"""
JVP (Jacobian-Vector Product) analysis library.

Computes approximate (first-order) or full tangent propagation for
value-side feature vectors. Used by analysis/run.py.
"""

import os
import sys
from typing import List, Dict, Any, Optional

import torch
import torch.distributed as dist

from sae_lens import HookedSAETransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import (
    log_info,
    distribute_batch,
    Feature,
)

from analysis.common import (
    ResidCache,
    _expand_to_BSD,
    jvp_mlp, jvp_attn,
    get_feature_vector, distributed_mean_vector,
    apply_unembed, get_logits_tokens,
    jacobian_linearity,
)


def analyze_feature(
    model: HookedSAETransformer,
    f: Feature,
    sae,
    pile_sample: torch.Tensor,
    rank: int,
    world_size: int,
    stream_transition: str,  # "first_order" or "full"
    submodule: str = "both",
    k: int = 25,
    use_unembed_layer_norm: bool = True,
    use_jvp_layer_norm: bool = False,
    accumulation_steps: int = 1,
    resid_cache: Optional[ResidCache] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single feature using JVP tangent propagation.

    stream_transition="first_order": per-layer fixed-tangent JVP (approximate).
    stream_transition="full": chained tangent propagation through all layers.

    Returns uniform output keys: top_tokens, bottom_tokens, top_ids, bottom_ids,
    stats, and (for first_order) per-layer tokens and energy scores.

    NOTE: This function is NOT wrapped in try/except. Any exception must
    propagate so torchrun reports the failure instead of silently returning
    None on one rank while the other rank enters a collective and deadlocks.
    """
    submodule = submodule.lower()
    use_attn = submodule in ("both", "attn")
    use_mlp = submodule in ("both", "mlp")

    do_approx = (stream_transition == "first_order")
    do_full = (stream_transition == "full")

    # shard
    batch_size = pile_sample.shape[0]
    start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
    local_pile_sample = pile_sample[start_idx:end_idx]
    local_batch_size = local_pile_sample.shape[0]

    num_layers = model.cfg.n_layers
    layer_idx = int(f.layer)
    device = model.cfg.device
    dtype = model.cfg.dtype

    # direction vector
    f_dir = get_feature_vector(model, f, sae)

    # micro-batching
    accumulation_steps = max(int(accumulation_steps), 1)
    micro_batch_size = max(1, local_batch_size // accumulation_steps)
    if accumulation_steps == 1:
        micro_batch_size = local_batch_size

    # per-layer data (first_order only)
    layer_top_tokens: List[List[str]] = []
    layer_bottom_tokens: List[List[str]] = []
    layer_top_ids: List[List[int]] = []
    layer_bottom_ids: List[List[int]] = []
    layer_indices: List[int] = []
    layer_energy_scores: List[float] = []

    accumulated_jvp_per_layer: Dict[int, List[torch.Tensor]] = {}
    token_count_local: Optional[torch.Tensor] = None

    # full tangents and masks
    full_tangent_chunks: List[torch.Tensor] = []
    valid_mask_chunks: List[torch.Tensor] = []

    S = None
    d_model = None

    if resid_cache is None:
        raise RuntimeError("resid_cache is None. This version expects resid_cache enabled.")

    for micro_step in range(accumulation_steps):
        micro_start = micro_step * micro_batch_size
        micro_end = min((micro_step + 1) * micro_batch_size, local_batch_size)
        if micro_start >= local_batch_size:
            break

        micro_batch_size_actual = micro_end - micro_start

        # cache get
        if use_attn and use_mlp:
            R_layers_attn, R_layers_mlp, Vmask = resid_cache.get(micro_step, kind="all")
        elif use_attn:
            R_layers_attn, R_layers_mlp, Vmask = resid_cache.get(micro_step, kind="attn")
        elif use_mlp:
            R_layers_attn, R_layers_mlp, Vmask = resid_cache.get(micro_step, kind="mlp")
        else:
            raise ValueError("submodule must enable at least one of attn or mlp")

        # infer S,d
        if S is None or d_model is None:
            if use_attn and R_layers_attn is not None:
                _, _, S, d_model = R_layers_attn.shape
            elif use_mlp and R_layers_mlp is not None:
                _, _, S, d_model = R_layers_mlp.shape
            else:
                raise RuntimeError("Cached residuals are None for enabled submodule(s).")

        # mask
        if Vmask is None:
            mask_micro = torch.ones((micro_batch_size_actual, S), device=device, dtype=torch.float32)
        else:
            mask_micro = Vmask.to(device=device, non_blocking=True).float()
        valid_mask_chunks.append(mask_micro.bool())
        mask_bsd = mask_micro.to(dtype=dtype).unsqueeze(-1)  # [B,S,1]
        count_micro = mask_micro.sum().float()

        if token_count_local is None:
            token_count_local = count_micro
        else:
            token_count_local = token_count_local + count_micro

        # unpack layers
        if use_attn and R_layers_attn is not None:
            R_layers_attn_list = list(R_layers_attn.unbind(0))
        else:
            R_layers_attn_list = [None] * num_layers

        if use_mlp and R_layers_mlp is not None:
            R_layers_mlp_list = list(R_layers_mlp.unbind(0))
        else:
            R_layers_mlp_list = [None] * num_layers

        # expand f_dir
        f_dir_exp_micro = _expand_to_BSD(f_dir, micro_batch_size_actual, S, d_model, device, dtype)

        # Full tangent propagation
        if do_full:
            t = (f_dir_exp_micro.clone()) * mask_bsd
            for k_layer in range(layer_idx + 1, num_layers):
                if use_attn and R_layers_attn_list[k_layer] is not None:
                    x_attn = R_layers_attn_list[k_layer].to(device=device, dtype=dtype, non_blocking=True)
                    j_attn = jvp_attn(
                        model,
                        layer_idx=k_layer,
                        X=x_attn,
                        f_vec=t,
                        use_layer_norm=use_jvp_layer_norm,
                        attention_mask=mask_micro.bool(),
                    )
                    t = (t + j_attn) * mask_bsd
                    del x_attn, j_attn

                if use_mlp and R_layers_mlp_list[k_layer] is not None:
                    x_mlp = R_layers_mlp_list[k_layer].to(device=device, dtype=dtype, non_blocking=True)
                    j_mlp = jvp_mlp(
                        model,
                        layer_idx=k_layer,
                        X=x_mlp,
                        f_vec=t,
                        use_layer_norm=use_jvp_layer_norm,
                    )
                    t = (t + j_mlp) * mask_bsd
                    del x_mlp, j_mlp

            full_tangent_chunks.append(t.detach())
            del t

        # Approx (fixed tangent f_dir) per-layer accumulation
        if do_approx:
            for l in range(layer_idx, num_layers):
                layer_resid_attn = (
                    R_layers_attn_list[l].to(device=device, dtype=dtype, non_blocking=True)
                    if use_attn and R_layers_attn_list[l] is not None
                    else None
                )
                layer_resid_mlp = (
                    R_layers_mlp_list[l].to(device=device, dtype=dtype, non_blocking=True)
                    if use_mlp and R_layers_mlp_list[l] is not None
                    else None
                )

                if l == layer_idx:
                    jvp_individual_micro = f_dir_exp_micro.clone()
                else:
                    jvp_individual_micro = torch.zeros(micro_batch_size_actual, S, d_model, device=device, dtype=dtype)
                    if use_attn and layer_resid_attn is not None:
                        j_attn = jvp_attn(
                            model,
                            layer_idx=l,
                            X=layer_resid_attn,
                            f_vec=f_dir,
                            use_layer_norm=use_jvp_layer_norm,
                            attention_mask=mask_micro.bool(),
                        )
                        jvp_individual_micro = jvp_individual_micro + j_attn
                        del j_attn
                    if use_mlp and layer_resid_mlp is not None:
                        j_mlp = jvp_mlp(
                            model,
                            layer_idx=l,
                            X=layer_resid_mlp,
                            f_vec=f_dir,
                            use_layer_norm=use_jvp_layer_norm,
                        )
                        jvp_individual_micro = jvp_individual_micro + j_mlp
                        del j_mlp

                del layer_resid_attn, layer_resid_mlp

                jvp_individual_micro = jvp_individual_micro * mask_bsd
                if l not in accumulated_jvp_per_layer:
                    accumulated_jvp_per_layer[l] = []
                accumulated_jvp_per_layer[l].append(jvp_individual_micro)
                del jvp_individual_micro

        del R_layers_attn_list, R_layers_mlp_list, f_dir_exp_micro, mask_micro, mask_bsd, Vmask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # NOTE: Do NOT raise here on rank-local zero token counts. distributed_mean_vector
    # validates the GLOBAL count collectively after the all_reduce. Raising solo would
    # desynchronize the NCCL process group.
    if token_count_local is None:
        token_count_local = torch.zeros((), device=device, dtype=torch.float32)

    valid_mask_all = torch.cat(valid_mask_chunks, dim=0)  # [B_total,S] bool

    # Compute mean vector and tokens based on stream_transition
    if do_approx:
        # concat approx per-layer tensors
        for l in range(layer_idx, num_layers):
            if l in accumulated_jvp_per_layer:
                accumulated_jvp_per_layer[l] = torch.cat(accumulated_jvp_per_layer[l], dim=0)

        sum_jvp_all = None
        for l in range(layer_idx, num_layers):
            if l in accumulated_jvp_per_layer:
                layer_tensor = accumulated_jvp_per_layer[l]
                sum_jvp_all = layer_tensor.clone() if sum_jvp_all is None else sum_jvp_all + layer_tensor
        if sum_jvp_all is None:
            raise ValueError("No per-layer JVP tensors accumulated.")

        stats = jacobian_linearity(sum_jvp_all, valid_mask_all)

        approx_sum_local = sum_jvp_all.float().sum(dim=(0, 1))
        mean_vec = distributed_mean_vector(approx_sum_local, token_count_local).to(dtype=sum_jvp_all.dtype)

        # per-layer scoring
        a = mean_vec.float()
        a_norm2 = (a @ a).clamp_min(1e-12)
        for l in range(layer_idx, num_layers):
            if l not in accumulated_jvp_per_layer:
                continue
            jvp_individual = accumulated_jvp_per_layer[l]
            local_sum = jvp_individual.sum(dim=(0, 1))
            jvp_mean_vec = distributed_mean_vector(local_sum, token_count_local)
            mean_logits = apply_unembed(model, jvp_mean_vec, use_ln_final=use_unembed_layer_norm).view(-1)

            (tokens_tuple, ids_tuple) = get_logits_tokens(model, mean_logits, k)
            layer_top_tokens.append(tokens_tuple[0][0])
            layer_bottom_tokens.append(tokens_tuple[1][0])
            layer_top_ids.append(ids_tuple[0][0])
            layer_bottom_ids.append(ids_tuple[1][0])
            layer_indices.append(l)

            m = jvp_mean_vec.float()
            contrib = (m @ a) / a_norm2
            layer_energy_scores.append(contrib.item())
            del mean_logits

    elif do_full:
        if len(full_tangent_chunks) == 0:
            raise ValueError("No full tangent chunks accumulated.")
        sum_jvp_all_full = torch.cat(full_tangent_chunks, dim=0)

        stats = jacobian_linearity(sum_jvp_all_full, valid_mask_all)

        full_sum_local = sum_jvp_all_full.float().sum(dim=(0, 1))
        mean_vec = distributed_mean_vector(full_sum_local, token_count_local).to(dtype=sum_jvp_all_full.dtype)
    else:
        raise ValueError(f"Invalid stream_transition for JVP: {stream_transition}")

    # Apply readout (unembed)
    logits = apply_unembed(model, mean_vec, use_ln_final=use_unembed_layer_norm).view(-1)
    (tokens_tuple, ids_tuple) = get_logits_tokens(model, logits, k)

    result = {
        "layer": layer_idx,
        "top_tokens": tokens_tuple[0][0],
        "bottom_tokens": tokens_tuple[1][0],
        "top_ids": ids_tuple[0][0],
        "bottom_ids": ids_tuple[1][0],
        "stats": stats,
    }

    # Include per-layer data for first_order
    if do_approx:
        result.update({
            "layer_indices": layer_indices,
            "layer_top_tokens": layer_top_tokens,
            "layer_bottom_tokens": layer_bottom_tokens,
            "layer_top_ids": layer_top_ids,
            "layer_bottom_ids": layer_bottom_ids,
            "layer_energy_scores": layer_energy_scores,
        })

    del f_dir, valid_mask_chunks
    if do_approx:
        del accumulated_jvp_per_layer
    if do_full:
        del full_tangent_chunks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

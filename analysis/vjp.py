#!/usr/bin/env python3
"""
VJP (Vector-Jacobian Product) analysis library.

Computes approximate (first-order) or full cotangent propagation for
key-side feature vectors. Used by analysis/run.py.
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
    VJPResidChunk, VJPResidCache,
    _expand_to_BSD,
    vjp_mlp, vjp_attn,
    get_feature_input_vector, distributed_mean_vector,
    apply_embed_cosine_centered, get_logits_tokens,
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
    use_vjp_layer_norm: bool = False,
    accumulation_steps: int = 1,
    resid_cache: Optional[VJPResidCache] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single feature using VJP cotangent propagation.

    stream_transition="first_order": per-layer fixed-cotangent VJP (approximate).
    stream_transition="full": chained cotangent propagation through all layers.

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

    # distributed shard
    batch_size = pile_sample.shape[0]
    start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
    local_pile_sample = pile_sample[start_idx:end_idx]
    local_batch_size = local_pile_sample.shape[0]

    layer_idx = int(f.layer)
    device = model.cfg.device
    dtype = model.cfg.dtype
    num_layers = getattr(model.cfg, "n_layers", len(getattr(model, "blocks", [])))

    # fixed seed direction (input-side)
    seed_vec = get_feature_input_vector(model, f, sae).to(device=device, dtype=dtype)

    if resid_cache is None:
        raise RuntimeError("resid_cache is None. This VJP version expects resid_cache enabled.")

    # micro-batching
    accumulation_steps = max(int(accumulation_steps), 1)
    micro_bs = max(1, local_batch_size // accumulation_steps)
    if accumulation_steps == 1:
        micro_bs = local_batch_size

    # per-layer data (first_order only)
    layer_top_tokens: List[List[str]] = []
    layer_bottom_tokens: List[List[str]] = []
    layer_top_ids: List[List[int]] = []
    layer_bottom_ids: List[List[int]] = []
    layer_indices: List[int] = []
    layer_energy_scores: List[float] = []

    accumulated_vjp_per_layer: Dict[int, List[torch.Tensor]] = {}

    # token-embedding mean accumulators
    embed_sum_local: Optional[torch.Tensor] = None
    token_count_local: Optional[torch.Tensor] = None

    # full cotangent chunks + masks
    full_cot_chunks: List[torch.Tensor] = []
    valid_mask_chunks: List[torch.Tensor] = []

    for micro_step in range(accumulation_steps):
        ms = micro_step * micro_bs
        me = min((micro_step + 1) * micro_bs, local_batch_size)
        if ms >= local_batch_size:
            break

        R_pre, R_mid, E_emb, Vmask = resid_cache.get(micro_step, kind="all")
        if R_pre is None or R_mid is None:
            raise RuntimeError("ResidCache must contain both resid_pre(attn) and resid_mid(mlp).")
        if E_emb is None:
            raise RuntimeError("ResidCache must contain hook_embed. Build cache with cache_embed=True.")

        _, Bm, S, d_model = R_pre.shape

        embed_micro = E_emb.to(device=device, dtype=dtype, non_blocking=True)
        if Vmask is None:
            mask_micro = torch.ones((Bm, S), device=device, dtype=torch.float32)
        else:
            mask_micro = Vmask.to(device=device, non_blocking=True).float()

        valid_mask_chunks.append(mask_micro.bool())
        mask_bsd = mask_micro.unsqueeze(-1).to(dtype=dtype)

        embed_sum_micro = (embed_micro.float() * mask_micro.unsqueeze(-1)).sum(dim=(0, 1))
        count_micro = mask_micro.sum().float()

        if embed_sum_local is None:
            embed_sum_local = embed_sum_micro
            token_count_local = count_micro
        else:
            embed_sum_local = embed_sum_local + embed_sum_micro
            token_count_local = token_count_local + count_micro

        del embed_micro, embed_sum_micro, count_micro

        # expand FIXED cotangent
        fixed_cot_exp_micro = _expand_to_BSD(seed_vec, Bm, S, d_model, device, dtype)

        # Full chained cotangent propagation
        if do_full:
            t = fixed_cot_exp_micro.clone() * mask_bsd
            for l in range(layer_idx, -1, -1):
                skip_for_mlp_transcoder = (sae is None or "transcoder" in sae.cfg.architecture()) and l == layer_idx
                if use_mlp and not skip_for_mlp_transcoder:
                    resid_mid_l = R_mid[l].to(device=device, dtype=dtype, non_blocking=True)
                    xbar_mlp = vjp_mlp(
                        model=model,
                        layer_idx=l,
                        X=resid_mid_l,
                        out_cotangent=t,
                        use_layer_norm=use_vjp_layer_norm,
                    )
                    t = (t + xbar_mlp) * mask_bsd
                    del resid_mid_l, xbar_mlp

                if use_attn:
                    resid_pre_l = R_pre[l].to(device=device, dtype=dtype, non_blocking=True)
                    xbar_attn = vjp_attn(
                        model=model,
                        layer_idx=l,
                        X=resid_pre_l,
                        out_cotangent=t,
                        use_layer_norm=use_vjp_layer_norm,
                        attention_mask=mask_micro.bool(),
                    )
                    t = (t + xbar_attn) * mask_bsd
                    del resid_pre_l, xbar_attn

            full_cot_chunks.append(t.detach())
            del t

        # Approx fixed-cot VJP per layer
        if do_approx:
            for l in range(layer_idx, -1, -1):
                resid_pre_l = (
                    R_pre[l].to(device=device, dtype=dtype, non_blocking=True)
                    if use_attn else None
                )
                resid_mid_l = (
                    R_mid[l].to(device=device, dtype=dtype, non_blocking=True)
                    if use_mlp else None
                )

                vjp_individual_micro = torch.zeros(Bm, S, d_model, device=device, dtype=dtype)

                skip_for_mlp_transcoder = (sae is None or "transcoder" in sae.cfg.architecture()) and l == layer_idx

                if use_mlp and resid_mid_l is not None and not skip_for_mlp_transcoder:
                    xbar_mlp = vjp_mlp(
                        model=model,
                        layer_idx=l,
                        X=resid_mid_l,
                        out_cotangent=seed_vec,
                        use_layer_norm=use_vjp_layer_norm,
                    )
                    vjp_individual_micro = vjp_individual_micro + xbar_mlp
                    del xbar_mlp

                if use_attn and resid_pre_l is not None:
                    xbar_attn = vjp_attn(
                        model=model,
                        layer_idx=l,
                        X=resid_pre_l,
                        out_cotangent=seed_vec,
                        use_layer_norm=use_vjp_layer_norm,
                        attention_mask=mask_micro.bool(),
                    )
                    vjp_individual_micro = vjp_individual_micro + xbar_attn
                    del xbar_attn

                del resid_pre_l, resid_mid_l

                if l not in accumulated_vjp_per_layer:
                    accumulated_vjp_per_layer[l] = []
                accumulated_vjp_per_layer[l].append(vjp_individual_micro * mask_micro.unsqueeze(-1))

                del vjp_individual_micro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # identity term (I) at layer_idx+1
            if (layer_idx + 1) not in accumulated_vjp_per_layer:
                accumulated_vjp_per_layer[layer_idx + 1] = []
            accumulated_vjp_per_layer[layer_idx + 1].append(fixed_cot_exp_micro * mask_bsd)

        del fixed_cot_exp_micro, R_pre, R_mid, E_emb, Vmask, mask_micro, mask_bsd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # NOTE: Do NOT raise here on rank-local zero token counts. Pass zero
    # sums/counts through distributed_mean_vector, which validates the GLOBAL
    # count collectively after the all_reduce. Raising solo here would
    # desynchronize the NCCL process group.
    if embed_sum_local is None:
        d_model_fallback = getattr(model.cfg, "d_model", None)
        if d_model_fallback is None:
            raise RuntimeError("Cannot determine d_model for zero-sum fallback.")
        embed_sum_local = torch.zeros(d_model_fallback, device=device, dtype=torch.float32)
        token_count_local = torch.zeros((), device=device, dtype=torch.float32)

    # rank-global embedding mean for centering
    mean_embed_f32 = distributed_mean_vector(embed_sum_local, token_count_local)
    center_vec = mean_embed_f32.to(dtype=model.W_E.dtype).unsqueeze(0)

    valid_mask_all = torch.cat(valid_mask_chunks, dim=0)

    # Compute mean vector and tokens based on stream_transition
    if do_approx:
        # concat approx per-layer
        for l in range(layer_idx + 1, -1, -1):
            if l in accumulated_vjp_per_layer:
                accumulated_vjp_per_layer[l] = torch.cat(accumulated_vjp_per_layer[l], dim=0)

        sum_vjp_all = None
        for l in range(layer_idx + 1, -1, -1):
            if l in accumulated_vjp_per_layer:
                layer_tensor = accumulated_vjp_per_layer[l]
                sum_vjp_all = layer_tensor.clone() if sum_vjp_all is None else sum_vjp_all + layer_tensor
        if sum_vjp_all is None:
            raise RuntimeError("No per-layer VJP tensors accumulated.")

        stats = jacobian_linearity(sum_vjp_all, valid_mask_all)

        approx_mean_sum = sum_vjp_all.float().sum(dim=(0, 1))
        mean_vec = distributed_mean_vector(approx_mean_sum, token_count_local).to(dtype=model.W_E.dtype)

        # per-layer scoring
        a = mean_vec.float()
        a_norm2 = (a @ a).clamp_min(1e-12)
        for l in range(layer_idx + 1, -1, -1):
            if l not in accumulated_vjp_per_layer:
                continue
            vjp_tensor = accumulated_vjp_per_layer[l]
            mean_vec_sum = vjp_tensor.float().sum(dim=(0, 1))
            layer_mean_vec = distributed_mean_vector(mean_vec_sum, token_count_local).to(dtype=model.W_E.dtype)

            scores = apply_embed_cosine_centered(model, layer_mean_vec, center_vec=center_vec)
            (top_toks, bot_toks), (top_ids, bot_ids) = get_logits_tokens(model, scores, k)

            layer_top_tokens.append(top_toks[0])
            layer_bottom_tokens.append(bot_toks[0])
            layer_top_ids.append(top_ids[0])
            layer_bottom_ids.append(bot_ids[0])
            layer_indices.append(l)

            m = layer_mean_vec.float()
            contrib = (m @ a) / a_norm2
            layer_energy_scores.append(contrib.item())
            del scores

    elif do_full:
        if len(full_cot_chunks) == 0:
            raise RuntimeError("No full cotangent chunks accumulated.")
        sum_vjp_all_full = torch.cat(full_cot_chunks, dim=0)

        stats = jacobian_linearity(sum_vjp_all_full, valid_mask_all)

        full_mean_sum = sum_vjp_all_full.float().sum(dim=(0, 1))
        mean_vec = distributed_mean_vector(full_mean_sum, token_count_local).to(dtype=model.W_E.dtype)
    else:
        raise ValueError(f"Invalid stream_transition for VJP: {stream_transition}")

    # Apply readout (embed_cosine_centered)
    logits = apply_embed_cosine_centered(model, mean_vec, center_vec=center_vec)
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

    del valid_mask_chunks
    if do_approx:
        del accumulated_vjp_per_layer
    if do_full:
        del full_cot_chunks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

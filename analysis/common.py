#!/usr/bin/env python3
"""
Shared utilities for JVP/VJP analysis modules.

Contains common data structures (ResidChunk, ResidCache), model loading,
tensor helpers, distributed computation, token scoring, and feature
vector extraction used across all analysis variants.
"""

import os
import json
from functools import partial
from typing import List, Dict, Any, Sequence, Optional, Union, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.func import jvp as _torch_jvp
from torch.func import vjp as _torch_vjp

from sae_lens import HookedSAETransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import (
    log_info,
    Feature,
    load_hooked_transformer,
)


# =========================================================
# Resid cache (JVP variant: 3-field ResidChunk)
# =========================================================

@dataclass
class ResidChunk:
    attn: Optional[torch.Tensor]        # [L, B, S, d] or None
    mlp: Optional[torch.Tensor]         # [L, B, S, d] or None
    valid_mask: Optional[torch.Tensor]  # [B, S] bool or None


class ResidCache:
    """
    Cache of residual streams for a fixed token batch split into micro-batches.
    One cache per rank, built from local_pile_sample (rank shard).

    Additionally:
      - valid_mask marks non-pad tokens if pad_token_id available
    """

    def __init__(self, chunks: List[ResidChunk], module_type: str):
        self.chunks = chunks
        self.module_type = module_type  # "attn" / "mlp" / "all"

    def __len__(self):
        return len(self.chunks)

    def get(self, micro_step: int, kind: str = "all"):
        """
        Always returns a 3-tuple: (attn_layers, mlp_layers, valid_mask)
        with None placeholders depending on kind.
        """
        c = self.chunks[micro_step]
        if kind == "all":
            return c.attn, c.mlp, c.valid_mask
        if kind == "attn":
            return c.attn, None, c.valid_mask
        if kind == "mlp":
            return None, c.mlp, c.valid_mask
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    @torch.no_grad()
    def build_for_local_pile_sample(
        model: HookedSAETransformer,
        local_pile_sample: torch.Tensor,     # [B_local, S]
        accumulation_steps: int,
        module_type: str,                    # "attn" / "mlp" / "all"
        return_device: str = "cpu",          # "cpu" recommended
        pad_token_id: Optional[int] = None,  # if None: infer, else mask=all ones
        use_layer_norm: bool = False,
    ) -> "ResidCache":
        """
        Build cache ONCE: for each micro-batch, run forward and collect:
          - blocks.*.hook_resid_pre for all layers (if attn enabled)
          - blocks.*.hook_resid_mid for all layers (if mlp enabled)
          - valid_mask [B,S]
        """
        module_type = module_type.lower()
        hook_attn = (module_type in ("attn", "all"))
        hook_mlp  = (module_type in ("mlp", "all"))

        num_layers = getattr(model.cfg, "n_layers", len(getattr(model, "blocks", [])))

        hook_names: List[str] = []
        if hook_attn:
            hook_names.extend([f"blocks.{l}.hook_resid_pre" for l in range(num_layers)])
        if hook_mlp:
            hook_names.extend([f"blocks.{l}.hook_resid_mid" for l in range(num_layers)])

        B_local = local_pile_sample.shape[0]
        accumulation_steps = max(int(accumulation_steps), 1)

        micro_bs = max(1, B_local // accumulation_steps)
        if accumulation_steps == 1:
            micro_bs = B_local

        # infer pad_token_id if not given
        if pad_token_id is None:
            tok = getattr(model, "tokenizer", None)
            pad_token_id = getattr(tok, "pad_token_id", None) if tok is not None else None

        chunks: List[ResidChunk] = []

        for micro_step in range(accumulation_steps):
            ms = micro_step * micro_bs
            me = min((micro_step + 1) * micro_bs, B_local)
            if ms >= B_local:
                break

            tokens = local_pile_sample[ms:me]  # [B_micro, S]
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

            # valid mask
            if pad_token_id is None:
                valid_mask = torch.ones(tokens.shape, device=tokens.device, dtype=torch.bool)
            else:
                valid_mask = (tokens != pad_token_id)

            attn_layers = None
            mlp_layers = None

            if hook_attn:
                attn_list = []
                for l in range(num_layers):
                    x = cache[f"blocks.{l}.hook_resid_pre"]  # [B,S,d]
                    if not use_layer_norm:
                        x = getattr(model.blocks[l], "ln1")(x)
                    attn_list.append(x)
                attn_layers = torch.stack(attn_list, dim=0)  # [L,B,S,d]

            if hook_mlp:
                mlp_list = []
                for l in range(num_layers):
                    x = cache[f"blocks.{l}.hook_resid_mid"]  # [B,S,d]
                    if not use_layer_norm:
                        x = getattr(model.blocks[l], "ln2")(x)
                    mlp_list.append(x)
                mlp_layers = torch.stack(mlp_list, dim=0)    # [L,B,S,d]

            def _mv(t):
                if t is None:
                    return None
                return t.detach().to(return_device)

            chunks.append(
                ResidChunk(
                    attn=_mv(attn_layers),
                    mlp=_mv(mlp_layers),
                    valid_mask=_mv(valid_mask),
                )
            )

            del cache, attn_layers, mlp_layers, valid_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ResidCache(chunks=chunks, module_type=module_type)


# =========================================================
# Resid cache (VJP variant: 4-field ResidChunk with embed)
# =========================================================

@dataclass
class VJPResidChunk:
    attn: Optional[torch.Tensor]       # [L, B, S, d] or None  (resid_pre)
    mlp: Optional[torch.Tensor]        # [L, B, S, d] or None  (resid_mid)
    embed: Optional[torch.Tensor]      # [B, S, d] or None     (hook_embed)
    valid_mask: Optional[torch.Tensor] # [B, S] bool or None


class VJPResidCache:
    """
    Cache of residual streams for VJP analysis, split into micro-batches.

    VJP primal aligned with JVP:
      - attn cache stores blocks.*.hook_resid_pre
      - mlp  cache stores blocks.*.hook_resid_mid

    Additionally:
      - embed cache stores hook_embed (token embeddings)
      - valid_mask marks non-pad tokens if pad_token_id available
    """

    def __init__(self, chunks: List[VJPResidChunk], module_type: str):
        self.chunks = chunks
        self.module_type = module_type  # "attn" / "mlp" / "all"

    def __len__(self):
        return len(self.chunks)

    def get(self, micro_step: int, kind: str = "all"):
        c = self.chunks[micro_step]
        if kind == "all":
            return c.attn, c.mlp, c.embed, c.valid_mask
        if kind == "attn":
            return c.attn, None, c.embed, c.valid_mask
        if kind == "mlp":
            return None, c.mlp, c.embed, c.valid_mask
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    @torch.no_grad()
    def build_for_local_pile_sample(
        model: HookedSAETransformer,
        local_pile_sample: torch.Tensor,     # [B_local, S]
        accumulation_steps: int,
        module_type: str,                    # "attn" / "mlp" / "all"
        return_device: str = "cpu",
        cache_embed: bool = True,
        pad_token_id: Optional[int] = None,
        use_layer_norm: bool = False,
    ) -> "VJPResidCache":
        module_type = module_type.lower()
        hook_attn = (module_type in ("attn", "all"))
        hook_mlp  = (module_type in ("mlp", "all"))

        num_layers = getattr(model.cfg, "n_layers", len(getattr(model, "blocks", [])))

        hook_names: List[str] = []
        if cache_embed:
            hook_names.append("hook_embed")
        if hook_attn:
            hook_names.extend([f"blocks.{l}.hook_resid_pre" for l in range(num_layers)])
        if hook_mlp:
            hook_names.extend([f"blocks.{l}.hook_resid_mid" for l in range(num_layers)])

        B_local = local_pile_sample.shape[0]
        accumulation_steps = max(int(accumulation_steps), 1)

        micro_bs = max(1, B_local // accumulation_steps)
        if accumulation_steps == 1:
            micro_bs = B_local

        if pad_token_id is None:
            tok = getattr(model, "tokenizer", None)
            pad_token_id = getattr(tok, "pad_token_id", None) if tok is not None else None

        chunks: List[VJPResidChunk] = []

        for micro_step in range(accumulation_steps):
            ms = micro_step * micro_bs
            me = min((micro_step + 1) * micro_bs, B_local)
            if ms >= B_local:
                break

            tokens = local_pile_sample[ms:me]
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

            embed = None
            valid_mask = None
            if cache_embed:
                embed = cache["hook_embed"]
                if pad_token_id is None:
                    valid_mask = torch.ones(tokens.shape, device=tokens.device, dtype=torch.bool)
                else:
                    valid_mask = (tokens != pad_token_id)

            attn_layers = None
            mlp_layers = None

            if hook_attn:
                attn_list = []
                for l in range(num_layers):
                    x = cache[f"blocks.{l}.hook_resid_pre"]
                    if not use_layer_norm:
                        x = model.blocks[l].ln1(x)
                    attn_list.append(x)
                attn_layers = torch.stack(attn_list, dim=0)

            if hook_mlp:
                mlp_list = []
                for l in range(num_layers):
                    x = cache[f"blocks.{l}.hook_resid_mid"]
                    if not use_layer_norm:
                        x = model.blocks[l].ln2(x)
                    mlp_list.append(x)
                mlp_layers = torch.stack(mlp_list, dim=0)

            def _mv(t):
                if t is None:
                    return None
                return t.detach().to(return_device)

            chunks.append(
                VJPResidChunk(
                    attn=_mv(attn_layers),
                    mlp=_mv(mlp_layers),
                    embed=_mv(embed),
                    valid_mask=_mv(valid_mask),
                )
            )

            del cache, attn_layers, mlp_layers, embed, valid_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return VJPResidCache(chunks=chunks, module_type=module_type)


# =========================================================
# Model loading
# =========================================================

def load_model(
    device: str,
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    *,
    hf_weights_override: Optional[str] = None,
) -> HookedSAETransformer:
    """Load a HookedSAETransformer for analysis (no-processing variant).

    If ``model_name`` is unknown to transformer_lens, an auto-fallback tries
    to toggle the ``-Base`` suffix and load HF weights from ``model_name``
    onto the sibling config. Pass ``hf_weights_override`` to force the
    weights source explicitly.
    """
    model = load_hooked_transformer(
        model_name,
        device=device,
        dtype=dtype,
        fold_processing=False,
        hf_weights_override=hf_weights_override,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


# =========================================================
# Tensor helpers
# =========================================================

def _expand_to_BSD(vec: torch.Tensor, B: int, S: int, d_model: int, device, dtype) -> torch.Tensor:
    if vec.ndim == 1:
        assert vec.shape[0] == d_model
        return vec.to(device=device, dtype=dtype).view(1, 1, d_model).expand(B, S, d_model)
    if vec.ndim == 2:
        assert vec.shape == (S, d_model)
        return vec.to(device=device, dtype=dtype).view(1, S, d_model).expand(B, S, d_model)
    assert vec.shape == (B, S, d_model)
    return vec.to(device=device, dtype=dtype)


# =========================================================
# JVP primitives
# =========================================================

def jvp_mlp(
    model: "HookedSAETransformer",
    layer_idx: int,
    X: torch.Tensor,          # [B, S, d]
    f_vec: torch.Tensor,      # [d] or [S,d] or [B,S,d]
    return_output: bool = False,
    use_layer_norm: bool = False,
):
    mlp = model.blocks[layer_idx].mlp
    ln2 = model.blocks[layer_idx].ln2
    ln2_post = model.blocks[layer_idx].ln2_post if hasattr(model.blocks[layer_idx], "ln2_post") else None
    mlp.eval()
    if ln2 is not None:
        ln2.eval()
    if ln2_post is not None:
        ln2_post.eval()

    assert X.ndim == 3, f"X must be [B, S, d], got {X.shape}"
    B, S, d_model = X.shape
    device = next(mlp.parameters()).device
    dtype = next(mlp.parameters()).dtype
    assert X.device == device and X.dtype == dtype, f"X must be on device {device} dtype {dtype}, got {X.device} {X.dtype}"

    F_tan = _expand_to_BSD(f_vec, B, S, d_model, device, dtype)

    if use_layer_norm:
        assert ln2 is not None, "This model block has no ln2."
        def f_whole(x):
            return ln2_post(mlp(ln2(x))) if ln2_post is not None else mlp(ln2(x))
    else:
        def f_whole(x):
            return mlp(x)

    y_all, jvp_all = _torch_jvp(f_whole, (X,), (F_tan,), strict=True)
    return (y_all, jvp_all) if return_output else jvp_all


def jvp_attn(
    model: "HookedSAETransformer",
    layer_idx: int,
    X: torch.Tensor,          # [B, S, d]
    f_vec: torch.Tensor,      # [d] or [S,d] or [B,S,d]
    return_output: bool = False,
    use_layer_norm: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
):
    attn = model.blocks[layer_idx].attn
    ln1 = model.blocks[layer_idx].ln1
    ln1_post = model.blocks[layer_idx].ln1_post if hasattr(model.blocks[layer_idx], "ln1_post") else None
    attn.eval()
    if ln1 is not None:
        ln1.eval()
    if ln1_post is not None:
        ln1_post.eval()

    assert X.ndim == 3, f"X must be [B, S, d], got {X.shape}"
    B, S, d_model = X.shape
    device = next(attn.parameters()).device
    dtype = next(attn.parameters()).dtype
    assert X.device == device and X.dtype == dtype, f"X must be on device {device} dtype {dtype}, got {X.device} {X.dtype}"

    F_tan = _expand_to_BSD(f_vec, B, S, d_model, device, dtype)

    if use_layer_norm:
        assert ln1 is not None, "This model block has no ln1."
        def g_whole(x):
            x_ln = ln1(x)
            return ln1_post(attn(x_ln, x_ln, x_ln, attention_mask=attention_mask)) if ln1_post is not None else attn(x_ln, x_ln, x_ln, attention_mask=attention_mask)
    else:
        def g_whole(x):
            return attn(x, x, x, attention_mask=attention_mask)

    y_all, jvp_all = _torch_jvp(g_whole, (X,), (F_tan,), strict=True)
    return (y_all, jvp_all) if return_output else jvp_all


# =========================================================
# VJP primitives
# =========================================================

def vjp_mlp(
    model: HookedSAETransformer,
    layer_idx: int,
    X: torch.Tensor,             # [B,S,d]  (primal: resid_mid)
    out_cotangent: torch.Tensor, # [d] or [S,d] or [B,S,d]
    use_layer_norm: bool = False,
):
    mlp = model.blocks[layer_idx].mlp
    ln2 = model.blocks[layer_idx].ln2
    ln2_post = model.blocks[layer_idx].ln2_post if hasattr(model.blocks[layer_idx], "ln2_post") else None
    mlp.eval()
    if ln2 is not None:
        ln2.eval()
    if ln2_post is not None:
        ln2_post.eval()

    B, S, d_model = X.shape
    device = next(mlp.parameters()).device
    dtype  = next(mlp.parameters()).dtype
    assert X.device == device and X.dtype == dtype

    Ybar = _expand_to_BSD(out_cotangent, B, S, d_model, device, dtype)

    if use_layer_norm:
        assert ln2 is not None, "This model block has no ln2."
        def f_whole(x):
            return ln2_post(mlp(ln2(x))) if ln2_post is not None else mlp(ln2(x))
    else:
        def f_whole(x):
            return mlp(x)

    _, pullback = _torch_vjp(f_whole, X)
    (Xbar,) = pullback(Ybar)
    return Xbar  # [B,S,d]


def vjp_attn(
    model: HookedSAETransformer,
    layer_idx: int,
    X: torch.Tensor,             # [B,S,d]  (primal: resid_pre)
    out_cotangent: torch.Tensor, # [d] or [S,d] or [B,S,d]
    use_layer_norm: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
):
    attn = model.blocks[layer_idx].attn
    ln1  = model.blocks[layer_idx].ln1
    ln1_post = model.blocks[layer_idx].ln1_post if hasattr(model.blocks[layer_idx], "ln1_post") else None
    attn.eval()
    if ln1 is not None:
        ln1.eval()
    if ln1_post is not None:
        ln1_post.eval()

    B, S, d_model = X.shape
    device = next(attn.parameters()).device
    dtype  = next(attn.parameters()).dtype
    assert X.device == device and X.dtype == dtype

    Ybar = _expand_to_BSD(out_cotangent, B, S, d_model, device, dtype)

    if use_layer_norm:
        assert ln1 is not None, "This model block has no ln1."
        def g_whole(x):
            x_ln = ln1(x)
            return ln1_post(attn(x_ln, x_ln, x_ln, attention_mask=attention_mask)) if ln1_post is not None else attn(x_ln, x_ln, x_ln, attention_mask=attention_mask)
    else:
        def g_whole(x):
            return attn(x, x, x, attention_mask=attention_mask)

    _, pullback = _torch_vjp(g_whole, X)
    (Xbar,) = pullback(Ybar)
    return Xbar  # [B,S,d]


# =========================================================
# Feature vector extraction
# =========================================================

def get_feature_vector(model: HookedSAETransformer, f: Feature, sae=None) -> torch.Tensor:
    if sae is None:
        return model.blocks[f.layer].mlp.W_out[f.feature]
    sae_w_dec = sae.W_dec
    if sae_w_dec.dtype != model.W_U.dtype:
        sae_w_dec = sae_w_dec.to(model.W_U.dtype)
    return sae_w_dec[f.feature]


def get_feature_input_vector(model: HookedSAETransformer, f: Feature, sae=None) -> torch.Tensor:
    if sae is None:
        W_in = model.blocks[f.layer].mlp.W_in   # [d_model, d_mlp]
        return W_in[:, f.feature]               # [d_model]

    W_enc = sae.W_enc                           # [d_model, d_sae]
    v = W_enc[:, f.feature]                     # [d_model]
    if v.dtype != model.W_E.dtype:
        v = v.to(model.W_E.dtype)
    return v


# =========================================================
# Embed functions (VJP-specific)
# =========================================================

def apply_embed(model: HookedSAETransformer, vec: torch.Tensor) -> torch.Tensor:
    if vec.dtype != model.W_E.dtype:
        vec = vec.to(model.W_E.dtype)
    return vec @ model.W_E.T


def apply_embed_cosine_centered(
    model: HookedSAETransformer,
    vec: torch.Tensor,
    center_vec: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    W_E = model.W_E  # [V, d]
    if vec.dtype != W_E.dtype:
        vec = vec.to(W_E.dtype)

    center_vec = W_E.mean(dim=0, keepdim=True) if center_vec is None else center_vec  # [1,d]
    W_Ec = W_E - center_vec  # [V,d]

    emb_norms = W_Ec.norm(p=2, dim=1).clamp_min(eps)  # [V]
    W_Ec = W_Ec / emb_norms.unsqueeze(1)

    dots = vec @ W_Ec.T  # [V]
    return dots


# =========================================================
# Distributed computation
# =========================================================

def distributed_mean_vector(vec: torch.Tensor, count: Optional[Union[int, float, torch.Tensor]] = None) -> torch.Tensor:
    """
    If count is provided, vec is treated as SUM and we divide by global SUM(count).

    NOTE: Validation of `count` is done AFTER the all-reduce so the check is a
    collective decision — every rank sees the same global count and either all
    raise or all proceed. Raising before all_reduce on a subset of ranks would
    desynchronize the NCCL process group and hang the surviving ranks until
    the watchdog fires.
    """
    if count is not None:
        if torch.is_tensor(count):
            count_tensor = count.to(device=vec.device, dtype=vec.dtype).clone()
        else:
            count_tensor = torch.tensor(count, device=vec.device, dtype=vec.dtype)
    else:
        count_tensor = None

    if not dist.is_initialized():
        if count_tensor is not None:
            if torch.any(count_tensor <= 0):
                raise ValueError(f"Invalid count for distributed mean: {count}")
            return vec / count_tensor
        return vec

    reduced = vec.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    if count_tensor is None:
        reduced /= float(dist.get_world_size())
        return reduced

    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    if torch.any(count_tensor <= 0):
        raise ValueError(f"Invalid global count for distributed mean: {count_tensor.tolist()}")
    return reduced / count_tensor


# =========================================================
# Unembed / logits / token scoring
# =========================================================

def apply_unembed(model, vec: torch.Tensor, use_ln_final: bool = True) -> torch.Tensor:
    if vec.dtype != model.cfg.dtype:
        vec = vec.to(model.cfg.dtype)
    mid = model.ln_final(vec) if use_ln_final else vec
    return model.unembed(mid)


def get_logits_tokens(model_wrapper, logits: torch.Tensor, k: int = 10):
    model = model_wrapper if isinstance(model_wrapper, HookedSAETransformer) else model_wrapper.m
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    N, V = logits.shape
    k = min(k, V)

    topk = logits.topk(k, dim=-1)
    bottomk = logits.topk(k, dim=-1, largest=False)

    top_tokens = [model.to_str_tokens(topk.indices[i]) for i in range(N)]
    bottom_tokens = [model.to_str_tokens(bottomk.indices[i]) for i in range(N)]

    top_ids = [topk.indices[i].cpu().tolist() for i in range(N)]
    bottom_ids = [bottomk.indices[i].cpu().tolist() for i in range(N)]
    return (top_tokens, bottom_tokens), (top_ids, bottom_ids)


# =========================================================
# Token-change (positive) utilities
# =========================================================

def set_feature_act_hook(act: torch.Tensor, hook: Any, feature: int, value: float) -> torch.Tensor:
    act[:, :, feature] = value
    return act


@torch.no_grad()
def compute_token_change(
    model: HookedSAETransformer,
    prompts: Union[torch.Tensor, Sequence[str]],
    f: Feature,
    sae,
    value: float,
    invert_sign: bool = False,
) -> torch.Tensor:
    """
    Returns mean over batch+seq of logit differences, shape [V].

    When invert_sign=False (default): returns mean(intervened_logits - clean_logits).
    When invert_sign=True: returns mean(clean_logits - intervened_logits).
    The invert_sign=True mode is used by zero_out to measure feature contribution
    (positive values = tokens the feature promotes).
    """
    def _mean_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            return logits.mean(dim=(0, 1))
        if logits.ndim == 2:
            return logits.mean(dim=0)
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    if sae is None:
        clean_logits = model(prompts)
        pos_logits = model.run_with_hooks(
            prompts,
            fwd_hooks=[(f"blocks.{f.layer}.mlp.hook_post", partial(set_feature_act_hook, feature=f.feature, value=value))],
        )
    else:
        # For transcoder SAEs, sae.cfg.metadata.hook_name is the MLP *input*
        # (e.g. blocks.L.mlp.hook_in) but sae_lens installs hook_sae_acts_post
        # under the SAE's *output* attach point (blocks.L.hook_mlp_out). For
        # standard resid SAEs these two are the same. Mirrors the helper in
        # data/write_feature_sample.py:_sae_output_hook_name.
        sae_output_hook = getattr(sae.cfg.metadata, "hook_name_out", None) or sae.cfg.metadata.hook_name
        hook_name = f"{sae_output_hook}.hook_sae_acts_post"
        clean_logits = model.run_with_saes(prompts, saes=[sae])
        pos_logits = model.run_with_hooks_with_saes(
            prompts,
            saes=[sae],
            fwd_hooks=[(hook_name, partial(set_feature_act_hook, feature=f.feature, value=value))],
        )

    pos_diff_logits = _mean_logits(clean_logits - pos_logits) if invert_sign else _mean_logits(pos_logits - clean_logits)
    return pos_diff_logits


# =========================================================
# Reference token changes
# =========================================================

def _maybe_int_index(idx: Any) -> Optional[int]:
    if isinstance(idx, int):
        return idx
    if isinstance(idx, str) and idx.isdigit():
        return int(idx)
    return None


def load_reference_token_changes(jsonl_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    If compute_token_change_enabled is False, we can reuse previously computed
    positive token-change tokens from a reference JSONL.
    """
    ref = {"by_str": {}, "by_idx": {}}
    if not jsonl_path:
        return ref
    if not os.path.exists(jsonl_path):
        log_info(f"Reference results JSONL not found: {jsonl_path}")
        return ref

    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                feature_str = entry.get("feature_str")
                feature_idx = _maybe_int_index(entry.get("feature_index"))
                token_info = {
                    "positive_top_tokens": entry.get("positive_top_tokens"),
                    "positive_top_ids": entry.get("positive_top_ids"),
                    "positive_bottom_tokens": entry.get("positive_bottom_tokens"),
                    "positive_bottom_ids": entry.get("positive_bottom_ids"),
                }
                if feature_str is not None:
                    ref["by_str"][feature_str] = token_info
                if feature_idx is not None:
                    ref["by_idx"][feature_idx] = token_info
        log_info(f"Loaded {len(ref['by_str'])} reference token-change entries from {jsonl_path}")
    except Exception as e:
        log_info(f"Failed to load reference results from {jsonl_path}: {e}")

    return ref


# =========================================================
# Jacobian linearity statistics
# =========================================================

@torch.no_grad()
def _flatten_valid_vectors(
    V: torch.Tensor,                      # [B,S,d]
    valid_mask: Optional[torch.Tensor],   # [B,S] bool or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert V.ndim == 3
    B, S, d = V.shape
    device = V.device

    if valid_mask is None:
        M = torch.ones((B, S), device=device, dtype=torch.bool)
    else:
        M = valid_mask.to(device=device, dtype=torch.bool)

    Vf = V.reshape(B * S, d)
    Mf = M.reshape(B * S)

    Vf = Vf[Mf]  # [N,d]
    N = torch.tensor(float(Vf.shape[0]), device=device)
    return Vf, N


def jacobian_linearity(
    V: torch.Tensor,                      # [B,S,d]
    valid_mask: Optional[torch.Tensor],   # [B,S] bool or None
    eps: float = 1e-12,
    energy_eps: float = 1e-12,            # filter near-zero ||v||^2
) -> Dict[str, float]:
    """
    Energy-weighted explained energy by the global mean direction + relative variance.

    Steps
      1) mu = E[v],  mhat = mu / ||mu||
      2) EV_mu      =  sum_i (v_i . mhat)^2  /  sum_i ||v_i||^2
      3) rel_var    =  (E||v - mu||^2) / (E||v||^2)
                   =  1 - (||mu||^2 / E||v||^2)

    Returns dict with N, N_used, E_mu, sum_energy, mean_energy,
    sum_parallel2, EV_mu, perp_EV, rel_var.
    """
    # IMPORTANT: Do NOT return early on a rank-local decision (N_local,
    # N_used_local) before calling the collectives below. Doing so causes
    # one rank to skip allreduces while the other rank blocks on them,
    # hanging the NCCL process group until the watchdog fires. The early
    # returns were the root cause of the query_lens_key 2-GPU hang
    # observed at feature 115 of the qwen3 run — for feature-specific VJP
    # vectors, one rank's local norms fell under energy_eps while the
    # other's didn't. All collectives below must be called from every rank.
    Vf, N_local = _flatten_valid_vectors(V, valid_mask)
    device = V.device
    d = V.shape[-1]

    if Vf.numel() == 0:
        # Rank-local sums are zero-valued; still participate in collectives.
        sum_v_local = torch.zeros(d, device=device, dtype=V.dtype)
    else:
        sum_v_local = Vf.sum(dim=0)
    N_global = N_local.clone()
    sum_v_global = sum_v_local.clone()

    if dist.is_initialized():
        dist.all_reduce(sum_v_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(N_global, op=dist.ReduceOp.SUM)

    N = float(N_global.item())
    if N <= 0:
        # Globally empty — safe to return on all ranks.
        return {
            "N": 0.0, "N_used": 0.0, "E_mu": 0.0,
            "sum_energy": 0.0, "mean_energy": 0.0, "sum_parallel2": 0.0,
            "EV_mu": 0.0, "perp_EV": 0.0, "rel_var": 0.0,
        }
    mu = sum_v_global / max(N, 1.0)
    mu_norm = mu.norm().clamp_min(eps)
    mhat = mu / mu_norm
    E_mu = float((mu_norm * mu_norm).item())

    if Vf.numel() == 0:
        N_used_local = torch.tensor(0.0, device=device)
        sum_energy = torch.tensor(0.0, device=device)
        sum_parallel2 = torch.tensor(0.0, device=device)
    else:
        v_norm2_local_all = (Vf * Vf).sum(dim=-1)
        used_mask = v_norm2_local_all > energy_eps
        Vf_used = Vf[used_mask]

        N_used_local = torch.tensor(float(used_mask.sum().item()), device=device)

        if Vf_used.numel() == 0:
            sum_energy = torch.tensor(0.0, device=device)
            sum_parallel2 = torch.tensor(0.0, device=device)
        else:
            energy_local = (Vf_used * Vf_used).sum(dim=-1)
            dot_local = (Vf_used @ mhat)
            parallel2_local = dot_local * dot_local
            sum_energy = energy_local.sum()
            sum_parallel2 = parallel2_local.sum()

    if dist.is_initialized():
        dist.all_reduce(sum_energy, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_parallel2, op=dist.ReduceOp.SUM)
        dist.all_reduce(N_used_local, op=dist.ReduceOp.SUM)

    if float(N_used_local.item()) <= 0:
        return {
            "N": float(N), "N_used": 0.0, "E_mu": E_mu,
            "sum_energy": 0.0, "mean_energy": 0.0, "sum_parallel2": 0.0,
            "EV_mu": 0.0, "perp_EV": 0.0, "rel_var": 0.0,
        }

    N_used = float(N_used_local.item())
    sum_energy_f = float(sum_energy.item())
    denom = sum_energy_f + eps

    EV = float(sum_parallel2.item()) / denom
    EV = max(0.0, min(1.0, EV))
    perp_EV = 1.0 - EV

    mean_energy = sum_energy_f / max(N_used, 1.0)
    rel_var = 1.0 - (E_mu / (mean_energy + eps))
    rel_var = max(0.0, min(1.0, rel_var))

    return {
        "N": float(N), "N_used": float(N_used), "E_mu": E_mu,
        "sum_energy": sum_energy_f, "mean_energy": float(mean_energy),
        "sum_parallel2": float(sum_parallel2.item()),
        "EV_mu": EV, "perp_EV": perp_EV, "rel_var": rel_var,
    }


# =========================================================
# Baseline registry
# =========================================================

BASELINE_REGISTRY = {
    "logit_lens_value":        {"feature_vector": "value", "stream_transition": "identity",     "readout": "unembed"},
    "logit_lens_key":          {"feature_vector": "key",   "stream_transition": "identity",     "readout": "embed"},
    "query_lens_value_approx": {"feature_vector": "value", "stream_transition": "first_order",  "readout": "unembed"},
    "query_lens_key_approx":   {"feature_vector": "key",   "stream_transition": "first_order",  "readout": "embed_cosine_centered"},
    "query_lens_value":  {"feature_vector": "value", "stream_transition": "full",        "readout": "unembed"},
    "query_lens_key":    {"feature_vector": "key",   "stream_transition": "full",        "readout": "embed_cosine_centered"},
    "token_change":            {"feature_vector": "value", "stream_transition": "token_change",  "readout": None},
    "zero_out":                {"feature_vector": "value", "stream_transition": "zero_out",       "readout": None},
    "tuned_lens":              {"feature_vector": "value", "stream_transition": "tuned_lens",    "readout": "unembed"},
}


def resolve_baseline(cfg) -> dict:
    """Resolve baseline config from preset name or individual components."""
    baseline_name = getattr(cfg, "baseline", None)
    if baseline_name:
        if baseline_name not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline: {baseline_name}. Valid: {list(BASELINE_REGISTRY.keys())}")
        return {**BASELINE_REGISTRY[baseline_name], "baseline": baseline_name}
    return {
        "feature_vector": cfg.feature_vector,
        "stream_transition": cfg.stream_transition,
        "readout": cfg.readout,
        "baseline": f"{cfg.feature_vector}_{cfg.stream_transition}_{cfg.readout}",
    }

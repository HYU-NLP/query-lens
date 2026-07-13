#!/usr/bin/env python3
"""
Unified analysis entry point.

Replaces jvp.py / vjp.py as CLI scripts.
Each run produces ONE token set for ONE baseline, configured via Hydra YAML.

Usage:
    torchrun --nproc_per_node=N analysis/run.py -cn config_name baseline=query_lens_value_approx
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sae_lens import HookedSAETransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import (
    log_info,
    set_seed,
    setup_distributed,
    cleanup_distributed,
    distribute_batch,
    read_pkl_file,
    append_result_to_jsonl,
    load_processed_features,
    SAEGPUCache,
    load_pile_sample,
    Feature,
    Activation,
    sort_features_by_layer,
    parse_dtype,
)

from analysis.common import (
    ResidCache, VJPResidCache,
    load_model,
    get_feature_vector, get_feature_input_vector,
    apply_unembed, apply_embed, apply_embed_cosine_centered,
    get_logits_tokens,
    distributed_mean_vector,
    compute_token_change,
    resolve_baseline,
)
from analysis.jvp import analyze_feature as jvp_analyze_feature
from analysis.vjp import analyze_feature as vjp_analyze_feature


# =========================================================
# Tuned lens utilities
# =========================================================

def _load_params_state(params_path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not params_path:
        return None
    if not os.path.exists(params_path):
        log_info(f"params_path not found: {params_path}")
        return None
    params_obj = torch.load(params_path, map_location="cpu")
    if isinstance(params_obj, dict):
        return params_obj
    try:
        return dict(params_obj)
    except Exception as e:
        log_info(f"Failed to parse params file: {params_path} ({e})")
        return None


def apply_layer_params_to_feature(
    f_dir: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    params_state: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if layer_idx == num_layers - 1:
        return f_dir.to(device=device, dtype=dtype)
    if not params_state:
        return f_dir.to(device=device, dtype=dtype)

    w_key = f"{layer_idx}.weight"
    b_key = f"{layer_idx}.bias"
    if w_key not in params_state or b_key not in params_state:
        return f_dir.to(device=device, dtype=dtype)

    x = f_dir.to(device=device, dtype=dtype)
    W = params_state[w_key].to(device=device, dtype=dtype)
    b = params_state[b_key].to(device=device, dtype=dtype)

    # Tuned lens translators are residual: output = h + translator(h)
    # See: https://arxiv.org/abs/2303.08112
    return x + F.linear(x, W, b)


# =========================================================
# Simple baselines (no ResidCache needed)
# =========================================================

def analyze_identity(model, f, sae, feature_vector, readout, k, use_unembed_layer_norm):
    """identity: feature_vector -> readout -> tokens. No ResidCache needed."""
    if feature_vector == "value":
        vec = get_feature_vector(model, f, sae)
    else:
        vec = get_feature_input_vector(model, f, sae)

    if readout == "unembed":
        logits = apply_unembed(model, vec, use_ln_final=use_unembed_layer_norm).view(-1)
    elif readout == "embed":
        logits = apply_embed(model, vec)
    elif readout == "embed_cosine_centered":
        logits = apply_embed_cosine_centered(model, vec)
    else:
        raise ValueError(f"Unknown readout for identity: {readout}")

    (top_toks, bot_toks), (top_ids, bot_ids) = get_logits_tokens(model, logits, k)
    return {
        "layer": int(f.layer),
        "top_tokens": top_toks[0],
        "bottom_tokens": bot_toks[0],
        "top_ids": top_ids[0],
        "bottom_ids": bot_ids[0],
    }


def analyze_token_change(
    model, f, sae, pile_sample, rank, world_size,
    cfg, resid_cache, k, invert_sign=False,
):
    """token_change / zero_out: micro-batch compute_token_change -> distributed_mean -> get_logits_tokens."""
    batch_size = pile_sample.shape[0]
    start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
    local_pile_sample = pile_sample[start_idx:end_idx]
    local_batch_size = local_pile_sample.shape[0]

    token_change_value = getattr(cfg, "token_change_value", 10.0)
    accumulation_steps = max(int(getattr(cfg, "accumulation_steps", 1)), 1)
    micro_batch_size = max(1, local_batch_size // accumulation_steps)
    if accumulation_steps == 1:
        micro_batch_size = local_batch_size

    device = model.cfg.device

    token_change_logits_sum = None
    token_change_token_count = 0

    for micro_step in range(accumulation_steps):
        micro_start = micro_step * micro_batch_size
        micro_end = min((micro_step + 1) * micro_batch_size, local_batch_size)
        if micro_start >= local_batch_size:
            break

        micro_batch = local_pile_sample[micro_start:micro_end]
        micro_batch_size_actual = micro_batch.shape[0]

        # Get valid mask from resid_cache for token counting
        if resid_cache is not None:
            _, _, Vmask = resid_cache.get(micro_step, kind="all")
            if Vmask is None:
                S = micro_batch.shape[1]
                mask_micro = torch.ones((micro_batch_size_actual, S), device=device, dtype=torch.float32)
            else:
                mask_micro = Vmask.to(device=device, non_blocking=True).float()
        else:
            S = micro_batch.shape[1]
            mask_micro = torch.ones((micro_batch_size_actual, S), device=device, dtype=torch.float32)

        valid_tokens_micro = int(mask_micro.sum().float().item())

        if valid_tokens_micro > 0:
            diff_logits_micro = compute_token_change(
                model=model,
                prompts=micro_batch,
                f=f,
                sae=sae,
                value=0.0 if invert_sign else token_change_value,
                invert_sign=invert_sign,
            )
            weighted = diff_logits_micro * float(valid_tokens_micro)
            token_change_logits_sum = weighted if token_change_logits_sum is None else token_change_logits_sum + weighted
            token_change_token_count += valid_tokens_micro

    if token_change_token_count > 0 and token_change_logits_sum is not None:
        token_change_logits_mean = distributed_mean_vector(token_change_logits_sum, token_change_token_count)
        (tok_tuple, id_tuple) = get_logits_tokens(model, token_change_logits_mean, k)
        return {
            "layer": int(f.layer),
            "top_tokens": tok_tuple[0][0],
            "bottom_tokens": tok_tuple[1][0],
            "top_ids": id_tuple[0][0],
            "bottom_ids": id_tuple[1][0],
        }

    return None


def analyze_tuned_lens(model, f, sae, k, use_unembed_layer_norm, params_state):
    """tuned_lens: value + learned params -> unembed -> tokens."""
    vec = get_feature_vector(model, f, sae)
    vec = apply_layer_params_to_feature(
        f_dir=vec,
        layer_idx=int(f.layer),
        num_layers=model.cfg.n_layers,
        params_state=params_state,
        device=torch.device(model.cfg.device),
        dtype=model.cfg.dtype,
    )
    logits = apply_unembed(model, vec, use_ln_final=use_unembed_layer_norm).view(-1)
    (top_toks, bot_toks), (top_ids, bot_ids) = get_logits_tokens(model, logits, k)
    return {
        "layer": int(f.layer),
        "top_tokens": top_toks[0],
        "bottom_tokens": bot_toks[0],
        "top_ids": top_ids[0],
        "bottom_ids": bot_ids[0],
    }


# =========================================================
# Main
# =========================================================

def main(cfg: DictConfig):
    if cfg.features_pkl is None:
        log_info("Error: features_pkl is required.")
        sys.exit(1)

    bl = resolve_baseline(cfg)
    fv = bl["feature_vector"]
    st = bl["stream_transition"]
    ro = bl["readout"]
    baseline_name = bl["baseline"]

    base_logs_dir = cfg.baseline if cfg.baseline else "base"
    features_base = os.path.splitext(os.path.basename(cfg.features_pkl))[0]
    logs_dir = os.path.join("experiments", features_base, base_logs_dir)

    log_info(f"Baseline: {baseline_name} (fv={fv}, st={st}, ro={ro})")
    log_info(f"Setting random seed: {cfg.seed}")
    set_seed(cfg.seed)

    dtype = parse_dtype(cfg.dtype)
    log_info(f"Using dtype: {cfg.dtype} ({dtype})")

    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    os.makedirs(logs_dir, exist_ok=True)
    log_info(f"Using device: {device}")
    log_info(f"Distributed: rank={rank}, world_size={world_size}")

    log_info(f"Loading model: {cfg.model}...")
    model = load_model(device, cfg.model, dtype=dtype)
    log_info(f"Loaded {cfg.model}")

    if not os.path.exists(cfg.features_pkl):
        log_info(f"Missing {cfg.features_pkl}")
        sys.exit(1)

    all_features_sample = read_pkl_file(cfg.features_pkl)
    all_features_sample = sort_features_by_layer(all_features_sample)
    log_info(f"Loaded {len(all_features_sample)} features")

    # Load pile sample (needed for first_order, full, token_change)
    pile_sample = None
    if st in ("first_order", "full", "token_change", "zero_out"):
        log_info(f"Loading Pile sample with num_samples={cfg.num_samples}, max_length={cfg.max_length}...")
        pile_sample = load_pile_sample(
            model, device,
            num_samples=cfg.num_samples,
            max_length=cfg.max_length,
            seed=cfg.seed,
        )

    # Build ResidCache based on needs
    resid_cache = None
    if st in ("first_order", "full") and pile_sample is not None:
        batch_size = pile_sample.shape[0]
        start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
        local_pile_sample = pile_sample[start_idx:end_idx]

        sub = cfg.submodule.lower()
        cache_kind = {"both": "all", "attn": "attn", "mlp": "mlp"}.get(sub, "all")
        pad_token_id = getattr(cfg, "pad_token_id", None)

        if fv == "key":
            log_info(f"[rank {rank}] Building VJP resid cache (kind={cache_kind}, device={cfg.resid_device}) ...")
            resid_cache = VJPResidCache.build_for_local_pile_sample(
                model=model,
                local_pile_sample=local_pile_sample,
                accumulation_steps=cfg.accumulation_steps,
                module_type=cache_kind,
                return_device=cfg.resid_device,
                cache_embed=True,
                pad_token_id=pad_token_id,
                use_layer_norm=getattr(cfg, "use_vjp_layer_norm", False),
            )
        else:
            log_info(f"[rank {rank}] Building JVP resid cache (kind={cache_kind}, device={cfg.resid_device}) ...")
            resid_cache = ResidCache.build_for_local_pile_sample(
                model=model,
                local_pile_sample=local_pile_sample,
                accumulation_steps=cfg.accumulation_steps,
                module_type=cache_kind,
                return_device=cfg.resid_device,
                pad_token_id=pad_token_id,
                use_layer_norm=getattr(cfg, "use_jvp_layer_norm", False),
            )
        log_info(f"[rank {rank}] Resid cache built: chunks={len(resid_cache)}")

    elif st in ("token_change", "zero_out") and pile_sample is not None:
        batch_size = pile_sample.shape[0]
        start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
        local_pile_sample = pile_sample[start_idx:end_idx]
        pad_token_id = getattr(cfg, "pad_token_id", None)

        log_info(f"[rank {rank}] Building resid cache for valid_mask (token_change)...")
        resid_cache = ResidCache.build_for_local_pile_sample(
            model=model,
            local_pile_sample=local_pile_sample,
            accumulation_steps=cfg.accumulation_steps,
            module_type="all",
            return_device=cfg.resid_device,
            pad_token_id=pad_token_id,
            use_layer_norm=False,
        )

    # Load tuned lens params
    params_state = None
    if st == "tuned_lens":
        params_path = getattr(cfg, "params_path", None)
        params_state = _load_params_state(params_path)
        if params_state is not None:
            log_info(f"Loaded tuned lens params from: {params_path}")


    # SAE cache
    log_info("Initializing SAE GPU cache...")
    model_basename = cfg.model.split("/")[-1] if "/" in cfg.model else cfg.model
    sae_cache = SAEGPUCache(device=device, model_name=cfg.model, max_cache_size=1, max_memory_gb=1.0, dtype=dtype, model_name_aliases={model_basename})

    # Output path
    if os.path.isabs(cfg.output_filename):
        jsonl_out_path = cfg.output_filename
    else:
        jsonl_out_path = os.path.join(logs_dir, cfg.output_filename)
    jsonl_out_dir = os.path.dirname(jsonl_out_path) or "."
    os.makedirs(jsonl_out_dir, exist_ok=True)

    config_save_path = os.path.join(jsonl_out_dir, "config.yaml")
    with open(config_save_path, "w") as f_cfg:
        OmegaConf.save(cfg, f_cfg)
    log_info(f"Saved execution config to: {config_save_path}")

    # Resume logic
    processed_indices = set()
    if not dist.is_initialized() or dist.get_rank() == 0:
        if os.path.exists(jsonl_out_path):
            log_info(f"Found existing JSONL file: {jsonl_out_path}")
            processed_indices = load_processed_features(jsonl_out_path)
            if processed_indices:
                log_info(f"Resuming: {len(processed_indices)} features already processed")
        else:
            with open(jsonl_out_path, "w") as f_out:
                pass
            log_info(f"Created new JSONL output file: {jsonl_out_path}")

    if dist.is_initialized():
        processed_list = list(processed_indices) if processed_indices else []
        gathered_lists = [None] * world_size
        dist.all_gather_object(gathered_lists, processed_list)
        for pl in gathered_lists:
            if pl:
                processed_indices.update(pl)

    total_features = len(all_features_sample)
    remaining_features = total_features - len(processed_indices)
    log_info(f"Processing {remaining_features} remaining out of {total_features}")

    # Main loop
    #
    # NOTE: No per-feature try/except. Any exception during a feature must
    # propagate so torchrun reports the failure instead of letting one rank
    # silently skip and then desynchronize the NCCL process group at the
    # next collective. Recovery is via JSONL resume: relaunch the command
    # and it picks up from where it left off.
    with tqdm(
        total=total_features,
        desc=f"[{baseline_name}] rank {rank}",
        ncols=100,
        disable=dist.is_initialized() and dist.get_rank() != 0,
    ) as pbar:
        for i, ftr in enumerate(all_features_sample):
            if i in processed_indices:
                pbar.update(1)
                continue

            sae = sae_cache.get_sae(ftr)

            result = None
            if st == "identity":
                result = analyze_identity(
                    model, ftr, sae, fv, ro, cfg.k,
                    getattr(cfg, "use_unembed_layer_norm", True),
                )
            elif st in ("first_order", "full") and fv == "value":
                result = jvp_analyze_feature(
                    model=model, f=ftr, sae=sae,
                    pile_sample=pile_sample, rank=rank, world_size=world_size,
                    stream_transition=st,
                    submodule=cfg.submodule, k=cfg.k,
                    use_unembed_layer_norm=getattr(cfg, "use_unembed_layer_norm", True),
                    use_jvp_layer_norm=getattr(cfg, "use_jvp_layer_norm", False),
                    accumulation_steps=cfg.accumulation_steps,
                    resid_cache=resid_cache,
                )
            elif st in ("first_order", "full") and fv == "key":
                result = vjp_analyze_feature(
                    model=model, f=ftr, sae=sae,
                    pile_sample=pile_sample, rank=rank, world_size=world_size,
                    stream_transition=st,
                    submodule=cfg.submodule, k=cfg.k,
                    use_vjp_layer_norm=getattr(cfg, "use_vjp_layer_norm", False),
                    accumulation_steps=cfg.accumulation_steps,
                    resid_cache=resid_cache,
                )
            elif st in ("token_change", "zero_out"):
                result = analyze_token_change(
                    model, ftr, sae, pile_sample, rank, world_size,
                    cfg, resid_cache, k=cfg.k,
                    invert_sign=(st == "zero_out"),
                )
            elif st == "tuned_lens":
                result = analyze_tuned_lens(
                    model, ftr, sae, cfg.k,
                    getattr(cfg, "use_unembed_layer_norm", True),
                    params_state,
                )
            else:
                log_info(f"Unknown stream_transition: {st}")

            if result:
                row = {
                    "feature_index": i,
                    "feature_str": str(ftr),
                    **bl,
                    **result,
                }
                if not dist.is_initialized() or dist.get_rank() == 0:
                    append_result_to_jsonl(row, jsonl_out_path)

            pbar.update(1)

    log_info("Clearing SAE GPU cache...")
    sae_cache.clear_cache()

    cleanup_distributed()
    log_info("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config-path", "-cp", default=None)
    p.add_argument("--config-name", "-cn", default=None)
    args, unknown = p.parse_known_args(sys.argv[1:])

    config_dir = args.config_path or os.path.join(script_dir, "..", "conf")
    config_name = args.config_name or "config"

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=unknown)

    sys.exit(main(cfg))

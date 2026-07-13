#!/usr/bin/env python3
"""
Train full d_model x d_model linear maps for ALL (source_layer -> target_layer) pairs
with source_layer < target_layer.

Always token-parallel (base-style):
  - shard tokens across ranks
  - all ranks participate in EACH feature computation
  - mask padded tokens via valid_mask
  - compute global mean JVP via all_reduce(sum_vec, count)
  - rank0 collects (X,Y) and trains full dxd map
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import (
    Activation,
    SAEGPUCache,
    load_pile_sample,
    log_info,
    parse_dtype,
    read_pkl_file,
    set_seed,
    sort_features_by_layer,
    setup_distributed,
    cleanup_distributed,
    distribute_batch,
    Feature,
)

from analysis.common import (
    load_model, _expand_to_BSD, get_feature_vector,
    jvp_mlp, jvp_attn,
    ResidChunk, ResidCache,
    distributed_mean_vector,
)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train full maps for all source->target pairs (always token-sharded + per-feature all-reduce)."
    )
    p.add_argument("--features-pkl", type=str, required=True, help="Path to features sample pkl.")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--num-samples", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    # Full-map learning
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=0, help="0 means full-batch.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lambda-reg", type=float, default=0.0, help="Optional explicit L2 penalty on W.")
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 disables grad clipping.")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Fraction held out for validation per pair.")
    p.add_argument("--rank-r", type=int, required=True, help="Low-rank r for W=BA (A: rxd, B: dxr).")
    # ResidCache + masking
    p.add_argument("--accumulation-steps", type=int, default=1, help="Micro-batch count for ResidCache build.")
    p.add_argument("--resid-device", type=str, default="cuda", help='Where to store cached residuals: "cpu" recommended.')
    p.add_argument(
        "--use-jvp-layer-norm",
        action="store_true",
        help=(
            "If set, cache stores pre-LN residuals and JVP includes LN inside the function. "
            "If not set, cache stores LN-applied residuals and JVP runs without LN."
        ),
    )

    # output
    p.add_argument("--output-dir", type=str, default="subspace_channel_analysis", help="Directory to save all results.")
    return p.parse_args()


# -----------------------------
# Small helpers
# -----------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_num_layers(model) -> int:
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return int(model.cfg.n_layers)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    raise ValueError("Cannot infer number of layers from model.")

def pair_key(s: int, t: int) -> str:
    return f"{s}_to_{t}"


# -----------------------------
# Compute target-layer mean JVP using local ResidCache, then all-reduce
# -----------------------------
@torch.no_grad()
def compute_target_layer_jvp_mean_distributed(
    *,
    model,
    resid_cache: ResidCache,
    target_layer: int,
    f_dir: torch.Tensor,            # [d]
    device: str,
    use_jvp_layer_norm: bool,
) -> torch.Tensor:
    sum_vec = None
    count_total = None

    for micro_step in range(len(resid_cache)):
        R_attn_layers, R_mlp_layers, Vmask = resid_cache.get(micro_step, kind="all")

        # build mask on device
        if Vmask is None:
            if R_attn_layers is not None:
                _, Bm, S, _ = R_attn_layers.shape
            elif R_mlp_layers is not None:
                _, Bm, S, _ = R_mlp_layers.shape
            else:
                continue
            mask_micro = torch.ones((Bm, S), device=device, dtype=torch.float32)
        else:
            mask_micro = Vmask.to(device=device, non_blocking=True).float()

        count_micro = mask_micro.sum().float()
        count_total = count_micro if count_total is None else (count_total + count_micro)

        mask_bsd = mask_micro.unsqueeze(-1).to(dtype=model.cfg.dtype)

        jvp_micro = None

        if R_attn_layers is not None:
            X_attn = R_attn_layers[target_layer].to(device=device, dtype=model.cfg.dtype, non_blocking=True)
            j_attn = jvp_attn(
                model, layer_idx=target_layer, X=X_attn, f_vec=f_dir,
                use_layer_norm=use_jvp_layer_norm,
                attention_mask=mask_micro.bool(),
            )
            jvp_micro = j_attn if jvp_micro is None else (jvp_micro + j_attn)

        if R_mlp_layers is not None:
            X_mlp = R_mlp_layers[target_layer].to(device=device, dtype=model.cfg.dtype, non_blocking=True)
            j_mlp = jvp_mlp(
                model, layer_idx=target_layer, X=X_mlp, f_vec=f_dir,
                use_layer_norm=use_jvp_layer_norm,
            )
            jvp_micro = j_mlp if jvp_micro is None else (jvp_micro + j_mlp)

        if jvp_micro is None:
            continue

        jvp_micro = jvp_micro * mask_bsd
        micro_sum = jvp_micro.float().sum(dim=(0, 1))  # [d]

        sum_vec = micro_sum if sum_vec is None else (sum_vec + micro_sum)

        del jvp_micro, micro_sum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sum_vec is None:
        sum_vec = torch.zeros((model.cfg.d_model,), device=device, dtype=torch.float32)
    if count_total is None:
        count_total = torch.tensor(0.0, device=device, dtype=torch.float32)

    # all-reduce to get global mean
    mean_vec = distributed_mean_vector(sum_vec, count_total)
    return mean_vec.to(dtype=model.cfg.dtype)


# -----------------------------
# Map training
# -----------------------------
def _energy_explained(Y: torch.Tensor, Y_hat: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    resid = (Y_hat - Y)
    resid_energy = float(resid.pow(2).sum().item())
    total_energy = float(Y.pow(2).sum().item())
    pred_energy = float(Y_hat.pow(2).sum().item())
    explained = 1.0 - (resid_energy / (total_energy + eps))
    pred_ratio = pred_energy / (total_energy + eps)
    return {
        "total_energy": total_energy,
        "resid_energy": resid_energy,
        "pred_energy": pred_energy,
        "energy_explained": explained,
        "pred_energy_ratio": pred_ratio,
    }

def _compute_metrics(Y: torch.Tensor, Y_hat: torch.Tensor) -> Dict[str, float]:
    resid = Y_hat - Y
    mse = float(resid.pow(2).mean().item())
    rmse = float(math.sqrt(mse + 1e-12))
    rel_frob = float(torch.linalg.norm(resid).item() / (torch.linalg.norm(Y).item() + 1e-12))
    energy = _energy_explained(Y, Y_hat)
    l2 = float(torch.norm(resid, dim=1).mean().item())
    cosine = float(F.cosine_similarity(Y_hat, Y, dim=1).mean().item())
    return {"mse": mse, "rmse": rmse, "rel_frob": rel_frob, "l2": l2, "cosine": cosine, **energy}

def train_full_map(
    X: torch.Tensor,
    Y: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float = 0.0,
    lambda_reg: float = 0.0,
    grad_clip: float = 0.0,
    device: str = "cuda",
    val_ratio: float = 0.0,
    rank_r: int = 4,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, float]], Dict[str, torch.Tensor]]:
    """
    Low-rank map: W = B @ A
      A: [r, d]
      B: [d, r]
    Predict (row-batch): Y_hat = (X @ A.T) @ B.T
    Returns:
      W_full: [d,d]
      metrics
      factors: {"A": A, "B": B}
    """
    assert X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape
    N, d = X.shape
    assert N > 0
    assert rank_r > 0 and rank_r <= d, f"rank_r must be in [1, d], got {rank_r} vs d={d}"

    Xd = X.to(device=device, dtype=torch.float32)
    Yd = Y.to(device=device, dtype=torch.float32)

    val_count = int(round(N * val_ratio))
    val_count = min(max(val_count, 0), max(N - 1, 0))
    perm = torch.randperm(N)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    X_train, Y_train = Xd[train_idx], Yd[train_idx]
    X_val, Y_val = (Xd[val_idx], Yd[val_idx]) if val_count > 0 else (None, None)

    # LoRA-ish init: A ~ N(0, 0.01), B = 0 so initial W=0
    A = torch.nn.Parameter(torch.empty((rank_r, d), device=device, dtype=torch.float32))
    B = torch.nn.Parameter(torch.zeros((d, rank_r), device=device, dtype=torch.float32))
    torch.nn.init.normal_(A, mean=0.0, std=0.01)

    opt = torch.optim.AdamW([A, B], lr=lr, weight_decay=weight_decay)

    if batch_size is None or batch_size <= 0 or batch_size >= int(X_train.shape[0]):
        batches = [(X_train, Y_train)]
    else:
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_train, Y_train)
        batches = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best: Dict[str, Optional[torch.Tensor]] = {"loss": float("inf"), "A": None, "B": None}

    def forward_lowrank(Xb: torch.Tensor) -> torch.Tensor:
        # Xb: [bs,d]
        # Z: [bs,r]
        Z = Xb @ A.T
        # Yhat: [bs,d]
        return Z @ B.T

    for _ep in range(1, epochs + 1):
        with torch.enable_grad():
            for Xb, Yb in batches:
                opt.zero_grad(set_to_none=True)
                Y_hat = forward_lowrank(Xb)
                resid = Y_hat - Yb
                loss = resid.pow(2).mean()

                # Optional explicit L2 on factors (or on BA indirectly)
                if lambda_reg > 0.0:
                    loss = loss + lambda_reg * (A.pow(2).mean() + B.pow(2).mean())

                loss.backward()
                if grad_clip and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_([A, B], max_norm=grad_clip)
                opt.step()

        with torch.no_grad():
            train_pred = forward_lowrank(X_train)
            train_mse = float((train_pred - Y_train).pow(2).mean().item())

            if X_val is not None and Y_val is not None:
                val_pred = forward_lowrank(X_val)
                val_mse = float((val_pred - Y_val).pow(2).mean().item())
                score = val_mse
            else:
                val_mse = None
                score = train_mse

            if score < best["loss"]:
                best["loss"] = score
                best["A"] = A.detach().clone()
                best["B"] = B.detach().clone()

    A_best = best["A"] if best["A"] is not None else A.detach().clone()
    B_best = best["B"] if best["B"] is not None else B.detach().clone()

    with torch.no_grad():
        def forward_best(Xb: torch.Tensor) -> torch.Tensor:
            return (Xb @ A_best.T) @ B_best.T

        train_pred = forward_best(X_train)
        train_metrics = _compute_metrics(Y_train, train_pred)

        if X_val is not None and Y_val is not None:
            val_pred = forward_best(X_val)
            val_metrics = _compute_metrics(Y_val, val_pred)
        else:
            val_metrics = None

        W_full = (B_best @ A_best)  # [d,d]

    metrics: Dict[str, Dict[str, float]] = {
        "train": train_metrics,
        "val": val_metrics if val_metrics is not None else {},
        "n_train": int(X_train.shape[0]),
        "n_val": int(val_count),
    }
    factors = {"A": A_best.cpu(), "B": B_best.cpu()}
    return W_full.cpu(), metrics, factors


# -----------------------------
# Pair data collection: all ranks participate per feature, rank0 collects
# -----------------------------
@torch.no_grad()
def collect_XY_for_pair_distributed(
    *,
    model,
    features: List[Feature],
    source_indices: List[int],
    target_layer: int,
    sae_cache: SAEGPUCache,
    resid_cache: ResidCache,
    device: str,
    rank: int,
    use_jvp_layer_norm: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
    if not source_indices:
        return None, None, []

    X_list_rank0: List[torch.Tensor] = []
    Y_list_rank0: List[torch.Tensor] = []
    used_idx_rank0: List[int] = []

    # IMPORTANT: every rank iterates over exactly the same indices, in the same order.
    for idx in source_indices:
        f = features[idx]
        sae = sae_cache.get_sae(f)

        f_dir = get_feature_vector(model, f, sae).to(device=device, dtype=model.cfg.dtype)
        f_dir_vec = f_dir if f_dir.ndim == 1 else f_dir.view(-1)

        # all ranks compute global mean JVP for this feature at target_layer
        jvp_mean = compute_target_layer_jvp_mean_distributed(
            model=model,
            resid_cache=resid_cache,
            target_layer=target_layer,
            f_dir=f_dir_vec,
            device=device,
            use_jvp_layer_norm=use_jvp_layer_norm,
        )

        # rank0 collects (X,Y)
        if rank == 0:
            X_list_rank0.append(f_dir_vec.detach().cpu().float())
            Y_list_rank0.append(jvp_mean.detach().cpu().float())
            used_idx_rank0.append(idx)

    if rank != 0:
        return None, None, []

    if not X_list_rank0:
        return None, None, []

    X = torch.stack(X_list_rank0, dim=0)
    Y = torch.stack(Y_list_rank0, dim=0)
    return X, Y, used_idx_rank0


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    if rank is None:
        rank = 0

    set_seed(args.seed + rank)
    dtype = parse_dtype(args.dtype)

    if world_size > 1 and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = get_device()

    log_info(f"[rank {rank}] device={device}, dtype={dtype}, world_size={world_size}")

    model = load_model(device, args.model, dtype=dtype)
    n_layers = get_num_layers(model)
    if rank == 0:
        log_info(f"Loaded model {args.model}, n_layers={n_layers}")

    # features
    features = sort_features_by_layer(read_pkl_file(args.features_pkl))
    indices_by_source: Dict[int, List[int]] = {l: [] for l in range(n_layers)}
    for idx, f in enumerate(features):
        try:
            l = int(getattr(f, "layer", -1))
        except Exception:
            continue
        if 0 <= l < n_layers:
            indices_by_source[l].append(idx)
    if rank == 0:
        counts = {l: len(v) for l, v in indices_by_source.items()}
        log_info(f"Feature counts by source layer: {counts}")

    # tokens (global load, then shard)
    pile_sample = load_pile_sample(
        model,
        device,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )
    batch_size = pile_sample.shape[0]
    start_idx, end_idx = distribute_batch(batch_size, rank, world_size)
    local_pile_sample = pile_sample[start_idx:end_idx]  # [B_local,S]
    if rank == 0:
        log_info(f"pile_sample={tuple(pile_sample.shape)}, local shards ~ {end_idx-start_idx} each")

    log_info(
        f"[rank {rank}] Building ResidCache once from local tokens {tuple(local_pile_sample.shape)}, "
        f"resid_device={args.resid_device}"
    )
    resid_cache = ResidCache.build_for_local_pile_sample(
        model=model,
        local_pile_sample=local_pile_sample,
        accumulation_steps=args.accumulation_steps,
        module_type="all",
        return_device=args.resid_device,
        use_layer_norm=args.use_jvp_layer_norm,
    )
    log_info(f"[rank {rank}] ResidCache built: chunks={len(resid_cache)}")

    # SAE cache per rank
    model_basename = args.model.split("/")[-1] if "/" in args.model else args.model
    sae_cache = SAEGPUCache(
        device=device,
        model_name=args.model,
        max_cache_size=1,
        max_memory_gb=1.0,
        dtype=dtype,
        model_name_aliases={model_basename},
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # train all pairs
    pairs = [(s, t) for s in range(n_layers) for t in range(n_layers) if s < t]

    for (s, t) in pairs:
        source_indices = indices_by_source.get(s, [])
        if not source_indices:
            if rank == 0:
                log_info(f"[rank0] Skip {s}->{t}: no features for source {s}")
            continue

        # IMPORTANT: all ranks call this, rank0 receives X,Y
        X, Y, used_feature_indices = collect_XY_for_pair_distributed(
            model=model,
            features=features,
            source_indices=source_indices,
            target_layer=t,
            sae_cache=sae_cache,
            resid_cache=resid_cache,
            device=device,
            rank=rank,
            use_jvp_layer_norm=args.use_jvp_layer_norm,
        )

        if rank != 0:
            continue

        if X is None or Y is None:
            log_info(f"[rank0] Skip {s}->{t}: no data")
            continue

        N = int(X.shape[0])
        train_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        W, metrics, factors = train_full_map(
            X=X.float(),
            Y=Y.float(),
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            lambda_reg=args.lambda_reg,
            grad_clip=args.grad_clip,
            device=train_device,
            val_ratio=args.val_ratio,
            rank_r=args.rank_r,
        )

        out_path = os.path.join(args.output_dir, f"full_map_{pair_key(s,t)}.pt")
        torch.save(
            {
                "W": W,                 # [d,d] = B@A
                "A": factors["A"],      # [r,d]
                "B": factors["B"],      # [d,r]
                "rank_r": args.rank_r,
                "metrics": metrics,
                "N": N,
                "n_train": metrics["n_train"],
                "n_val": metrics["n_val"],
                "source_layer": s,
                "target_layer": t,
                "feature_indices": used_feature_indices,
                "model": args.model,
                "dtype": str(dtype),
                "seed": args.seed,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "lambda_reg": args.lambda_reg,
                "val_ratio": args.val_ratio,
                "use_jvp_layer_norm": args.use_jvp_layer_norm,
                "accumulation_steps": args.accumulation_steps,
                "resid_device": args.resid_device,
                "world_size": world_size,
            },
            out_path,
        )

        train_m = metrics["train"]
        val_m = metrics["val"] if metrics["n_val"] > 0 else None
        if val_m:
            log_info(
                f"[rank0] Saved {out_path} | N={N} (train {metrics['n_train']}, val {metrics['n_val']}) | "
                f"train mse {train_m['mse']:.6g}, l2 {train_m['l2']:.6g}, cos {train_m['cosine']:.6g}, energy {train_m['energy_explained']:.6f} | "
                f"val mse {val_m['mse']:.6g}, l2 {val_m['l2']:.6g}, cos {val_m['cosine']:.6g}, energy {val_m['energy_explained']:.6f}"
            )
        else:
            log_info(
                f"[rank0] Saved {out_path} | N={N} | "
                f"train mse {train_m['mse']:.6g}, l2 {train_m['l2']:.6g}, cos {train_m['cosine']:.6g}, energy {train_m['energy_explained']:.6f}"
            )

    sae_cache.clear_cache()
    cleanup_distributed()


if __name__ == "__main__":
    main()

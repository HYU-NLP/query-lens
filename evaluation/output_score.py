#!/usr/bin/env python3
"""
Output Score (NEW primary metric): set-overlap between a lens prediction T
and the per-feature "rank-percentile-increasing" oracle S.

For each feature, we compute the per-token mean rank-percentile delta across
KL-calibrated steering levels α:

    Δ(t)  = mean_α [ (r_clean(t) - r_steered(t, α)) / V ]                ∈ [-1, +1]
    S(K)  = top-K vocabulary tokens by Δ                                  (|S| = K)

Choosing K = |T| keeps the metric symmetric:

    score(T)  =  | S ∩ T | / | S |                                       ∈ [0, 1]

This captures how well the lens recovers the tokens whose ranks rise the
most under steering, as opposed to averaging rank-percentile deltas over T
(which dilutes the signal across whatever T proposed). Sweep across 11
model/SAE cells × 9 baselines under this metric is reported in
`experiments/leaderboard.md`.

Legacy primary (mean rank-percentile delta over T) is kept in the per-row
JSONL as `rpd_score_legacy` for backward inspection. The per-row JSONL still
exposes `clean_score`, `steered_score`, and `delta` as length-normalized
generation frequencies for spot-checking the sampled continuations.
The summary.json also reports `steered_p90` as a saturation indicator.

Uses precomputed steering caches on Feature objects (no GPU required).
Each feature's steering_cache contains clean and steered token ID lists at
per-feature-calibrated KL targets (typically +0.25 and +0.5).

See paper Section 5.2 (Output-Side Evaluation) for the originating definition.

Usage:
    python evaluation/output_score.py \
        --jsonl experiments/baseline/feature_analysis.jsonl \
        --features features/my_features.pkl
"""

import os
import json
import math
import ast
from typing import Dict, List, Optional, Any

from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import (
    Feature,
    Activation,
    read_pkl_file,
    append_result_to_jsonl,
    sort_features_by_layer,
)
from visualization.visualize import visualize_steering_results


LOGS_DIR = "experiments"


# =========================================================
# Parsing / scoring utils
# =========================================================

def parse_id_list(obj: Any) -> List[int]:
    """
    Parse a list of token ids that is either:
    - already a Python list of ints (or castable), or
    - a string representation of such a list, e.g. "[123, 456]".
    """
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []

    if isinstance(obj, list):
        ids: List[int] = []
        for x in obj:
            if x is None:
                continue
            try:
                ids.append(int(x))
            except (TypeError, ValueError):
                continue
        return ids

    try:
        parsed = ast.literal_eval(str(obj))
    except (ValueError, SyntaxError) as e:
        preview = str(obj)[:100]
        logger.warning(f"Failed to parse id list: {preview}... Error: {e}")
        return []

    if isinstance(parsed, list):
        ids: List[int] = []
        for x in parsed:
            if x is None:
                continue
            try:
                ids.append(int(x))
            except (TypeError, ValueError):
                continue
        return ids

    return []


def calculate_frequency_score(
    continuation_tokens_list: List[List[int]],
    target_tokens: List[int]
) -> float:
    """
    For each continuation, compute the fraction of generated positions that
    are occupied by a token in the target set, then average over continuations.

        M_freq(S, T) = (1/|P|) Σ_p |{i : S^p[i] ∈ T}| / |S^p|

    This is the length-normalized frequency variant of the paper's M(S,T).
    The original paper formula `|{t ∈ S^p | t ∈ T}|` (interpreted as a set
    intersection) only counted whether a target token was *present* in the
    continuation; the prose, however, says "we quantify how much those tokens
    become more frequent under feature steering" — multiplicity matters.
    Frequency aligns the implementation with the paper's stated intent and
    is strictly more sensitive: a target token appearing 8× in a steered
    generation vs 1× clean produces a Δ of (8-1)/L instead of 0.

    Returns a value in [0, 1] per prompt; the difference (steered - clean)
    is in [-1, +1].
    """
    if len(continuation_tokens_list) == 0 or len(target_tokens) == 0:
        return 0.0

    target_set = {int(tok) for tok in target_tokens}

    total_fraction = 0.0
    for continuation_tokens in continuation_tokens_list:
        if not continuation_tokens:
            continue
        hits = sum(1 for tok in continuation_tokens if int(tok) in target_set)
        total_fraction += hits / len(continuation_tokens)

    return total_fraction / len(continuation_tokens_list)


# =========================================================
# Loading features/token-ids
# =========================================================

def load_features_from_pickle(features_file: str) -> List[Feature]:
    if not os.path.exists(features_file):
        logger.error(f"Features file not found: {features_file}")
        raise FileNotFoundError(features_file)

    logger.info(f"Loading features from {features_file}...")
    features = read_pkl_file(features_file)
    logger.info(f"Loaded {len(features)} features from {features_file}")
    return features


def load_token_lists_from_jsonl(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load token id lists from JSONL file, indexed by feature_str.

    Reads unified format: top_ids (one per JSONL entry, one baseline per run).
    """
    token_data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        logger.warning(f"Token lists JSONL file not found: {jsonl_path}")
        return token_data

    logger.info(f"Loading token lists from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line {line_no} in {jsonl_path}: {e}")
                continue

            feature_str = entry.get("feature_str")
            if feature_str is None:
                continue

            token_data[feature_str] = {
                "feature_index": entry.get("feature_index"),
                "top_ids": entry.get("top_ids", []),
            }

    logger.info(f"Loaded token id lists for {len(token_data)} features from {jsonl_path}")
    return token_data


# =========================================================
# Scoring from cache
# =========================================================

def score_features_from_cache(
    features_file: str,
    input_jsonl: str,
    output_file: str,
):
    """
    Score features using precomputed steering caches stored on Feature objects.
    KL values are read from each feature's steering_cache.
    """
    logger.info("Scoring features from precomputed steering caches (no GPU)")

    all_features = load_features_from_pickle(features_file)
    all_features = sort_features_by_layer(all_features)

    token_data: Dict[str, Dict[str, Any]] = {}
    if input_jsonl:
        token_data = load_token_lists_from_jsonl(input_jsonl)

    # Always start fresh — overwrite any existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    total_processed = 0
    total_skipped = 0
    all_results = []

    for feature_idx, f in enumerate(tqdm(all_features, desc="Scoring from cache")):
        feature_str = str(f)

        token_info = token_data.get(feature_str, {})
        top_ids = parse_id_list(token_info.get("top_ids", []))

        if not top_ids:
            logger.warning(f"No token id lists found for feature {feature_str}, skipping")
            continue

        cache = getattr(f, 'steering_cache', None)
        if cache is None:
            logger.warning(f"No steering cache for feature {feature_str}, skipping")
            total_skipped += 1
            continue

        clean_tokens_list = cache.get("clean_token_lists", [])
        steered_map = cache.get("steered_token_lists", {})
        clean_ranks = cache.get("clean_mean_ranks", [])
        steered_ranks_map = cache.get("steered_mean_ranks", {})
        vocab_size = len(clean_ranks)

        # Build result
        feature_layer = int(f.layer)
        clean_top_score = calculate_frequency_score(clean_tokens_list, top_ids)

        layer_result: Dict[str, Any] = {
            "layer_index": feature_layer,
            "clean_scores": {"top": clean_top_score},
            "steered_scores": {},         # per-KL freq M_steered (spot-checking)
            "deltas": {},                 # per-KL additive freq delta
            "rank_percentile_deltas": {}, # per-KL P_s - P_c (legacy signal, no longer primary)
        }

        # Clean rank percentile, averaged over target ids in vocab range
        valid_ids = [t for t in top_ids if 0 <= t < vocab_size] if vocab_size else []
        clean_pct = 0.0
        if valid_ids:
            clean_pct = sum(1.0 - clean_ranks[t] / vocab_size for t in valid_ids) / len(valid_ids)

        # Per-token rank-percentile delta across KL levels (used to build S).
        per_token_delta_sum = None
        n_kl = 0
        for kl_key, kl_tokens_list in steered_map.items():
            kl_val = float(kl_key.replace("+", ""))
            if kl_val < 0 or not top_ids:
                continue
            steered_score = calculate_frequency_score(kl_tokens_list, top_ids)
            layer_result["steered_scores"][kl_key] = steered_score
            layer_result["deltas"][kl_key] = steered_score - clean_top_score

            kl_steered_ranks = steered_ranks_map.get(kl_key, [])
            if valid_ids and kl_steered_ranks and len(kl_steered_ranks) == vocab_size:
                steer_pct = sum(1.0 - kl_steered_ranks[t] / vocab_size for t in valid_ids) / len(valid_ids)
                layer_result["rank_percentile_deltas"][kl_key] = steer_pct - clean_pct

            # Accumulate per-token rank-percentile delta = (clean_rank − steered_rank) / V
            if vocab_size and kl_steered_ranks and len(kl_steered_ranks) == vocab_size:
                if per_token_delta_sum is None:
                    per_token_delta_sum = [0.0] * vocab_size
                for t_idx in range(vocab_size):
                    per_token_delta_sum[t_idx] += (clean_ranks[t_idx] - kl_steered_ranks[t_idx]) / vocab_size
                n_kl += 1

        # ── Primary score (NEW): O_new(T) = |S ∩ T| / |S| ───────────────────────
        # S = per-feature top-K vocabulary tokens with the largest mean
        # rank-percentile-delta across KL levels (the "rank-percentile-increasing
        # set"). K matches |T| so the metric stays in [0, 1] symmetrically.
        S_size = 0
        if per_token_delta_sum is not None and n_kl > 0 and top_ids:
            mean_deltas = [d / n_kl for d in per_token_delta_sum]
            K = len(top_ids)
            # argpartition-style top-K selection in pure Python
            indexed = list(enumerate(mean_deltas))
            indexed.sort(key=lambda kv: -kv[1])
            S_set = {idx for idx, _ in indexed[:K]}
            T_set = {int(t) for t in top_ids}
            S_size = len(S_set)
            score = (len(S_set & T_set) / S_size) if S_size else 0.0
        else:
            score = 0.0

        # ── Legacy primary score (commented out): mean rank-percentile delta ────
        # rpd = layer_result["rank_percentile_deltas"]
        # if rpd:
        #     score = sum(rpd.values()) / len(rpd)
        # else:
        #     score = 0.0

        include_top = bool(top_ids) and bool(layer_result["steered_scores"])
        if include_top:
            steered_mean = sum(layer_result["steered_scores"].values()) / len(layer_result["steered_scores"])
            delta_mean = steered_mean - clean_top_score
        else:
            steered_mean = 0.0
            delta_mean = 0.0
        # Legacy rank-percentile-delta primary (kept for backward inspection)
        legacy_rpd = layer_result["rank_percentile_deltas"]
        legacy_rpd_score = (sum(legacy_rpd.values()) / len(legacy_rpd)) if legacy_rpd else 0.0
        layer_result["stats"] = {
            "clean_score": clean_top_score if include_top else 0.0,
            "steered_score": steered_mean,
            "delta": delta_mean,
            "score": score,                       # NEW primary: O_new(T)
            "S_size": S_size,
            "rpd_score_legacy": legacy_rpd_score, # legacy primary (mean rank-%-delta)
            "clean_rank_percentile": clean_pct,
        }

        result = {
            "model_id": getattr(f, "model_id", "unknown"),
            "feature": f.feature,
            "layer": str(f.layer),
            "type": getattr(f, "type", "unknown"),
            "size": getattr(f, "size", ""),
            "result_layer": layer_result,
            "feature_index": feature_idx,
            "feature_str": feature_str,
        }

        all_results.append(result)
        append_result_to_jsonl(result, output_file)
        total_processed += 1

    # Compute summary statistics (overall + per-layer)
    from collections import defaultdict
    import math as _math

    Z_95 = 1.959964

    def _p90(values: List[float]) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        idx = max(0, min(len(s) - 1, int(round(0.9 * (len(s) - 1)))))
        return s[idx]

    def _layer_ci_stats(values: List[float]) -> Dict[str, Any]:
        """Per-layer mean ± 1.96·s/√n where s is sample std (ddof=1)."""
        n = len(values)
        if n == 0:
            return {"se": None, "ci95_lo": None, "ci95_hi": None, "std": None}
        if n < 2:
            return {"se": None, "ci95_lo": None, "ci95_hi": None, "std": 0.0}
        m = sum(values) / n
        var = sum((x - m) ** 2 for x in values) / (n - 1)
        s = _math.sqrt(var)
        se = s / _math.sqrt(n)
        return {
            "std": round(s, 6),
            "se": round(se, 6),
            "ci95_lo": round(m - Z_95 * se, 6),
            "ci95_hi": round(m + Z_95 * se, 6),
        }

    def _stratified_mean_ci(layer_to_scores: Dict[int, List[float]]) -> Dict[str, Any]:
        """Stratified mean estimator with equal layer weights w_c = 1/L.
        μ̂ = (1/L) · Σ μ_c
        SE(μ̂) = (1/L) · sqrt(Σ s_c² / n_c)  (assumes equal weight per layer)
        """
        layers = sorted(k for k, v in layer_to_scores.items() if len(v) >= 1)
        L = len(layers)
        if L == 0:
            return {"L_layers": 0, "mean": None, "se": None, "ci95_lo": None, "ci95_hi": None}
        layer_means: List[float] = []
        var_sum = 0.0
        var_sum_valid = True
        for c in layers:
            vals = layer_to_scores[c]
            n_c = len(vals)
            mu_c = sum(vals) / n_c
            layer_means.append(mu_c)
            if n_c < 2:
                var_sum_valid = False
                continue
            s2_c = sum((x - mu_c) ** 2 for x in vals) / (n_c - 1)
            var_sum += s2_c / n_c
        m_hat = sum(layer_means) / L
        if not var_sum_valid:
            return {"L_layers": L, "mean": round(m_hat, 6), "se": None, "ci95_lo": None, "ci95_hi": None}
        se_hat = _math.sqrt(var_sum) / L
        return {
            "L_layers": L,
            "mean": round(m_hat, 6),
            "se": round(se_hat, 6),
            "ci95_lo": round(m_hat - Z_95 * se_hat, 6),
            "ci95_hi": round(m_hat + Z_95 * se_hat, 6),
        }

    all_scores: List[float] = []
    all_steered: List[float] = []
    layer_scores: Dict[int, List[float]] = defaultdict(list)
    layer_steered: Dict[int, List[float]] = defaultdict(list)
    for r in all_results:
        stats = r.get("result_layer", {}).get("stats", {})
        score = stats.get("score")
        steered = stats.get("steered_score")
        if score is None:
            continue
        all_scores.append(score)
        if steered is not None:
            all_steered.append(steered)
        try:
            layer_idx = int(r.get("result_layer", {}).get("layer_index"))
        except (TypeError, ValueError):
            continue
        layer_scores[layer_idx].append(score)
        if steered is not None:
            layer_steered[layer_idx].append(steered)

    overall_block = {
        "count": len(all_scores),
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
        "steered_p90": _p90(all_steered),
    }
    # Stratified mean (equal-weighted across layers)
    overall_block.update(_stratified_mean_ci(layer_scores))

    layer_blocks = []
    for layer_idx in sorted(layer_scores.keys()):
        vals = layer_scores[layer_idx]
        block = {
            "layer": layer_idx,
            "count": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "steered_p90": _p90(layer_steered.get(layer_idx, [])),
        }
        block.update(_layer_ci_stats(vals))
        layer_blocks.append(block)

    summary_stats = {"overall": overall_block, "layers": layer_blocks}

    # Merge into summary.json
    output_dir = os.path.dirname(os.path.abspath(output_file))
    summary_path = os.path.join(output_dir, "summary.json")
    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    existing_summary = loaded
        except Exception:
            existing_summary = {}

    existing_summary["steering"] = summary_stats
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(existing_summary, fh, indent=2)

    logger.info(f"Scoring complete: {total_processed} scored, {total_skipped} skipped (no cache)")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Summary saved to {summary_path}")

    return summary_stats


# =========================================================
# CLI
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Score features using precomputed steering caches (no GPU required)"
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL file with token id lists (e.g., feature_analysis.jsonl)")
    parser.add_argument("--features", type=str, required=True, help="Path to features pickle file (enriched with steering cache)")
    parser.add_argument("--output", type=str, default="feature_steering_results.jsonl", help="Output JSONL file path")
    parser.add_argument("--visualize", action="store_true", help="Generate steering plots and summary after processing.")
    args = parser.parse_args()

    if not os.path.isabs(args.output):
        input_dir = os.path.dirname(os.path.abspath(args.jsonl))
        args.output = os.path.join(input_dir, args.output)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    summary_stats = score_features_from_cache(
        features_file=args.features,
        input_jsonl=args.jsonl,
        output_file=args.output,
    )

    overall = summary_stats.get("overall", {})
    print(f"\n=== Summary Statistics ===")
    print(f"Processed {overall.get('count', 0)} features")
    if overall.get("mean") is not None:
        print(f"Output Score - Mean: {overall['mean']:.4f}, Min: {overall['min']:.4f}, Max: {overall['max']:.4f}")
    else:
        print("Output Score - n/a")

    if args.visualize:
        try:
            base, _ = os.path.splitext(os.path.abspath(args.output))
            plot_path = f"{base}.png"
            visualize_steering_results(
                jsonl_path=args.output,
                output_path=plot_path,
                try_positive_fallback=True,
            )
            logger.info(f"Saved steering visualization to {plot_path}")
        except Exception as e:
            logger.warning(f"Visualization step failed: {e}")


if __name__ == "__main__":
    main()

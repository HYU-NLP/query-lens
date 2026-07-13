#!/usr/bin/env python3
"""
Input Score: I(T) = |{t in A | t in T}| / |A|

For each feature, collects the max-activated token from each activation example
(set A) and measures the fraction that appears in the method's token set T.

See paper Section 5.1 (Input-Side Evaluation) for details.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import Feature, Activation, read_pkl_file, normalize_tokens
# visualization imports removed: summarize_scores / save_layerwise_plot no longer needed


def _safe_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    return []

def collect_max_activating_tokens(feature: Feature) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Collect per-activation max activating tokens (duplicates preserved).
    Also collects activating example strings per token.

    Returns:
        (tokens_with_duplicates, token_to_activating_strings)
    """
    max_activating_tokens: List[str] = []
    max_activating_strs: Dict[str, set] = {}

    if not getattr(feature, "activations", None):
        return max_activating_tokens, {}

    for activation in feature.activations:
        if activation.max_value_token_index is None:
            continue

        idx = activation.max_value_token_index
        values = activation.get_values()
        tokens = activation.get_tokens()

        if idx < len(values) and idx < len(tokens):
            token = tokens[idx]
            max_activating_tokens.append(token)

            activating_str = activation.get_tokens_str()
            if token not in max_activating_strs:
                max_activating_strs[token] = set()
            max_activating_strs[token].add(activating_str)

    max_activating_strs_clean = {t: list(s) for t, s in max_activating_strs.items()}
    return max_activating_tokens, max_activating_strs_clean


def calculate_overlap_ratio(tokens: List[str], max_activating_tokens: List[str]) -> float:
    """Ratio of (duplicate-preserving) max-activating tokens that are in target token set."""
    if not tokens or len(max_activating_tokens) == 0:
        return 0.0
    target_token_set = set(normalize_tokens(tokens))
    overlap_count = sum(1 for tok in normalize_tokens(max_activating_tokens) if tok in target_token_set)
    return overlap_count / len(max_activating_tokens)


def _extract_features_from_results(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    candidate = payload
    if "activation" in candidate and isinstance(candidate["activation"], dict):
        candidate = candidate["activation"]
    if "features" in candidate and isinstance(candidate["features"], dict):
        candidate = candidate["features"]
    if not isinstance(candidate, dict):
        return {}
    return candidate


def load_existing_activation_results(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load existing activation results (feature_activation_results.json) if available.
    Used only as a fallback if tokens/max tokens are missing.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load existing activation results from {path}: {e}")
        return {}

    features = _extract_features_from_results(payload)
    existing: Dict[str, Dict[str, Any]] = {}
    for key, entry in features.items():
        if not isinstance(entry, dict):
            continue
        existing[key] = {
            "score": entry.get("score"),
        }

    print(f"Loaded existing scores for {len(existing)} features from {path}")
    return existing

def load_token_lists_from_jsonl(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load token lists from JSONL file, indexed by feature_str.

    Reads unified format: top_tokens (one per JSONL entry, one baseline per run).
    """
    token_data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        print(f"Warning: Token lists JSONL file not found: {jsonl_path}")
        return token_data

    print(f"Loading token lists from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line {line_no} in {jsonl_path}: {e}")
                continue

            feature_str = entry.get("feature_str")
            if feature_str is None:
                continue

            token_data[feature_str] = {
                "feature_index": entry.get("feature_index"),
                "top_tokens": _safe_list(entry.get("top_tokens")),
            }

    print(f"Loaded token lists for {len(token_data)} features from {jsonl_path}")
    return token_data


def process_features(
    jsonl_path: str,
    features_file: str,
    output_file: Optional[str] = None,
    reference_results_path: Optional[str] = None,  # kept for signature compatibility, but unused
) -> Dict[str, Any]:
    """
    Process features and calculate activation token overlap scores.

    Reads unified JSONL format with single top_tokens per entry.
    Computes a single overlap score per feature.
    """
    token_data = load_token_lists_from_jsonl(jsonl_path)

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")

    print(f"Loading features from {features_file}...")
    features = read_pkl_file(features_file)
    print(f"Loaded {len(features)} features from {features_file}")

    feature_dict = {str(f): f for f in features}

    results: Dict[str, Any] = {}

    for feature_str, token_info in token_data.items():
        if feature_str not in feature_dict:
            print(f"Warning: Feature '{feature_str}' not found in features file")
            continue

        feature = feature_dict[feature_str]
        max_activating_tokens, max_activating_strs = collect_max_activating_tokens(feature)

        top_tokens = token_info.get("top_tokens", [])

        score = calculate_overlap_ratio(top_tokens, max_activating_tokens)

        feature_index = token_info.get("feature_index")
        results[feature_str] = {
            "feature_index": feature_index,
            "feature_str": feature_str,

            "top_tokens": top_tokens,

            "max_activating_tokens": list(max_activating_tokens),
            "max_activating_strs": max_activating_strs,
            "num_max_activating_tokens": len(max_activating_tokens),

            "score": score,
        }

    # Compute per-layer statistics
    import re
    import math as _math
    from collections import defaultdict

    Z_95 = 1.959964

    layer_scores: Dict[int, List[float]] = defaultdict(list)
    for feature_str, r in results.items():
        score = r.get("score")
        if score is None:
            continue
        match = re.search(r"/(\d+)/", feature_str)
        if match:
            layer_idx = int(match.group(1))
            layer_scores[layer_idx].append(score)

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
        Var(μ̂) = Σ w_c² · s_c² / n_c  (here w_c = 1/L)
        SE(μ̂) = sqrt(Var(μ̂))
        95% CI = μ̂ ± 1.96·SE(μ̂)
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
        se_hat = _math.sqrt(var_sum) / L  # equivalent to sqrt(sum (1/L)^2 * s_c^2/n_c)
        return {
            "L_layers": L,
            "mean": round(m_hat, 6),
            "se": round(se_hat, 6),
            "ci95_lo": round(m_hat - Z_95 * se_hat, 6),
            "ci95_hi": round(m_hat + Z_95 * se_hat, 6),
        }

    all_scores = [r["score"] for r in results.values() if r.get("score") is not None]
    overall_block = {
        "count": len(all_scores),
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
    }
    # Use stratified mean (equal-weighted across layers) — overrides naive mean
    overall_block.update(_stratified_mean_ci(layer_scores))

    layer_rows = []
    for layer_idx in sorted(layer_scores.keys()):
        scores = layer_scores[layer_idx]
        block = {
            "layer": layer_idx,
            "count": len(scores),
            "mean": sum(scores) / len(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
        block.update(_layer_ci_stats(scores))
        layer_rows.append(block)

    summary_stats = {"overall": overall_block, "layers": layer_rows}

    # Save results
    jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
    if output_file is None:
        output_file = os.path.join(jsonl_dir, "feature_activation_results.json")
    else:
        output_file = os.path.join(jsonl_dir, output_file)

    results_payload = {
        "features": results,
        "summary": summary_stats,
        "output_path": output_file,
    }

    print(f"Saving results to {output_file}...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Results saved to {output_file}")

    # Save summary separately (merge into summary.json)
    summary_path = os.path.join(output_dir if output_dir else ".", "summary.json")
    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_summary = loaded
        except Exception:
            existing_summary = {}

    existing_summary["activation"] = summary_stats
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    results_payload["summary_path"] = summary_path
    return results_payload


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process features and calculate activation token overlap scores")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL file with token lists")
    parser.add_argument("--features", type=str, required=True, help="Path to pickle file with Feature objects")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional filename to save results as JSON (saved in the same directory as the input JSONL file)",
    )
    parser.add_argument(
        "--reference-results",
        type=str,
        default="experiments/features-sample-gpt2-small-v5-32k/full/feature_activation_results.json",
        help="Optional path to existing feature_activation_results.json used to fill scores when tokens are missing",
    )
    parser.add_argument("--visualize", action="store_true", help="Save a layerwise mean score plot alongside the JSON output")

    args = parser.parse_args()

    results = process_features(
        jsonl_path=args.jsonl,
        features_file=args.features,
        output_file=args.output,
        reference_results_path=args.reference_results,
    )

    features_map = results.get("features", {})
    summary_stats = results.get("summary", {})

    overall = summary_stats.get("overall", {})

    def _fmt(stat):
        if not stat or stat.get("count", 0) == 0 or stat.get("mean") is None:
            return "n/a"
        return f"Mean: {stat['mean']:.4f}, Min: {stat['min']:.4f}, Max: {stat['max']:.4f}"

    print("\n=== Summary Statistics ===")
    print(f"Processed {len(features_map)} features")
    print(f"Score - {_fmt(overall)}")


if __name__ == "__main__":
    main()
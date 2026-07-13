#!/usr/bin/env python3
"""
Utilities for visualizing activation score outputs.

Updated:
- Activation summarize_scores supports approx/full in parallel with first/positive.
- Activation save_layerwise_plot plots first/positive/approx/full (4 lines).
- Steering summarize_steering_scores supports approx_layer/full_layer (and falls back to final_layer).
- Steering save_steering_plot plots first/positive/approx/full (4 lines).
- visualize_steering_results updated accordingly.

Backward compatible:
- If approx_score/full_score are missing, they are ignored (NaN).
- If steering approx_layer/full_layer are missing, falls back to final_layer for approx when available.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _extract_layer(feature_str: str) -> Optional[int]:
    """Extract the layer index from a feature string like 'Feature mlp-out-32k/0/549'."""
    match = re.search(r"/(\d+)/", feature_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _normalize_root(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize results to the activation block if present.
    We keep a thin shim so callers can pass either the raw payload or {"activation": payload}.
    """
    if isinstance(results, dict) and "activation" in results and isinstance(results["activation"], dict):
        return results["activation"]
    return results


def _coerce_feature_map(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the results dictionary to a mapping of feature_key -> feature_data.
    Accepts either the raw mapping or a dict containing a 'features' key.
    """
    if "features" in results and isinstance(results["features"], dict):
        return results["features"]
    return results


def _score_summary(scores: List[float]) -> Dict[str, Any]:
    """Return count/mean/min/max for a score list with JSON-serializable values."""
    if not scores:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(scores),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


# =========================
# Activation score utilities
# =========================

def summarize_scores(
    features: Dict[str, Any],
) -> Tuple[Dict[str, Any], Tuple[List[int], List[float], List[float], List[float], List[float]]]:
    """
    Compute per-layer means and overall summary statistics from feature scores.

    Expected score fields:
      - first_score
      - positive_score
      - approx_score
      - full_score

    Returns:
      summary_stats: Overall stats plus per-layer means.
      plot_data: (layers, first_means, positive_means, approx_means, full_means)
    """
    first_scores = defaultdict(list)
    positive_scores = defaultdict(list)
    approx_scores = defaultdict(list)
    full_scores = defaultdict(list)

    for feature_key, feature_data in features.items():
        if not isinstance(feature_data, dict):
            continue
        feature_str = feature_data.get("feature_str", feature_key)
        layer_idx = _extract_layer(str(feature_str))
        if layer_idx is None:
            continue

        first_score = feature_data.get("first_score")
        positive_score = feature_data.get("positive_score")
        approx_score = feature_data.get("approx_score")
        full_score = feature_data.get("full_score")

        if first_score is not None:
            first_scores[layer_idx].append(first_score)
        if positive_score is not None:
            positive_scores[layer_idx].append(positive_score)
        if approx_score is not None:
            approx_scores[layer_idx].append(approx_score)
        if full_score is not None:
            full_scores[layer_idx].append(full_score)

    layers = sorted(
        set(first_scores.keys())
        | set(positive_scores.keys())
        | set(approx_scores.keys())
        | set(full_scores.keys())
    )

    def _means(d):
        return [float(np.mean(d[l])) if d[l] else np.nan for l in layers]

    first_means = _means(first_scores)
    positive_means = _means(positive_scores)
    approx_means = _means(approx_scores)
    full_means = _means(full_scores)

    all_first = [s for scores in first_scores.values() for s in scores]
    all_positive = [s for scores in positive_scores.values() for s in scores]
    all_approx = [s for scores in approx_scores.values() for s in scores]
    all_full = [s for scores in full_scores.values() for s in scores]

    layer_rows = []
    for idx, l in enumerate(layers):
        layer_rows.append(
            {
                "layer": int(l),
                "first_mean": None if np.isnan(first_means[idx]) else float(first_means[idx]),
                "positive_mean": None if np.isnan(positive_means[idx]) else float(positive_means[idx]),
                "approx_mean": None if np.isnan(approx_means[idx]) else float(approx_means[idx]),
                "full_mean": None if np.isnan(full_means[idx]) else float(full_means[idx]),
            }
        )

    summary_stats = {
        "overall": {
            "first": _score_summary(all_first),
            "positive": _score_summary(all_positive),
            "approx": _score_summary(all_approx),
            "full": _score_summary(all_full),
        },
        "layers": layer_rows,
    }

    return summary_stats, (layers, first_means, positive_means, approx_means, full_means)


def save_layerwise_plot(
    layers: List[int],
    first_means: List[float],
    positive_means: List[float],
    approx_means: List[float],
    full_means: List[float],
    output_path: str,
) -> str:
    """Save the layerwise mean activation score plot (first/positive/approx/full)."""
    if not layers:
        raise ValueError("No layer data available for plotting.")

    layers_arr = np.array(layers)
    first_arr = np.array(first_means)
    positive_arr = np.array(positive_means)
    approx_arr = np.array(approx_means)
    full_arr = np.array(full_means)

    fig, ax = plt.subplots(figsize=(9.5, 5))

    ax.plot(layers_arr, first_arr, marker="o", markersize=5, linewidth=2, label="First (Logit Lens)", color="tomato")
    ax.plot(layers_arr, positive_arr, marker="^", markersize=5, linewidth=2, label="Positive (Token Change)", color="darkorange")
    ax.plot(layers_arr, approx_arr, marker="s", markersize=5, linewidth=2, label="Approx", color="dodgerblue")
    ax.plot(layers_arr, full_arr, marker="D", markersize=5, linewidth=2, label="Full", color="purple")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Layerwise Mean Activation Scores", fontsize=14)
    ax.set_xticks(layers_arr)
    ax.set_xlim(layers_arr.min() - 0.5, layers_arr.max() + 0.5)

    all_means = np.concatenate([first_arr, positive_arr, approx_arr, full_arr])
    valid = ~np.isnan(all_means)
    if valid.any():
        y_min = all_means[valid].min()
        y_max = all_means[valid].max()
        margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_activation_scores(results: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    """
    Convenience wrapper to summarize and plot activation scores from the given results mapping.
    """
    features = _coerce_feature_map(results)
    summary_stats, plot_data = summarize_scores(features)
    layers, first_means, positive_means, approx_means, full_means = plot_data

    if layers:
        save_layerwise_plot(
            layers=layers,
            first_means=first_means,
            positive_means=positive_means,
            approx_means=approx_means,
            full_means=full_means,
            output_path=output_path,
        )

    return summary_stats


def load_results(json_path: str, try_positive_fallback: bool = True) -> Dict[str, Any]:
    """
    Load results JSON saved by process_features_activation.py.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    root = _normalize_root(raw)
    features = _coerce_feature_map(root)

    if "activation" in raw and isinstance(raw["activation"], dict):
        raw["activation"]["features"] = features
        return raw
    root["features"] = features
    return root


# ======================
# Steering (JSONL) utils
# ======================

def load_steering_results(jsonl_path: str, try_positive_fallback: bool = True) -> Dict[str, Any]:
    """
    Load steering results JSONL.
    Returns mapping: feature_key -> entry.
    """
    features: Dict[str, Any] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            feature_str = entry.get("feature_str") or entry.get("feature") or f"feature_{len(features)}"
            features[feature_str] = entry
    return features


def summarize_steering_scores(
    features: Dict[str, Any],
    field: str = "output_score",
) -> Tuple[Dict[str, Any], Tuple[List[int], List[float], List[float], List[float], List[float]]]:
    """
    Aggregate steering scores per layer for first/positive/approx/full blocks.

    New format:
      - first_layer
      - positive_layer
      - approx_layer
      - full_layer

    Backward compatible:
      - If approx_layer missing, fall back to final_layer as approx when present.
    """
    first_scores = defaultdict(list)
    positive_scores = defaultdict(list)
    approx_scores = defaultdict(list)
    full_scores = defaultdict(list)

    for feature_data in features.values():
        if not isinstance(feature_data, dict):
            continue

        f_layer = feature_data.get("first_layer") or {}
        p_layer = feature_data.get("positive_layer") or {}
        a_layer = feature_data.get("approx_layer") or {}
        e_layer = feature_data.get("full_layer") or {}
        # old key
        l_layer = feature_data.get("final_layer") or {}

        layer_idx = None
        if isinstance(f_layer, dict) and "layer_index" in f_layer:
            try:
                layer_idx = int(f_layer["layer_index"])
            except (TypeError, ValueError):
                layer_idx = None
        elif "layer" in feature_data:
            try:
                layer_idx = int(feature_data["layer"])
            except (TypeError, ValueError):
                layer_idx = None
        if layer_idx is None:
            continue

        stats_first = (f_layer.get("stats") or {}) if isinstance(f_layer, dict) else {}
        stats_pos = (p_layer.get("stats") or {}) if isinstance(p_layer, dict) else {}
        stats_approx = (a_layer.get("stats") or {}) if isinstance(a_layer, dict) else {}
        stats_full = (e_layer.get("stats") or {}) if isinstance(e_layer, dict) else {}
        stats_final = (l_layer.get("stats") or {}) if isinstance(l_layer, dict) else {}

        v_first = stats_first.get(field)
        v_pos = stats_pos.get(field)

        v_approx = stats_approx.get(field)
        if v_approx is None:
            v_approx = stats_final.get(field)

        v_full = stats_full.get(field)

        if v_first is not None:
            first_scores[layer_idx].append(v_first)
        if v_pos is not None:
            positive_scores[layer_idx].append(v_pos)
        if v_approx is not None:
            approx_scores[layer_idx].append(v_approx)
        if v_full is not None:
            full_scores[layer_idx].append(v_full)

    layers = sorted(set(first_scores.keys()) | set(positive_scores.keys()) | set(approx_scores.keys()) | set(full_scores.keys()))

    first_means = [float(np.mean(first_scores[l])) if first_scores[l] else np.nan for l in layers]
    positive_means = [float(np.mean(positive_scores[l])) if positive_scores[l] else np.nan for l in layers]
    approx_means = [float(np.mean(approx_scores[l])) if approx_scores[l] else np.nan for l in layers]
    full_means = [float(np.mean(full_scores[l])) if full_scores[l] else np.nan for l in layers]

    all_first = [s for scores in first_scores.values() for s in scores]
    all_positive = [s for scores in positive_scores.values() for s in scores]
    all_approx = [s for scores in approx_scores.values() for s in scores]
    all_full = [s for scores in full_scores.values() for s in scores]

    layer_rows = []
    for idx, l in enumerate(layers):
        layer_rows.append(
            {
                "layer": int(l),
                "first_mean": None if np.isnan(first_means[idx]) else float(first_means[idx]),
                "positive_mean": None if np.isnan(positive_means[idx]) else float(positive_means[idx]),
                "approx_mean": None if np.isnan(approx_means[idx]) else float(approx_means[idx]),
                "full_mean": None if np.isnan(full_means[idx]) else float(full_means[idx]),
            }
        )

    summary_stats = {
        "overall": {
            "first": _score_summary(all_first),
            "positive": _score_summary(all_positive),
            "approx": _score_summary(all_approx),
            "full": _score_summary(all_full),
        },
        "layers": layer_rows,
    }

    return summary_stats, (layers, first_means, positive_means, approx_means, full_means)


def save_steering_plot(
    layers: List[int],
    first_means: List[float],
    positive_means: List[float],
    approx_means: List[float],
    full_means: List[float],
    output_path: str,
    ylabel: str = "Membership Score",
) -> str:
    """Save the steering layerwise mean score plot (first/positive/approx/full)."""
    if not layers:
        raise ValueError("No layer data available for plotting.")

    layers_arr = np.array(layers)
    first_arr = np.array(first_means)
    positive_arr = np.array(positive_means)
    approx_arr = np.array(approx_means)
    full_arr = np.array(full_means)

    fig, ax = plt.subplots(figsize=(9.5, 5))

    ax.plot(layers_arr, first_arr, marker="o", markersize=5, linewidth=2, label="First (Logit Lens)", color="tomato")
    ax.plot(layers_arr, positive_arr, marker="^", markersize=5, linewidth=2, label="Positive (Token Change)", color="darkorange")
    ax.plot(layers_arr, approx_arr, marker="s", markersize=5, linewidth=2, label="Approx", color="dodgerblue")
    ax.plot(layers_arr, full_arr, marker="D", markersize=5, linewidth=2, label="Full", color="purple")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Layerwise Mean {ylabel}", fontsize=14)
    ax.set_xticks(layers_arr)
    ax.set_xlim(layers_arr.min() - 0.5, layers_arr.max() + 0.5)

    all_means = np.concatenate([first_arr, positive_arr, approx_arr, full_arr])
    valid = ~np.isnan(all_means)
    if valid.any():
        y_min = all_means[valid].min()
        y_max = all_means[valid].max()
        margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_steering_results(
    jsonl_path: str,
    output_path: str,
    field: str = "output_score",
    try_positive_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Load steering JSONL, summarize, and save a plot.
    Supports approx_layer/full_layer (and final_layer fallback).
    """
    features = load_steering_results(jsonl_path, try_positive_fallback=try_positive_fallback)
    summary_stats, plot_data = summarize_steering_scores(features, field=field)
    layers, first_means, positive_means, approx_means, full_means = plot_data

    if layers:
        save_steering_plot(
            layers=layers,
            first_means=first_means,
            positive_means=positive_means,
            approx_means=approx_means,
            full_means=full_means,
            output_path=output_path,
            ylabel="Membership Score",
        )

    return summary_stats


# ==========================
# Interpretation utilities
# (unchanged, still 3 blocks)
# ==========================

def _extract_interp_score(block: Any) -> Optional[int]:
    if not isinstance(block, dict):
        return None
    result = block.get("result") or {}
    score = result.get("Score") if isinstance(result, dict) else None
    if score is None and isinstance(block.get("Score"), (int, float, str)):
        score = block.get("Score")
    if score is None:
        return None
    try:
        return int(score)
    except Exception:
        return None


def _extract_interp_scores(entry: Dict[str, Any]) -> Dict[str, Optional[int]]:
    def pick(key: str, fallback_key: str) -> Optional[int]:
        direct = entry.get(key)
        if direct is not None:
            try:
                return int(direct)
            except Exception:
                pass
        block = entry.get(fallback_key)
        return _extract_interp_score(block)

    return {
        "first": pick("first_score", "first_tokens"),
        "positive": pick("positive_score", "positive_tokens"),
        "last": pick("last_score", "last_tokens"),
    }


def load_interp_results(jsonl_path: str, try_positive_fallback: bool = True) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            feature_str = entry.get("feature_str") or entry.get("feature") or f"feature_{len(features)}"
            scores = _extract_interp_scores(entry)
            features[feature_str] = {
                "feature_str": feature_str,
                "first_score": scores["first"],
                "positive_score": scores["positive"],
                "final_score": scores["last"],
            }
    return features


def summarize_interp_scores(
    features: Dict[str, Any],
) -> Tuple[Dict[str, Any], Tuple[List[int], List[float], List[float], List[float]]]:
    first_scores = defaultdict(list)
    positive_scores = defaultdict(list)
    final_scores = defaultdict(list)

    for feature_key, feature_data in features.items():
        if not isinstance(feature_data, dict):
            continue
        feature_str = feature_data.get("feature_str", feature_key)
        layer_idx = _extract_layer(str(feature_str))
        if layer_idx is None:
            continue

        first_score = feature_data.get("first_score")
        positive_score = feature_data.get("positive_score")
        final_score = feature_data.get("final_score")

        if first_score is not None:
            first_scores[layer_idx].append(first_score)
        if positive_score is not None:
            positive_scores[layer_idx].append(positive_score)
        if final_score is not None:
            final_scores[layer_idx].append(final_score)

    layers = sorted(set(first_scores.keys()) | set(positive_scores.keys()) | set(final_scores.keys()))

    first_means = [float(np.mean(first_scores[l])) if first_scores[l] else np.nan for l in layers]
    positive_means = [float(np.mean(positive_scores[l])) if positive_scores[l] else np.nan for l in layers]
    final_means = [float(np.mean(final_scores[l])) if final_scores[l] else np.nan for l in layers]

    all_first = [s for scores in first_scores.values() for s in scores]
    all_positive = [s for scores in positive_scores.values() for s in scores]
    all_final = [s for scores in final_scores.values() for s in scores]

    layer_rows = []
    for idx, l in enumerate(layers):
        layer_rows.append(
            {
                "layer": int(l),
                "first_mean": None if np.isnan(first_means[idx]) else float(first_means[idx]),
                "positive_mean": None if np.isnan(positive_means[idx]) else float(positive_means[idx]),
                "final_mean": None if np.isnan(final_means[idx]) else float(final_means[idx]),
            }
        )

    summary_stats = {
        "overall": {
            "first": _score_summary(all_first),
            "positive": _score_summary(all_positive),
            "final": _score_summary(all_final),
        },
        "layers": layer_rows,
    }

    return summary_stats, (layers, first_means, positive_means, final_means)


def save_interp_plot(
    layers: List[int],
    first_means: List[float],
    positive_means: List[float],
    final_means: List[float],
    output_path: str,
    ylabel: str = "Mean Interpretability Score",
) -> str:
    if not layers:
        raise ValueError("No layer data available for plotting.")

    layers_arr = np.array(layers)
    first_arr = np.array(first_means)
    positive_arr = np.array(positive_means)
    final_arr = np.array(final_means)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(layers_arr, first_arr, marker="o", linewidth=2, label="Logit Lens", color="tomato")
    ax.plot(layers_arr, positive_arr, marker="^", linewidth=2, label="Positive Lens", color="darkorange")
    ax.plot(layers_arr, final_arr, marker="s", linewidth=2, label="Query Lens", color="dodgerblue")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Layerwise Mean Interpretability Score", fontsize=14)
    ax.set_xticks(layers_arr)
    ax.set_xlim(layers_arr.min() - 0.5, layers_arr.max() + 0.5)

    all_means = np.concatenate([first_arr, positive_arr, final_arr])
    valid = ~np.isnan(all_means)
    if valid.any():
        y_min = all_means[valid].min()
        y_max = all_means[valid].max()
        margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_interp_results(
    jsonl_path: str,
    output_path: str,
    try_positive_fallback: bool = True,
) -> Dict[str, Any]:
    features = load_interp_results(jsonl_path, try_positive_fallback=try_positive_fallback)
    summary_stats, plot_data = summarize_interp_scores(features)
    layers, first_means, positive_means, final_means = plot_data

    if layers:
        save_interp_plot(
            layers=layers,
            first_means=first_means,
            positive_means=positive_means,
            final_means=final_means,
            output_path=output_path,
        )

    return summary_stats
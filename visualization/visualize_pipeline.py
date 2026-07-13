#!/usr/bin/env python3
"""
Generate input score and output score bar plots from per-baseline experiment results.

Reads evaluation results from each baseline subdirectory under the experiment dir
and produces grouped bar charts following the visual style of notebooks/visualize_v2.ipynb.

Usage:
    python visualization/visualize_pipeline.py --experiment-dir experiments/features-sample-xxx
    python visualization/visualize_pipeline.py --experiment-dir experiments/features-sample-xxx --exclude-last-layer
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Constants
# =========================================================

# All known baselines -> display name (plot order)
BASELINE_DISPLAY = OrderedDict([
    ("logit_lens_key",          "Logit Lens (Key)"),
    ("query_lens_key_approx",   "Query Lens Key (Approx)"),
    ("query_lens_key",    "Query Lens Key (Full)"),
    ("logit_lens_value",        "Logit Lens (Value)"),
    ("query_lens_value_approx", "Query Lens Value (Approx)"),
    ("query_lens_value",  "Query Lens Value (Full)"),
    ("token_change",            "Token Change"),
    ("zero_out",                "Zero Out"),
    ("tuned_lens",              "Tuned Lens"),
])

DEFAULT_BASELINES = list(BASELINE_DISPLAY.keys())

COLOR_DICT = {
    "Logit Lens (Key)":           "salmon",
    "Query Lens Key (Approx)":    "deepskyblue",
    "Query Lens Key (Full)":     "steelblue",
    "Logit Lens (Value)":         "tomato",
    "Query Lens Value (Approx)":  "dodgerblue",
    "Query Lens Value (Full)":   "navy",
    "Token Change":               "darkorange",
    "Zero Out":                   "mediumpurple",
    "Tuned Lens":                 "mediumseagreen",
}

HATCH_DICT = {
    "Logit Lens (Key)":           "//",
    "Query Lens Key (Approx)":    "//",
    "Query Lens Key (Full)":     "//",
    "Logit Lens (Value)":         None,
    "Query Lens Value (Approx)":  None,
    "Query Lens Value (Full)":   None,
    "Token Change":               None,
    "Zero Out":                   None,
    "Tuned Lens":                 None,
}


# =========================================================
# Utilities
# =========================================================

def extract_layer(feature_str: str) -> Optional[int]:
    """Extract the layer index from a feature string like 'Feature mlp-out-32k/5/549'."""
    match = re.search(r"/(\d+)/", feature_str)
    if match:
        return int(match.group(1))
    return None


def detect_baselines(experiment_dir: str, candidates: Optional[OrderedDict] = None) -> OrderedDict:
    """Detect which baselines have results in the experiment directory."""
    if candidates is None:
        candidates = BASELINE_DISPLAY
    found = OrderedDict()
    for baseline, display_name in candidates.items():
        baseline_dir = os.path.join(experiment_dir, baseline)
        if os.path.isdir(baseline_dir):
            found[baseline] = display_name
    return found


def make_layer_groups(all_layers: List[int]) -> OrderedDict:
    """Auto-detect layer grouping: groups if many layers, per-layer if few."""
    n = len(all_layers)
    if n == 0:
        return OrderedDict()

    max_layer = max(all_layers)

    if n > 6:
        # Split into 3 roughly equal groups
        group_size = max(1, (max_layer + 1) // 3)
        remainder = (max_layer + 1) % 3
        groups = OrderedDict()
        labels = ["Early", "Middle", "Late"]
        start = 0
        for i, label in enumerate(labels):
            end = start + group_size + (1 if i < remainder else 0)
            layers_in = sorted(l for l in all_layers if start <= l < end)
            if layers_in:
                groups[f"{label} [{start}, {end})"] = layers_in
            start = end
        return groups
    else:
        return OrderedDict(
            (f"Layer {l}", [l]) for l in sorted(all_layers)
        )


# =========================================================
# Data loading
# =========================================================

def load_input_scores(
    experiment_dir: str,
    baselines: OrderedDict,
) -> Dict[str, Dict[int, List[float]]]:
    """Load input scores from feature_activation_results.json per baseline.

    Returns: {display_name: {layer: [score, ...]}}
    """
    scores: Dict[str, Dict[int, List[float]]] = {}
    for baseline, display_name in baselines.items():
        path = os.path.join(experiment_dir, baseline, "feature_activation_results.json")
        if not os.path.exists(path):
            print(f"  [warn] Input scores not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", {})
        layer_scores: Dict[int, List[float]] = defaultdict(list)
        for feature_str, feature_data in features.items():
            fstr = feature_data.get("feature_str", feature_str)
            layer = extract_layer(fstr)
            score = feature_data.get("score")
            if layer is not None and score is not None:
                layer_scores[layer].append(float(score))

        scores[display_name] = dict(layer_scores)
    return scores


def load_output_scores(
    experiment_dir: str,
    baselines: OrderedDict,
) -> Dict[str, Dict[int, List[float]]]:
    """Load output scores from feature_steering_results.jsonl per baseline.

    Returns: {display_name: {layer: [output_score, ...]}}
    """
    scores: Dict[str, Dict[int, List[float]]] = {}
    for baseline, display_name in baselines.items():
        path = os.path.join(experiment_dir, baseline, "feature_steering_results.jsonl")
        if not os.path.exists(path):
            print(f"  [warn] Output scores not found: {path}")
            continue

        layer_scores: Dict[int, List[float]] = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                feature_str = entry.get("feature_str", "")
                layer = extract_layer(feature_str)
                stats = entry.get("result_layer", {}).get("stats", {})
                # Primary field is "score" (normalized-diff of length-normalized
                # frequency). Historical fallbacks in priority order:
                #   - "frequency_score": additive frequency delta (earlier rev)
                #   - "membership_score": paper binary-presence scorer
                #   - "output_score":     rank-percentile scorer
                ms = stats.get("score")
                if ms is None:
                    ms = stats.get("frequency_score")
                if ms is None:
                    ms = stats.get("membership_score")
                if ms is None:
                    ms = stats.get("output_score")
                if layer is not None and ms is not None:
                    layer_scores[layer].append(float(ms))

        scores[display_name] = dict(layer_scores)

    return scores


# =========================================================
# Plotting
# =========================================================

def _collect_all_layers(scores: Dict[str, Dict[int, List[float]]]) -> List[int]:
    """Collect sorted unique layers across all methods."""
    layers = set()
    for ls in scores.values():
        layers.update(ls.keys())
    return sorted(layers)


def _stratified_fold_se(
    layer_to_scores: Dict[int, List[float]],
    k: int = 5,
    seed: int = 42,
) -> float:
    """Stratified k-fold SE: within each layer, shuffle and assign features
    round-robin to folds so every fold has equal layer representation."""
    import random, math
    rng = random.Random(seed)
    fold_members: List[List[float]] = [[] for _ in range(k)]
    for layer_idx in sorted(layer_to_scores.keys()):
        vals = list(layer_to_scores[layer_idx])
        rng.shuffle(vals)
        for i, v in enumerate(vals):
            fold_members[i % k].append(v)
    fold_means = [sum(m) / len(m) for m in fold_members if m]
    if len(fold_means) < 2:
        return 0.0
    mu = sum(fold_means) / len(fold_means)
    var = sum((x - mu) ** 2 for x in fold_means) / (len(fold_means) - 1)
    return math.sqrt(var) / math.sqrt(len(fold_means))


def _compute_group_stats(
    scores: Dict[str, Dict[int, List[float]]],
    methods: List[str],
    layer_groups: OrderedDict,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Compute per-group mean and stratified-fold SE error bars (in %, clipped at 0)."""
    x_labels = list(layer_groups.keys())
    vals = {m: [] for m in methods}
    lower_errs = {m: [] for m in methods}
    upper_errs = {m: [] for m in methods}

    for group_name, group_layers in layer_groups.items():
        for method in methods:
            method_scores = scores.get(method, {})
            layer_subset = {l: method_scores.get(l, []) for l in group_layers if method_scores.get(l)}

            if layer_subset:
                all_vals = [v for vs in layer_subset.values() for v in vs]
                mean = float(np.mean(all_vals)) * 100
                se = _stratified_fold_se(layer_subset) * 100
            else:
                mean = float("nan")
                se = 0.0

            vals[method].append(mean)
            lower = max(0.0, mean if mean - se < 0 else se)
            lower_errs[method].append(lower)
            upper_errs[method].append(se)

    return x_labels, vals, lower_errs, upper_errs


def plot_bar_chart(
    scores: Dict[str, Dict[int, List[float]]],
    score_type: str,
    output_path: str,
    layer_groups: Optional[OrderedDict] = None,
    baselines: Optional[OrderedDict] = None,
    exclude_last_layer: bool = False,
):
    """Create grouped bar chart for input or output scores."""
    if baselines is None:
        baselines = BASELINE_DISPLAY
    methods = [m for m in baselines.values() if m in scores]
    if not methods:
        print(f"  [warn] No data to plot for {score_type}")
        return

    all_layers = _collect_all_layers(scores)
    if not all_layers:
        print(f"  [warn] No layers found for {score_type}")
        return

    # Compute layer groups from ALL layers (including last) to keep boundaries stable
    if layer_groups is None:
        layer_groups = make_layer_groups(all_layers)

    # Then exclude last layer data from scores and groups
    if exclude_last_layer and all_layers:
        last = max(all_layers)
        for display_name in scores:
            scores[display_name].pop(last, None)
        for group_name in layer_groups:
            layer_groups[group_name] = [l for l in layer_groups[group_name] if l != last]

    x_labels, vals, lower_errs, upper_errs = _compute_group_stats(
        scores, methods, layer_groups
    )

    n_groups = len(x_labels)
    n_methods = len(methods)
    bar_width = 0.15
    # Space groups so bars don't overlap: total bar span + gap between groups
    group_width = n_methods * bar_width
    gap = max(bar_width, group_width * 0.3)
    group_spacing = group_width + gap
    x = np.arange(n_groups) * group_spacing

    # Position offsets centered on each group
    offsets = np.linspace(
        -(n_methods - 1) * bar_width / 2,
        (n_methods - 1) * bar_width / 2,
        n_methods,
    )

    # Wider figure scaled to actual plot width
    fig_width = max(10, n_groups * group_spacing * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.375))
    bar_style = dict(edgecolor="black", linewidth=1.2)

    for i, method in enumerate(methods):
        yerr = np.array([lower_errs[method], upper_errs[method]])
        hatch = HATCH_DICT.get(method)
        ax.bar(
            x + offsets[i],
            vals[method],
            bar_width,
            label=method,
            color=COLOR_DICT.get(method, "gray"),
            hatch=hatch if hatch else "",
            yerr=yerr,
            capsize=6,
            **bar_style,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=25)
    ax.set_ylabel(score_type, fontsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_legend(output_path: str, methods: Optional[List[str]] = None):
    """Save a standalone legend figure matching the notebook style."""
    if methods is None:
        methods = list(BASELINE_DISPLAY.values())

    handles = []
    labels = []
    for method in methods:
        hatch = HATCH_DICT.get(method)
        handles.append(
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=COLOR_DICT.get(method, "gray"),
                hatch=hatch if hatch else "",
                edgecolor="black",
                linewidth=1.2,
            )
        )
        labels.append(method)

    fig_legend = plt.figure(figsize=(12, 0.65))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=len(methods),
        frameon=False,
        fontsize=25,
        handleheight=1.0,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.2,
        borderaxespad=0.0,
    )
    fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig_legend.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig_legend)
    print(f"  Saved: {output_path}")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate input/output score bar plots from per-baseline experiment results."
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to experiment directory (e.g., experiments/features-sample-xxx)",
    )
    parser.add_argument(
        "--baselines", type=str, default=None,
        help="Comma-separated list of baselines to include (default: all known baselines)",
    )
    parser.add_argument(
        "--exclude-last-layer", action="store_true",
        help="Exclude the last layer from output score plots",
    )
    args = parser.parse_args()

    # Filter BASELINE_DISPLAY to only requested baselines
    if args.baselines is not None:
        requested = [b.strip() for b in args.baselines.split(",")]
        unknown = [b for b in requested if b not in BASELINE_DISPLAY]
        if unknown:
            print(f"Warning: Unknown baselines ignored: {', '.join(unknown)}")
        active_baselines = OrderedDict(
            (k, v) for k, v in BASELINE_DISPLAY.items() if k in requested
        )
    else:
        active_baselines = BASELINE_DISPLAY

    experiment_dir = args.experiment_dir
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Detect available baselines
    baselines = detect_baselines(experiment_dir, active_baselines)
    if not baselines:
        print(f"Error: No baseline directories found in {experiment_dir}")
        print(f"Expected subdirectories: {', '.join(active_baselines.keys())}")
        sys.exit(1)

    print(f"  Found baselines: {', '.join(baselines.keys())}")

    base_name = os.path.basename(os.path.normpath(experiment_dir))
    figures_dir = os.path.join(experiment_dir, "figures")

    # ---- Input Score ----
    print("\n  Loading input scores...")
    input_scores = load_input_scores(experiment_dir, baselines)
    if input_scores:
        plot_bar_chart(
            input_scores,
            score_type="Input Score (%)",
            output_path=os.path.join(figures_dir, f"{base_name}_input_score.pdf"),
            baselines=active_baselines,
        )
    else:
        print("  [warn] No input score data found")

    # ---- Output Score ----
    print("\n  Loading output scores...")
    output_scores = load_output_scores(
        experiment_dir, baselines,
    )
    if output_scores:
        plot_bar_chart(
            output_scores,
            score_type="Output Score (%)",
            output_path=os.path.join(figures_dir, f"{base_name}_output_score.pdf"),
            baselines=active_baselines,
            exclude_last_layer=args.exclude_last_layer,
        )
    else:
        print("  [warn] No output score data found")

    # ---- Legend ----
    available_methods = [
        name for name in BASELINE_DISPLAY.values()
        if name in input_scores or name in output_scores
    ]
    if available_methods:
        save_legend(
            output_path=os.path.join(figures_dir, "legend.pdf"),
            methods=available_methods,
        )

    print(f"\n  All visualizations saved to {figures_dir}/")


if __name__ == "__main__":
    main()

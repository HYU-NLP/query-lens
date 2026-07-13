#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# run_pipeline.sh — Full evaluation pipeline
#
# Steps:
#   1. Run token interpretation baselines (analysis/run.py)
#   2. Run input score evaluation (evaluation/input_score.py)
#   3. Run output score evaluation (evaluation/output_score.py)
#   4. Generate visualizations (visualization/visualize_pipeline.py)
#
# Usage:
#   ./run_pipeline.sh -c CONFIG -f FEATURES_PKL [-g GPUS] [-b BASELINES]
#
# Examples:
#   ./run_pipeline.sh -c gpt2-small -f features/features-sample-gpt2-small-v5-32k.pkl -g 4
#   ./run_pipeline.sh -c gemma3-270m -f features/features-sample-gemma-3-270m-65k-res.pkl -g 2
#   ./run_pipeline.sh -c gpt2-small -f features/my.pkl --skip-analysis  # eval + viz only
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DEFAULT_BASELINES="logit_lens_key,logit_lens_value,query_lens_key_approx,query_lens_value_approx,query_lens_key,query_lens_value,token_change,zero_out,tuned_lens"

usage() {
    cat <<EOF
Usage: $0 -c CONFIG -f FEATURES_PKL [OPTIONS]

Full evaluation pipeline: baselines -> evaluation -> visualization.

Required:
  -c, --config CONFIG       Hydra config name (e.g., gpt2-small, gemma3-270m)
  -f, --features PATH       Path to features pickle file

Options:
  -g, --gpus N              Number of GPUs for torchrun (default: 1)
  -b, --baselines LIST      Comma-separated baselines (default: all 5)
  --skip-analysis           Skip Step 1 (baseline analysis)
  --skip-eval               Skip Steps 2-3 (input/output scoring)
  --skip-viz                Skip Step 4 (visualization)
  --exclude-last-layer      Exclude the last layer from output score plots
  -h, --help                Show this help

Default baselines: $DEFAULT_BASELINES
EOF
    exit 0
}

# ---- Parse arguments ----
CONFIG=""
FEATURES_PKL=""
GPUS=1
BASELINES="$DEFAULT_BASELINES"
SKIP_ANALYSIS=false
SKIP_EVAL=false
SKIP_VIZ=false
EXCLUDE_LAST=""
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)       CONFIG="$2"; shift 2;;
        -f|--features)     FEATURES_PKL="$2"; shift 2;;
        -g|--gpus)         GPUS="$2"; shift 2;;
        -b|--baselines)    BASELINES="$2"; shift 2;;
        --skip-analysis)   SKIP_ANALYSIS=true; shift;;
        --skip-eval)       SKIP_EVAL=true; shift;;
        --skip-viz)        SKIP_VIZ=true; shift;;
        --exclude-last-layer) EXCLUDE_LAST="--exclude-last-layer"; shift;;
        -h|--help)         usage;;
        *)                 HYDRA_OVERRIDES+=("$1"); shift;;
    esac
done

# ---- Validate ----
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    usage
fi
if [[ -z "$FEATURES_PKL" ]]; then
    echo "Error: --features is required"
    usage
fi
if [[ ! -f "$FEATURES_PKL" ]]; then
    echo "Error: Features file not found: $FEATURES_PKL"
    exit 1
fi

# ---- Derived paths ----
FEATURES_BASE=$(basename "$FEATURES_PKL" .pkl)
EXPERIMENT_DIR="experiments/${FEATURES_BASE}"

echo ""
echo "============================================"
echo "  Pipeline Configuration"
echo "============================================"
echo "  Config:      $CONFIG"
echo "  Features:    $FEATURES_PKL"
echo "  GPUs:        $GPUS"
echo "  Baselines:   $BASELINES"
echo "  Experiment:  $EXPERIMENT_DIR"
echo "============================================"
echo ""

IFS=',' read -ra BASELINE_ARRAY <<< "$BASELINES"

# =========================================================
# Step 1: Run baseline analysis
# =========================================================
if [[ "$SKIP_ANALYSIS" == false ]]; then
    echo "============================================"
    echo "  Step 1: Running baseline analysis"
    echo "============================================"
    for baseline in "${BASELINE_ARRAY[@]}"; do
        BASELINE_DIR="$EXPERIMENT_DIR/$baseline"
        JSONL_PATH="$BASELINE_DIR/feature_analysis.jsonl"

        # Check if already complete (resume-friendly)
        if [[ -f "$JSONL_PATH" ]]; then
            echo "  [resume] Found existing $JSONL_PATH, analysis will resume"
        fi

        echo "  -> Running: $baseline"
        MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
        torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
            "$SCRIPT_DIR/analysis/run.py" \
            -cn "$CONFIG" \
            baseline="$baseline" \
            features_pkl="$FEATURES_PKL" \
            "${HYDRA_OVERRIDES[@]}"
        echo "  <- Done: $baseline"
        echo ""
    done
else
    echo "  [skip] Step 1: baseline analysis"
fi

# =========================================================
# Step 2: Run input score evaluation
# =========================================================
if [[ "$SKIP_EVAL" == false ]]; then
    echo "============================================"
    echo "  Step 2: Running input score evaluation"
    echo "============================================"
    for baseline in "${BASELINE_ARRAY[@]}"; do
        JSONL_PATH="$EXPERIMENT_DIR/$baseline/feature_analysis.jsonl"
        if [[ ! -f "$JSONL_PATH" ]]; then
            echo "  [warn] JSONL not found for $baseline, skipping input score"
            continue
        fi
        echo "  -> Input score: $baseline"
        python "$SCRIPT_DIR/evaluation/input_score.py" \
            --jsonl "$JSONL_PATH" \
            --features "$FEATURES_PKL"
        echo "  <- Done: $baseline"
    done

    # =========================================================
    # Step 3: Run output score evaluation
    # =========================================================
    echo ""
    echo "============================================"
    echo "  Step 3: Running output score evaluation"
    echo "============================================"
    for baseline in "${BASELINE_ARRAY[@]}"; do
        JSONL_PATH="$EXPERIMENT_DIR/$baseline/feature_analysis.jsonl"
        if [[ ! -f "$JSONL_PATH" ]]; then
            echo "  [warn] JSONL not found for $baseline, skipping output score"
            continue
        fi
        echo "  -> Output score: $baseline"
        python "$SCRIPT_DIR/evaluation/output_score.py" \
            --jsonl "$JSONL_PATH" \
            --features "$FEATURES_PKL" \
            $EXCLUDE_LAST
        echo "  <- Done: $baseline"
    done
else
    echo "  [skip] Steps 2-3: evaluation scoring"
fi

# =========================================================
# Step 4: Visualization
# =========================================================
if [[ "$SKIP_VIZ" == false ]]; then
    echo ""
    echo "============================================"
    echo "  Step 4: Generating visualizations"
    echo "============================================"
    python "$SCRIPT_DIR/visualization/visualize_pipeline.py" \
        --experiment-dir "$EXPERIMENT_DIR" \
        --baselines "$BASELINES" \
        $EXCLUDE_LAST
    echo "  <- Visualizations saved to $EXPERIMENT_DIR/figures/"
else
    echo "  [skip] Step 4: visualization"
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results: $EXPERIMENT_DIR"
echo "============================================"

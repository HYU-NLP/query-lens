# Query Lens

#### Official Repository for "Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects" [[Paper Link (arXiv)]](https://arxiv.org/abs/2606.07617)

##### Hwiyeong Lee, Ingyu Bang, Uiji Hwang, Hyelim Lim and Taeuk Kim. *Accepted to ICML 2026*.

## Summary

Query Lens extends Logit Lens to enable more comprehensive and faithful interpretations of sparse features (SAE / transcoder latents). By jointly considering encoder-side **key** features and decoder-side **value** features, it identifies both the inputs that activate a feature and the outputs it promotes, while accounting for **indirect, module-mediated effects** that arise when the feature is processed by downstream modules — going beyond the direct effect captured by Logit Lens.

## Method Overview

Each analysis run produces one token set for one baseline, configured via Hydra YAML. Baselines are combinations of three components:

- **Feature vector**: `value` (W_dec / W_out) or `key` (W_enc / W_in)
- **Stream transition**: `identity`, `first_order`, `full`, `token_change`, `zero_out`, `tuned_lens`
- **Readout**: `unembed` (Uᵀ), `embed` (W_Eᵀ), `embed_cosine_centered` (centered cosine)

Available preset baselines:

| Preset | Feature Vector | Stream Transition | Readout |
|--------|---------------|-------------------|---------|
| `logit_lens_value` | value | identity | unembed |
| `logit_lens_key` | key | identity | embed |
| `query_lens_value` | value | full | unembed |
| `query_lens_key` | key | full | embed_cosine_centered |
| `token_change` | value | token_change | — |
| `zero_out` | value | zero_out | — |
| `tuned_lens` | value | tuned_lens | unembed |

## Repository Structure

```
├── util.py                        # Shared utilities (Feature, Activation, SAE cache, distributed helpers)
├── conf/                          # Hydra YAML configs — one per model/SAE setting used in the paper
├── analysis/
│   ├── common.py                  # Shared primitives: ResidCache, JVP/VJP ops, scoring, BASELINE_REGISTRY
│   ├── run.py                     # Unified analysis entry point
│   ├── jvp.py                     # Value-side (forward / JVP) analysis
│   ├── vjp.py                     # Key-side (backward / VJP) analysis
│   └── subspace_channel.py        # Low-rank map training for the Subspace Channel Hypothesis (Sec. 6, App. J)
├── evaluation/
│   ├── input_score.py             # Input score I(T) — activation token overlap (Sec. 5.1)
│   ├── output_score.py            # Output score O(T) — steering membership (Sec. 5.2)
│   └── interp_score.py            # LLM-based interpretability scoring (Table 2, App. K)
├── data/
│   ├── write_feature_sample.py    # Feature sampling from SAE releases
│   ├── manual_activations.py      # Activation collection (App. I.1)
│   └── generation_prompts.json    # Neutral prefixes for steering (Table 11)
├── visualization/                 # Plotting utilities + Gradio feature viewer
├── notebooks/
│   ├── map_analysis.ipynb         # Overlap statistics & permutation tests for learned maps (Fig. 5, App. J.2)
│   └── plot_rel_var.ipynb         # Relative-variance analysis (App. E)
└── run_pipeline.sh                # Full pipeline: analysis → evaluation → visualization
```

## Setup

```bash
pip install -r requirements.txt
```

Interpretability scoring (`evaluation/interp_score.py`) calls the OpenAI API; set `OPENAI_API_KEY` in your environment.

## Models and Checkpoints

Main experiments (paper Table 1; HF repositories in Table 3):

| Setting | Config | Model Repository | SAE Repository |
|---------|--------|------------------|----------------|
| GPT-2 Small (32K) | `gpt2-small` | openai-community/gpt2 | jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs |
| Gemma-3-270M (65K) | `gemma3-270m` | google/gemma-3-270m | google/gemma-scope-2-270m-pt |
| Gemma-3-1B (65K) | `gemma3-1b` | google/gemma-3-1b-pt | google/gemma-scope-2-1b-pt |
| Qwen-3-1.7B (32K) | `qwen3-1.7b-base` | Qwen/Qwen3-1.7B-Base | Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100 |

Additional configurations (paper Appendix D; HF repositories in Table 4):

| Setting | Config | Dictionary Repository |
|---------|--------|-----------------------|
| GPT-2 Small (128K) | `gpt2-small` | jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs |
| Gemma-3-270M / 1B (16K), Gemma-3-4B (16K & 65K) | `gemma3-270m` / `gemma3-1b` / `gemma3-4b` | google/gemma-scope-2-{270m,1b,4b}-pt |
| Qwen-3-0.6B / 1.7B / 4B transcoders | `qwen3-0.6b` / `qwen3-1.7b` / `qwen3-4b` | mwhanna/qwen3-{0.6b,1.7b}-transcoders-lowl0, mwhanna/qwen3-4b-transcoders |

## Pipeline

The full pipeline (analysis → input/output scoring → visualization) is wrapped by `run_pipeline.sh`:

```bash
./run_pipeline.sh -c gpt2-small -f features/features-sample-gpt2-small-v5-32k.pkl -g 4
```

Or run the stages individually:

```bash
# 1. Sample features
python data/write_feature_sample.py --model gpt2-small --amount 100 --write --seed 42

# 2. Run one baseline analysis (Hydra config from conf/)
torchrun --nproc_per_node=N analysis/run.py -cn gpt2-small baseline=query_lens_key

# 3. Evaluate
python evaluation/input_score.py  --jsonl path/to/feature_analysis.jsonl --features path/to/features.pkl
python evaluation/output_score.py --jsonl path/to/feature_analysis.jsonl --features path/to/features.pkl
python evaluation/interp_score.py path/to/feature_analysis.jsonl --model gpt-5-nano

# 4. Visualize
python visualization/app.py
```

Analysis outputs are written under `experiments/<features-sample-...>/<baseline>/`, and figures under `figures/` (both git-ignored).

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{
lee2026query,
title={Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects},
author={Hwiyeong Lee and Ingyu Bang and Uiji Hwang and Hyelim Lim and Taeuk Kim},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=6t9xJWFjkq}
}
```

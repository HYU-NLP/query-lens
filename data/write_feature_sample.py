#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.multiprocessing as mp
from loguru import logger
from sae_lens import SAE, HookedSAETransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from util import SAEGPUCache, sort_features_by_layer, Activation, Feature, load_hooked_transformer
from data.manual_activations import collect_manual_activations, load_pile_corpus

NEURONPEDIA_EXPORTS_DIR = ROOT_DIR / "neuronpedia_exports"
FEATURE_SAMPLE_DIR = ROOT_DIR / "features"
TRANSLUCE_DIR = ROOT_DIR / "train"
TRANSLUCE_EXPORTS_DIR = ROOT_DIR / "transluce_exports"
TRANSLUCE_LAYER_PKL_TEMPLATE = "layer{layer}_transluce_activations.pkl"

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def size_to_int(size: str) -> int:
    size = size.lower()
    if not size.endswith("k"):
        raise ValueError(f"Invalid SAE size '{size}'")
    num = int(size[:-1])
    raw_value = num * (2**10)
    true_power = round(math.log2(raw_value))
    return 2**true_power


def parse_layer_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    parts = [part.strip() for part in value.split(",")]
    layers: List[int] = []
    for part in parts:
        if part == "":
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid layer index '{part}'. Expected comma-separated integers.")
        layers.append(int(part))
    return layers or None


def validate_layers(layers: Optional[List[int]], total_layers: int) -> Optional[List[int]]:
    if layers is None:
        return None
    unique_layers = sorted(set(layers))
    for layer in unique_layers:
        if layer < 0 or layer >= total_layers:
            raise ValueError(f"Layer index {layer} out of bounds [0, {total_layers - 1}]")
    return unique_layers


def write_pkl_file(data, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _sample_no_sae_features_worker(args) -> List[Feature]:
    layer, amount, model_id, layer_type, d_mlp, seed, tokenizer = args
    rng = random.Random(seed) if seed is not None else random
    population = range(d_mlp)
    sample = rng.sample(population, amount)
    features = [
        Feature(model_id, rand_feature, layer, layer_type)
        for rand_feature in sample
    ]

    for feature in features:
        activations = get_act_transluce(
            feature.layer,
            feature.feature,
            tokenizer=tokenizer,
        )
        if activations:
            feature.activations = activations

    return features


def _sample_sae_features_worker(job: Dict) -> List[Feature]:
    amount = job["amount"]
    size_int = job["size_int"]
    threshold = job.get("threshold", 20)
    prefer_manual = job.get("prefer_manual", False)

    rng = random.Random(job["seed"]) if job["seed"] is not None else random
    population = list(range(size_int))

    max_attempts = min(amount * 10, size_int)
    sampled_indices = rng.sample(population, min(max_attempts, len(population)))

    features: List[Feature] = []
    seen_features = set()

    for rand_feature in sampled_indices:
        if len(features) >= amount:
            break

        if rand_feature in seen_features:
            continue

        feature = Feature(
            job["model_id"],
            rand_feature,
            job["layer"],
            job["sae_type"],
            sae_id=job["sae_id"],
            sae_release=job["sae_release"],
            size=job["size"],
        )

        # manual mode면 Neuronpedia를 아예 조회하지 않음
        if not prefer_manual:
            activations = get_act_neuronpedia(
                job["model_id"],
                job["np_sae_id"],
                feature.feature,
                bin_contains_max=0.01,
            )
            if activations:
                feature = Feature(
                    feature.model_id,
                    feature.feature,
                    feature.layer,
                    feature.type,
                    activations,
                    feature.sae_id,
                    feature.sae_release,
                    feature.size,
                )

        act_count = len(feature.activations or [])
        if act_count >= threshold:
            features.append(feature)
            seen_features.add(rand_feature)

    if len(features) < amount:
        logger.warning(
            f"Only found {len(features)}/{amount} features with positive max activations >= {threshold} "
            f"for layer {job['layer']} (sae_type={job['sae_type']}, size={job['size']}, prefer_manual={prefer_manual})"
        )

    return features


def get_sae_export_dir(model_id: str, sae_id: str) -> Path:
    sae_id_no_layer = sae_id.split("-", 1)[1]
    model_dir = f"{model_id}-{sae_id_no_layer}"
    return NEURONPEDIA_EXPORTS_DIR / model_dir / sae_id


def get_feature_export_json(model_export_dir: Path, feature: int) -> Optional[Path]:
    if not model_export_dir.exists():
        return None
    for json_file in model_export_dir.glob("*.json"):
        try:
            start, end = map(int, json_file.stem.split("-"))
        except ValueError:
            continue
        if start <= feature < end:
            return json_file
    return None


def get_feature_json_index(json_file: Path, feature: int) -> int:
    start, _ = map(int, json_file.stem.split("-"))
    return feature - start


def get_feature_json_data(json_path: Path, feature: int) -> Dict:
    with json_path.open("r") as handle:
        data = json.load(handle)
    index = get_feature_json_index(json_path, feature)
    return data[index]


def get_export_data(model_id: str, sae_id: str, feature: int) -> Dict:
    model_export_dir = get_sae_export_dir(model_id, sae_id)
    json_file = get_feature_export_json(model_export_dir, feature)
    if json_file is None:
        return {}
    try:
        return get_feature_json_data(json_file, feature)
    except (IndexError, json.JSONDecodeError):
        return {}


def json_to_activations(activations_data) -> List[Activation]:
    activations = []
    for entry in activations_data:
        tokens = entry["tokens"]
        values = entry["values"]
        token_values = list(zip(tokens, values))
        activation = Activation(
            id="",
            token_values=token_values,
            max_value=entry["maxValue"],
            min_value=entry["minValue"],
            max_value_token_index=entry.get("maxValueTokenIndex"),
            loss_values=entry.get("lossValues"),
        )
        activations.append(activation)
    return deduplicate_activations(activations)


def deduplicate_activations(activations: List[Activation]) -> List[Activation]:
    seen = set()
    unique: List[Activation] = []
    for act in activations:
        key = str(act)
        if key in seen:
            continue
        seen.add(key)
        unique.append(act)
    return unique


def _filter_activations_by_bin_contains(
    activations_data: List[Dict],
    bin_contains_max: Optional[float],
) -> List[Dict]:
    if bin_contains_max is None:
        return activations_data
    return [
        entry
        for entry in activations_data
        if "binContains" not in entry or entry["binContains"] is None or entry["binContains"] <= bin_contains_max
    ]


def get_activations_data(
    model_id: str,
    sae_id: str,
    feature: int,
    bin_contains_max: Optional[float] = None,
) -> List[Activation]:
    data = get_export_data(model_id, sae_id, feature)
    if not data:
        return []
    activations_data = data.get("activations", [])
    filtered = _filter_activations_by_bin_contains(activations_data, bin_contains_max)
    return json_to_activations(filtered)


def get_feature_api(model_id: str, sae_id: str, feature: int) -> Optional[Dict]:
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.warning(f"Neuronpedia API request failed for {model_id}/{sae_id}/{feature}: {exc}")
        return None


def get_act_neuronpedia(
    model_id: str,
    sae_id: str,
    feature: int,
    bin_contains_max: Optional[float] = None,
) -> List[Activation]:
    result = get_activations_data(model_id, sae_id, feature, bin_contains_max=bin_contains_max)
    if result:
        return result
    api_result = get_feature_api(model_id, sae_id, feature)
    if api_result is None:
        return []
    activations_data = api_result.get("activations", [])
    filtered = _filter_activations_by_bin_contains(activations_data, bin_contains_max)
    return json_to_activations(filtered)


def count_max_activating_tokens(feature: Feature) -> int:
    if not feature.activations:
        return 0

    positive_count = 0
    for activation in feature.activations:
        if activation.max_value_token_index is None:
            continue

        idx = activation.max_value_token_index
        values = activation.get_values()

        if idx < len(values) and values[idx] > 0:
            positive_count += 1

    return positive_count


_TRANSLUCE_ARRAY_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
_TRANSLUCE_LAYER_PKL_CACHE: Dict[int, Dict[int, List[Activation]]] = {}


def _find_transluce_file(layer_dir: Path, stem: str) -> Optional[Path]:
    full = layer_dir / f"{stem}.npy"
    if full.exists():
        return full
    candidates = sorted(
        layer_dir.glob(f"{stem}.npy*"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_transluce_arrays(layer: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if layer in _TRANSLUCE_ARRAY_CACHE:
        return _TRANSLUCE_ARRAY_CACHE[layer]

    layer_dir = TRANSLUCE_DIR / f"layer{layer}"
    acts_path = _find_transluce_file(layer_dir, "max_seq_acts")
    ids_path = _find_transluce_file(layer_dir, "max_seq_ids")
    dataset_ids_path = _find_transluce_file(layer_dir, "max_dataset_ids")

    if acts_path is None or ids_path is None:
        raise FileNotFoundError(f"Missing Transluce files for layer {layer} in {layer_dir}")

    acts = np.load(acts_path, mmap_mode="r")
    ids = np.load(ids_path, mmap_mode="r")
    dataset_ids = np.load(dataset_ids_path, mmap_mode="r") if dataset_ids_path else None

    _TRANSLUCE_ARRAY_CACHE[layer] = (acts, ids, dataset_ids)
    return acts, ids, dataset_ids


def _ids_to_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    if isinstance(token_ids, (int, np.integer)):
        token_ids = [int(token_ids)]
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)

    if tokenizer is None:
        return [str(tid) for tid in token_ids]
    if hasattr(tokenizer, "to_str_tokens"):
        return tokenizer.to_str_tokens(torch.tensor(token_ids, device="cpu"))
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_ids)
    return [str(tid) for tid in token_ids]


def _load_layer_transluce_pkl(layer: int) -> Optional[Dict[int, List[Activation]]]:
    if layer in _TRANSLUCE_LAYER_PKL_CACHE:
        return _TRANSLUCE_LAYER_PKL_CACHE[layer]

    pkl_path = TRANSLUCE_EXPORTS_DIR / TRANSLUCE_LAYER_PKL_TEMPLATE.format(layer=layer)
    if not pkl_path.exists():
        return None

    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    _TRANSLUCE_LAYER_PKL_CACHE[layer] = data
    return data


def get_act_transluce(
    layer: int,
    feature: int,
    tokenizer=None,
    max_sequences: Optional[int] = None,
) -> List[Activation]:
    pkl_layer = _load_layer_transluce_pkl(layer)
    if pkl_layer is not None and feature in pkl_layer:
        return deduplicate_activations(pkl_layer[feature])

    try:
        acts, ids, dataset_ids = _load_transluce_arrays(layer)
    except FileNotFoundError:
        logger.warning(f"No Transluce data for layer {layer}")
        return []

    if feature >= acts.shape[0]:
        logger.warning(f"Feature {feature} out of bounds for layer {layer} (max {acts.shape[0]-1})")
        return []

    total_sequences = acts.shape[1]
    seq_limit = total_sequences if max_sequences is None else min(max_sequences, total_sequences)

    activations: List[Activation] = []
    for seq_idx in range(seq_limit):
        values = acts[feature, seq_idx]
        token_ids = ids[feature, seq_idx].tolist()
        tokens = _ids_to_tokens(tokenizer, token_ids)
        max_idx = int(np.argmax(values))

        activation = Activation(
            id=str(dataset_ids[feature, seq_idx]) if dataset_ids is not None else "",
            token_values=list(zip(tokens, [float(v) for v in values])),
            max_value=float(values[max_idx]),
            min_value=float(values.min()),
            max_value_token_index=max_idx,
        )
        activations.append(activation)

    return deduplicate_activations(activations)


_GENERATION_PROMPTS_FILE = Path(__file__).resolve().parent / "generation_prompts.json"

def _load_generation_prompts() -> List[str]:
    with open(_GENERATION_PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    assert isinstance(prompts, list) and len(prompts) > 0, f"Expected non-empty list in {_GENERATION_PROMPTS_FILE}"
    return prompts

OUTPUT_METRIC_GENERATION_PROMPTS = _load_generation_prompts()


def _kl_div(p, q, eps=1e-10):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def _set_feature_act_kl_hook(act, hook, feature: int, value):
    act[:, :, feature] = value


def _sae_output_hook_name(sae) -> str:
    """Return the hook name where the SAE wrapper is installed (output hook)."""
    return getattr(sae.cfg.metadata, "hook_name_out", None) or sae.cfg.metadata.hook_name


def _get_kl_div(model: HookedSAETransformer, prompts, f, value, sae=None):
    toks = model.to_tokens(prompts)
    if sae is None:
        clean_logits = model(toks)
        hooked_logits = model.run_with_hooks(
            toks,
            fwd_hooks=[(
                f"blocks.{f.layer}.mlp.hook_post",
                functools.partial(_set_feature_act_kl_hook, feature=f.feature, value=value),
            )],
        )
        hooked_probs = hooked_logits.softmax(dim=-1)
    else:
        clean_logits = model.run_with_saes(toks, saes=[sae])
        hooked_logits = model.run_with_hooks_with_saes(
            toks, saes=[sae],
            fwd_hooks=[(
                f"{_sae_output_hook_name(sae)}.hook_sae_acts_post",
                functools.partial(_set_feature_act_kl_hook, feature=f.feature, value=value),
            )],
        )
        hooked_probs = hooked_logits.softmax(dim=-1)

    clean_probs = clean_logits.softmax(dim=-1)
    clean_probs[toks == 0] = 0
    hooked_probs[toks == 0] = 0

    kl = _kl_div(clean_probs, hooked_probs)
    means = []
    for row in kl:
        nz = row[row != 0]
        if nz.numel() == 0:
            continue
        means.append(nz.mean().item())
    return float(np.mean(means)) if means else 0.0


def _get_activation_for_kl(model, prompts, f, target_kl, high_thresh=0.1, neg=False, sae=None):
    low, high = (-1000, -1) if neg else (1, 1000)
    kl = -1
    mid = 0
    while (low + 1 < high) and (kl < target_kl or kl > target_kl + high_thresh):
        mid = (low + high) // 2
        kl = _get_kl_div(model, prompts, f, mid, sae=sae)
        if (neg and kl < target_kl) or (not neg and kl > target_kl):
            high = mid
        else:
            low = mid
    return mid


def _make_gen_mask(B, T, lengths, prompt_max_len, device):
    """Build a [B, T] bool mask: True on positions to steer."""
    if T == prompt_max_len:
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            start = max(T - L, 0)
            mask[i, start:] = True
    else:
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
    return mask


def _make_masked_gen_hook(feature: int, value, sae, lengths, prompt_max_len: int):
    """Steering hook for residual-stream SAEs (encoder/decoder same space)."""
    def hook_fn(clean_act, hook):
        B, T, _ = clean_act.shape
        mask = _make_gen_mask(B, T, lengths, prompt_max_len, clean_act.device)

        if sae is None:
            feat_slice = clean_act[:, :, feature]
            feat_slice = torch.where(mask, feat_slice.new_full(feat_slice.shape, value), feat_slice)
            clean_act[:, :, feature] = feat_slice
            return clean_act

        encoded_act = sae.encode(clean_act)
        dirty_act = sae.decode(encoded_act)
        error_term = clean_act - dirty_act
        enc_feat_slice = encoded_act[:, :, feature]
        enc_feat_slice = torch.where(mask, enc_feat_slice.new_full(enc_feat_slice.shape, value), enc_feat_slice)
        encoded_act[:, :, feature] = enc_feat_slice
        return sae.decode(encoded_act) + error_term
    return hook_fn


def _make_masked_gen_hooks_transcoder(feature: int, value, sae, lengths, prompt_max_len: int):
    """Steering hooks for transcoders (encoder reads MLP input, decoder writes MLP output).

    Returns two hooks:
      input_hook  — registered at hook_name     (MLP input):  captures h_mid for encoding
      output_hook — registered at hook_name_out  (MLP output): applies the steered decode + error
    """
    captured = {}

    def input_hook(clean_act, hook):
        """Capture MLP input for the transcoder encoder. Pass through unchanged."""
        captured["mlp_input"] = clean_act
        return clean_act

    def output_hook(clean_act, hook):
        """Modify MLP output: encode(captured MLP input) → set feature → decode.
        error_term = actual_MLP_output - transcoder_reconstruction  (both in MLP-output space).
        """
        B, T, _ = clean_act.shape
        mask = _make_gen_mask(B, T, lengths, prompt_max_len, clean_act.device)

        mlp_input = captured.get("mlp_input")
        if mlp_input is None:
            return clean_act

        encoded_act = sae.encode(mlp_input)
        dirty_act = sae.decode(encoded_act)
        # Both clean_act and dirty_act are in MLP-output space — correct subtraction
        error_term = clean_act - dirty_act

        enc_feat_slice = encoded_act[:, :, feature]
        enc_feat_slice = torch.where(mask, enc_feat_slice.new_full(enc_feat_slice.shape, value), enc_feat_slice)
        encoded_act[:, :, feature] = enc_feat_slice
        return sae.decode(encoded_act) + error_term

    return input_hook, output_hook


def _hooked_gen(
    prompt_tokens: torch.Tensor, model: HookedSAETransformer, f,
    n: int = 25, value: Optional[float] = 10.0,
    temperature: float = 1.0, sae=None, return_gen_probs: bool = True,
):
    model.reset_hooks()
    pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0
    lengths = (prompt_tokens != pad_id).sum(dim=1).tolist()
    prompt_len = prompt_tokens.shape[1]

    if value is not None:
        is_transcoder = (sae is not None and hasattr(sae.cfg, "architecture")
                         and "transcoder" in sae.cfg.architecture())
        if is_transcoder:
            input_hook, output_hook = _make_masked_gen_hooks_transcoder(
                f.feature, value, sae, lengths, prompt_len,
            )
            model.add_hook(sae.cfg.metadata.hook_name, input_hook)
            model.add_hook(_sae_output_hook_name(sae), output_hook)
        else:
            block = (f"blocks.{f.layer}.mlp.hook_pre" if sae is None
                     else sae.cfg.metadata.hook_name)
            hook_fn = _make_masked_gen_hook(f.feature, value, sae, lengths, prompt_len)
            model.add_hook(block, hook_fn)

    logit_steps: List[torch.Tensor] = []

    def unembed_hook(module, inp, out):
        if return_gen_probs:
            logit_steps.append(out[:, -1, :].detach().clone())
        return out

    handle = model.unembed.register_forward_hook(unembed_hook)
    try:
        output_tokens = model.generate(
            prompt_tokens, max_new_tokens=n, stop_at_eos=True,
            temperature=temperature, do_sample=(temperature is None or temperature > 0.0),
            return_type="tokens", verbose=False,
        )
    finally:
        model.reset_hooks()
        handle.remove()

    completions = [
        x[len(model.to_string(prompt_tokens[i])):]
        for i, x in enumerate(model.to_string(output_tokens))
    ]

    gen_payload: Optional[Dict] = None
    if return_gen_probs and logit_steps:
        steered_step_logits = torch.stack(logit_steps, dim=0).transpose(0, 1)
        steered_step_probs = steered_step_logits.softmax(dim=-1)
        B, total_len = output_tokens.shape
        T_gen = min(steered_step_logits.shape[1], total_len - prompt_len)
        gen_tokens = output_tokens[:, prompt_len: prompt_len + T_gen]
        with torch.no_grad():
            if sae is None:
                clean_logits_full = model(output_tokens, return_type="logits")
            else:
                clean_logits_full = model.run_with_saes(output_tokens, saes=[sae])
            clean_step_probs = clean_logits_full[:, prompt_len - 1: prompt_len - 1 + T_gen, :].softmax(dim=-1)

        steered_ranks = steered_step_probs[:, :T_gen, :].argsort(dim=-1, descending=True).argsort(dim=-1) + 1
        clean_ranks = clean_step_probs.argsort(dim=-1, descending=True).argsort(dim=-1) + 1
        steered_mean_ranks = steered_ranks.float().mean(dim=(0, 1))
        clean_mean_ranks = clean_ranks.float().mean(dim=(0, 1))

        gen_payload = {
            "gen_tokens": gen_tokens.detach().cpu(),
            "steered_gen_probs": steered_step_probs[:, :T_gen, :].detach().cpu(),
            "clean_gen_probs": clean_step_probs.detach().cpu(),
            "steered_mean_ranks": steered_mean_ranks.cpu(),
            "clean_mean_ranks": clean_mean_ranks.cpu(),
        }
    return completions, gen_payload


def _get_completions_for_kl_val(model, prompts, f, kl, neg=False, sae=None, max_new_tokens=25):
    act = _get_activation_for_kl(model, prompts, f, kl, neg=neg, sae=sae) if kl != 0 else None
    completions, gen_payload = _hooked_gen(
        model.to_tokens(prompts, padding_side="left"), model, f,
        n=max_new_tokens, value=act, temperature=0.75, sae=sae, return_gen_probs=True,
    )
    return completions, act, gen_payload


def enrich_single_feature(
    model: HookedSAETransformer, f: Feature, prompts: List[str],
    sae=None, max_new_tokens: int = 20, kl_div_values: List[float] = None,
) -> Dict:
    def extract_token_lists(completions, gen_payload) -> List[List[int]]:
        token_lists: List[List[int]] = []
        if gen_payload is not None and "gen_tokens" in gen_payload:
            for tokens in gen_payload["gen_tokens"]:
                token_lists.append([int(t) for t in tokens.tolist() if t is not None])
            return token_lists
        for continuation in completions or []:
            tok_tensor = model.to_tokens(continuation)
            if isinstance(tok_tensor, torch.Tensor) and tok_tensor.ndim == 2:
                tok_tensor = tok_tensor[0]
            token_lists.append([int(t) for t in tok_tensor.tolist()])
        return token_lists

    clean_completions, _, clean_gen_payload = _get_completions_for_kl_val(
        model, prompts, f, 0.0, sae=sae, max_new_tokens=max_new_tokens,
    )
    clean_token_lists = extract_token_lists(clean_completions, clean_gen_payload)

    steered_token_lists: Dict[str, List[List[int]]] = {}
    steered_mean_ranks: Dict[str, List[float]] = {}
    activation_values: Dict[str, float] = {}

    clean_mean_ranks_list: List[float] = []
    if clean_gen_payload is not None and "clean_mean_ranks" in clean_gen_payload:
        clean_mean_ranks_list = clean_gen_payload["clean_mean_ranks"].tolist()

    for kl_val in kl_div_values:
        kl_key = f"{'+' if kl_val >= 0 else ''}{kl_val}"
        if kl_val == 0:
            steered_token_lists[kl_key] = clean_token_lists
            steered_mean_ranks[kl_key] = clean_mean_ranks_list
            activation_values[kl_key] = 0.0
            continue
        completions, act_value, gen_payload = _get_completions_for_kl_val(
            model, prompts, f, abs(kl_val), neg=(kl_val < 0), sae=sae, max_new_tokens=max_new_tokens,
        )
        steered_token_lists[kl_key] = extract_token_lists(completions, gen_payload)
        if gen_payload is not None and "steered_mean_ranks" in gen_payload:
            steered_mean_ranks[kl_key] = gen_payload["steered_mean_ranks"].tolist()
        activation_values[kl_key] = float(act_value) if act_value is not None else 0.0

    return {
        "clean_token_lists": clean_token_lists,
        "steered_token_lists": steered_token_lists,
        "clean_mean_ranks": clean_mean_ranks_list,
        "steered_mean_ranks": steered_mean_ranks,
        "activation_values": activation_values,
        "kl_values": kl_div_values,
        "prompt_count": len(prompts),
        "max_new_tokens": max_new_tokens,
    }


_TF_MODEL_MAP = {
    "gemma-3-1b": "google/gemma-3-1b-pt",
    "gemma-3-270m": "google/gemma-3-270m",
    "gemma-3-4b": "google/gemma-3-4b-pt",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B",
}


def _resolve_tf_model_id(model_id: str) -> str:
    return _TF_MODEL_MAP.get(model_id, model_id)


def _expand_model_aliases(tf_id: str, *extra: Optional[str]) -> set:
    # qwen-scope SAEs record `Qwen/Qwen3-*-Base` even when transformer_lens only
    # registers the non-Base variant (think/chat). Accept both forms so the
    # SAEGPUCache assertion passes — the architecture is identical.
    aliases = {tf_id, tf_id.split("/")[-1], *(x for x in extra if x)}
    short = tf_id.split("/")[-1]
    if tf_id.endswith("-Base"):
        non_base = tf_id.removesuffix("-Base")
        aliases.update({non_base, non_base.split("/")[-1]})
    else:
        aliases.update({f"{tf_id}-Base", f"{short}-Base"})
    return aliases


def _load_hooked_sae_transformer(
    model_id: str,
    device: str,
    hf_weights_id: Optional[str] = None,
) -> HookedSAETransformer:
    """Delegate to ``util.load_hooked_transformer`` with this script's alias map.

    ``model_id`` is this module's local alias (e.g. ``qwen3-1.7b-base``); we
    resolve it to a HF/TL name first, then let the shared helper handle the
    fallback + ``hf_model=`` override logic.
    """
    tf_id = _resolve_tf_model_id(model_id)
    return load_hooked_transformer(
        tf_id,
        device=device,
        fold_processing=True,
        hf_weights_override=hf_weights_id,
    )


def _enrich_worker(device_idx, model_id, shard, kl_div_values, gen_max_tokens, result_dict,
                   hf_weights_id=None):
    device = f"cuda:{device_idx}"
    torch.cuda.set_device(device_idx)
    torch.set_grad_enabled(False)

    tf_id = _resolve_tf_model_id(model_id)
    logger.info(f"[GPU {device_idx}] Loading model {tf_id}")
    model = _load_hooked_sae_transformer(model_id, device, hf_weights_id=hf_weights_id)

    sae_cache = SAEGPUCache(
        device=device, model_name=model_id, max_cache_size=5, max_memory_gb=4.0,
        model_name_aliases=_expand_model_aliases(tf_id, hf_weights_id),
    )
    prompts = OUTPUT_METRIC_GENERATION_PROMPTS

    enriched = 0
    for feat_idx, f in tqdm(shard, desc=f"GPU {device_idx}", position=device_idx):
        sae = sae_cache.get_sae(f)
        try:
            cache = enrich_single_feature(
                model, f, prompts, sae=sae,
                max_new_tokens=gen_max_tokens, kl_div_values=kl_div_values,
            )
            result_dict[feat_idx] = cache
            enriched += 1
            if enriched % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"[GPU {device_idx}] Error enriching {f}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    sae_cache.clear_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[GPU {device_idx}] Done: {enriched}/{len(shard)} enriched")


def _manual_activation_worker(device_idx, model_id, sae_groups, corpus_tokens, corpus_strings,
                               top_k, batch_size, result_dict, hf_weights_id=None):
    """Worker for multi-GPU manual activation collection.

    Args:
        sae_groups: list of (release, sae_id, feature_indices) tuples assigned to this GPU.
        result_dict: multiprocessing Manager dict, keyed by (release, sae_id, feature_index).
    """
    device = f"cuda:{device_idx}"
    torch.cuda.set_device(device_idx)
    torch.set_grad_enabled(False)

    logger.info(f"[GPU {device_idx}] Loading model for manual activations")
    model = _load_hooked_sae_transformer(model_id, device, hf_weights_id=hf_weights_id)
    corpus_tokens_dev = corpus_tokens.to(device)

    total_filled = 0
    for release, sae_id, feature_indices in tqdm(sae_groups, desc=f"GPU {device_idx} manual acts", position=device_idx):
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            manual_acts = collect_manual_activations(
                model=model, sae=sae,
                feature_indices=feature_indices,
                corpus_tokens=corpus_tokens_dev,
                corpus_strings=corpus_strings,
                top_k=top_k, batch_size=batch_size,
            )
            for fi, acts in manual_acts.items():
                if acts:
                    result_dict[(release, sae_id, fi)] = acts
                    total_filled += 1
            del sae
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[GPU {device_idx}] Manual activation failed for SAE {sae_id}: {e}")

    del model, corpus_tokens_dev
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[GPU {device_idx}] Manual activation worker done: {total_filled} features filled")


class Model:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        with_sae: bool,
        sae_release_prefix: Optional[str] = None,
        sae_sizes: Optional[List[str]] = None,
        sae_types: Optional[List[str]] = None,
        hf_weights_id: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.with_sae = with_sae
        self.sae_release_prefix = sae_release_prefix
        self.sae_sizes = sae_sizes
        self.sae_types = sae_types
        self.hf_weights_id = hf_weights_id
        self.saes_df: Optional[pd.DataFrame] = None
        self.m: Optional[HookedSAETransformer] = None

    def load_model(self, device: Optional[str] = None) -> None:
        device = device or DEVICE
        if self.m is None:
            self.m = _load_hooked_sae_transformer(
                self.model_id, device, hf_weights_id=self.hf_weights_id,
            )

    def is_gemma(self) -> bool:
        return self.model_id.startswith("gemma") and not self.is_gemma_3()

    def is_gemma_3(self) -> bool:
        return self.model_id.startswith("gemma-3-")

    def is_gpt_v5(self) -> bool:
        return self.model_name == "gpt2-small-v5"

    def is_qwen3(self) -> bool:
        return self.model_id.startswith("qwen3-")

    def is_qwen_scope(self) -> bool:
        return (self.sae_release_prefix or "").startswith("qwen-scope-")

    def get_size_int(self, size: str) -> int:
        if self.is_qwen_scope():
            return size_to_int(size)
        if self.is_qwen3():
            if self.m is None:
                raise ValueError("Base model must be loaded before sampling SAEs.")
            return self.m.cfg.d_mlp
        return size_to_int(size)

    def get_features_pkl_name(self, suffix: str = "", seed: Optional[int] = None) -> Path:
        seed_suffix = f"-s{seed}" if seed is not None else ""
        return FEATURE_SAMPLE_DIR / f"features-sample-{self.model_name}{suffix}{seed_suffix}.pkl"

    def _build_output_path(self, out_file: Optional[str], suffix: str, seed: Optional[int]) -> Path:
        if out_file:
            base_path = Path(out_file)
            stem = base_path.stem
            if suffix:
                stem += suffix
            if seed is not None:
                stem += f"-s{seed}"
            return base_path.with_name(stem + base_path.suffix)
        return self.get_features_pkl_name(suffix, seed)

    def write_features(
        self,
        data,
        out_file: Optional[str] = None,
        suffix: str = "",
        seed: Optional[int] = None,
    ) -> None:
        file_path = self._build_output_path(out_file, suffix, seed)
        logger.info(f"Writing {len(data)} features to {file_path}")
        write_pkl_file(data, file_path)

    def get_available_sae_layers(self) -> List[int]:
        all_saes_df = pd.DataFrame.from_records(
            {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
        ).T
        all_saes_df.drop(
            columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"],
            inplace=True,
        )
        sae_ids_df = self.get_sae_ids(all_saes_df)
        if not self.is_qwen_scope():
            sae_ids_df = sae_ids_df[sae_ids_df["np_id"].notna()].copy()
        else:
            sae_ids_df = sae_ids_df.copy()
        return sorted(int(x) for x in sae_ids_df["layer"].unique())

    def get_specific_layers(self, layers: Optional[List[int]] = None) -> List[Tuple[str, int, str]]:
        if not self.sae_types or not self.sae_sizes or self.m is None:
            raise ValueError("Model is missing SAE configuration or has not loaded the base model.")
        available = self.get_available_sae_layers()
        if layers is None:
            layer_list = available
        else:
            layer_list = [l for l in layers if l in available]
            skipped = [l for l in layers if l not in available]
            if skipped:
                logger.warning(f"Layers {skipped} have no SAEs available. Available: {available}")
        specific_layers: List[Tuple[str, int, str]] = []
        for sae_type in self.sae_types:
            for size in self.sae_sizes:
                specific_layers.extend([(sae_type, layer, size) for layer in layer_list])
        return specific_layers

    def extract_type(self, saes_df_row) -> str:
        if self.is_qwen_scope():
            return (self.sae_types or ["res"])[0]
        for sae_type in self.sae_types or []:
            if sae_type in saes_df_row.release or sae_type in saes_df_row.np_id:
                return sae_type
        raise ValueError(f"No valid SAE type found in row {saes_df_row.release}")

    def extract_size(self, saes_df_row) -> str:
        if self.is_qwen_scope():
            return (self.sae_sizes or ["32k"])[0]
        for size in self.sae_sizes or []:
            if size in saes_df_row.np_id:
                return size
        raise ValueError(f"No valid SAE size found in row {saes_df_row.id}")

    def get_np_sae_id(self, layer: str, sae_type: str, size: str) -> str:
        if self.saes_df is None:
            raise ValueError("SAE dataframe not populated. Call get_saes_info_specific_layers first.")
        if self.is_qwen_scope():
            # qwen-scope SAEs have no neuronpedia ids; return the local SAE id.
            # Only used by the Neuronpedia branch, which is skipped in --manual-activations mode.
            return f"layer{layer}"
        mask = (
            (self.saes_df["layer"] == layer)
            & (self.saes_df["width"] == size)
            & (
                self.saes_df["release"].str.contains(sae_type)
                | self.saes_df["np_id"].str.contains(sae_type)
            )
        )
        np_id_series = self.saes_df[mask]["np_id"]
        if np_id_series.empty:
            raise ValueError(f"Could not find SAE id for {sae_type}/{layer}/{size}")
        sae_id = np_id_series.iloc[0]
        return sae_id.removeprefix(f"{self.model_id}/")

    def enrich_sae_ids(self, sae_ids_df: pd.DataFrame) -> pd.DataFrame:
        if self.is_gemma():
            sae_ids_df["layer"] = sae_ids_df["id"].apply(lambda x: x.split("/")[0].split("_")[1])
            sae_ids_df["width"] = sae_ids_df["id"].apply(lambda x: x.split("/")[1].split("_")[1])
        elif self.is_gemma_3():
            sae_ids_df["layer"] = sae_ids_df["id"].apply(lambda x: x.split("_")[1])
            sae_ids_df["width"] = sae_ids_df["id"].apply(lambda x: x.split("_")[3])
        elif self.is_gpt_v5():
            sae_ids_df["layer"] = sae_ids_df["id"].apply(lambda x: x.split(".")[1])
            sae_ids_df["width"] = sae_ids_df["release"].apply(lambda x: x.split("-")[-1])
        elif self.is_qwen_scope():
            sae_ids_df["layer"] = sae_ids_df["id"].apply(lambda x: x.removeprefix("layer"))
            sae_ids_df["width"] = sae_ids_df["release"].apply(
                lambda x: next(
                    (p[1:] for p in x.split("-") if p.startswith("w") and p[1:].endswith("k")),
                    "",
                )
            )
        elif self.is_qwen3():
            sae_ids_df["layer"] = sae_ids_df["id"].apply(lambda x: x.split("_")[-1])
            sae_ids_df["width"] = sae_ids_df["np_id"].apply(lambda x: x.split("-")[-1])
        return sae_ids_df

    def get_sae_ids(self, all_saes_df: pd.DataFrame) -> pd.DataFrame:
        if self.is_gemma():
            saes_map = all_saes_df[
                (all_saes_df["release"].str.contains(self.sae_release_prefix))
                & (all_saes_df["release"].str.contains("canonical"))
            ][["saes_map", "neuronpedia_id"]]
        elif self.is_gpt_v5():
            saes_map = all_saes_df[
                (all_saes_df["release"].str.contains(self.sae_release_prefix))
                & (all_saes_df["release"].str.contains("v5"))
            ][["saes_map", "neuronpedia_id"]]
        else:
            saes_map = all_saes_df[
                (all_saes_df["release"].str.contains(self.sae_release_prefix))
            ][["saes_map", "neuronpedia_id"]]

        df = pd.DataFrame(saes_map)
        sae_ids_df = pd.DataFrame(columns=["id", "release"])
        for release in df.index:
            ids = df.loc[release]["saes_map"].keys()
            np_ids = df.loc[release]["neuronpedia_id"].values()
            temp_df = pd.DataFrame({"id": ids, "np_id": np_ids, "release": release})
            sae_ids_df = pd.concat([sae_ids_df, temp_df])

        sae_ids_df = sae_ids_df[
            ~(sae_ids_df["id"].str.contains("embedding") | sae_ids_df["id"].str.contains("hook_resid_pre"))
        ]
        return self.enrich_sae_ids(sae_ids_df)

    def get_saes_info_specific_layers(self, specific_layers: List[Tuple[str, int, str]]) -> None:
        all_saes_df = pd.DataFrame.from_records(
            {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
        ).T
        all_saes_df.drop(
            columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"],
            inplace=True,
        )
        sae_ids_df = self.get_sae_ids(all_saes_df)
        if not self.is_qwen_scope():
            sae_ids_df = sae_ids_df[sae_ids_df["np_id"].notna()].copy()
        else:
            sae_ids_df = sae_ids_df.copy()
        available_layers = set(sae_ids_df["layer"].unique())
        sae_ids = []
        for layer_type, layer_num, layer_width in specific_layers:
            if str(layer_num) not in available_layers:
                continue
            if self.is_qwen_scope():
                # Release prefix already isolates a homogeneous res-SAE set; np_id is null.
                mask = (
                    (sae_ids_df["layer"] == str(layer_num))
                    & (sae_ids_df["width"] == str(layer_width))
                )
            else:
                mask = (
                    (sae_ids_df["layer"] == str(layer_num))
                    & (sae_ids_df["width"] == str(layer_width))
                    & (
                        sae_ids_df["release"].str.contains(layer_type)
                        | sae_ids_df["np_id"].str.contains(layer_type)
                    )
                )
            matched = sae_ids_df[mask]
            if matched.empty:
                continue
            key_col = "id" if self.is_qwen_scope() else "np_id"
            sae_ids.append(matched.iloc[0][key_col])
        if not sae_ids:
            raise ValueError(
                f"No SAEs found for requested layers. Available SAE layers: {sorted(available_layers, key=int)}"
            )
        dedup_col = "id" if self.is_qwen_scope() else "np_id"
        sae_single_ids_df = sae_ids_df[sae_ids_df[dedup_col].isin(sae_ids)]
        logger.info(f"Using {len(sae_ids)} SAEs at layers: {sorted(available_layers & {str(l) for _, l, _ in specific_layers}, key=int)}")
        self.saes_df = sae_single_ids_df.reset_index(drop=True)

    def get_feature_sample_sae(
        self,
        amount: int,
        num_workers: int = 1,
        base_seed: Optional[int] = None,
        threshold: int = 20,
        sae_types: Optional[List[str]] = None,
        sae_sizes: Optional[List[str]] = None,
        prefer_manual: bool = False,
    ) -> List[Feature]:
        if self.saes_df is None:
            raise ValueError("SAE dataframe not populated.")

        df = self.saes_df
        if sae_types and not self.is_qwen_scope():
            # qwen-scope: release prefix already isolates a homogeneous SAE family;
            # "res" isn't a substring of the release name and np_id is null.
            type_mask = pd.Series(False, index=df.index)
            for sae_type in sae_types:
                type_mask |= df["release"].str.contains(sae_type) | df["np_id"].str.contains(sae_type)
            df = df[type_mask]
        if sae_sizes:
            df = df[df["width"].isin(sae_sizes)]

        if df.empty:
            logger.warning(
                f"No SAE rows match filters (types={sae_types}, sizes={sae_sizes}); skipping."
            )
            return []

        jobs = []
        logger.info(
            f"Sampling {amount} SAE features per layer ({len(df)}) for {self.model_name} "
            f"(threshold={threshold}, types={sae_types or self.sae_types}, "
            f"sizes={sae_sizes or self.sae_sizes}, prefer_manual={prefer_manual})"
        )
        for idx in range(len(df)):
            row = df.iloc[idx]
            size = self.extract_size(row)
            sae_type = self.extract_type(row)
            size_int = self.get_size_int(size)
            np_sae_id = self.get_np_sae_id(row.layer, sae_type, size)
            jobs.append(
                {
                    "model_id": self.model_id,
                    "layer": row.layer,
                    "sae_type": sae_type,
                    "sae_id": row.id,
                    "sae_release": row.release,
                    "size": size,
                    "size_int": size_int,
                    "np_sae_id": np_sae_id,
                    "amount": amount,
                    "seed": None if base_seed is None else base_seed + idx,
                    "threshold": threshold,
                    "prefer_manual": prefer_manual,
                }
            )

        if num_workers > 1:
            worker_count = min(num_workers, len(jobs))
            results: List[List[Feature]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_sample_sae_features_worker, job) for job in jobs]
                for future in tqdm(as_completed(futures), total=len(futures), desc="SAE layers"):
                    results.append(future.result())
        else:
            results = [
                _sample_sae_features_worker(job)
                for job in tqdm(jobs, desc="SAE layers")
            ]

        features = [feature for chunk in results for feature in chunk]
        return features

    def get_feature_sample_no_sae(
        self,
        amount: int,
        layer_type: str = "mlp",
        num_workers: int = 1,
        base_seed: Optional[int] = None,
        layers: Optional[List[int]] = None,
    ) -> List[Feature]:
        if self.m is None:
            raise ValueError("Model not loaded.")
        layer_list = list(range(self.m.cfg.n_layers)) if layers is None else layers
        logger.info(f"Sampling {amount} non-SAE features per layer ({len(layer_list)}) for {self.model_name}")
        jobs = []
        for layer in layer_list:
            jobs.append(
                (
                    layer,
                    amount,
                    self.model_id,
                    layer_type,
                    self.m.cfg.d_mlp,
                    None if base_seed is None else base_seed + layer,
                    self.m,
                )
            )

        if num_workers > 1:
            worker_count = min(num_workers, len(jobs))
            results: List[List[Feature]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_sample_no_sae_features_worker, job) for job in jobs]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Layers"):
                    results.append(future.result())
        else:
            results = [
                _sample_no_sae_features_worker(job)
                for job in tqdm(jobs, desc="Layers")
            ]

        features = [feature for chunk in results for feature in chunk]
        return features

    def write_feature_sample(
        self,
        amount: int = 100,
        write: bool = False,
        out_file: Optional[str] = None,
        seed: Optional[int] = None,
        num_workers: int = 20,
        threshold: int = 20,
        layers: Optional[List[int]] = None,
        enrich: bool = False,
        kl_div_values: Optional[List[float]] = None,
        gen_max_tokens: int = 20,
        devices: Optional[List[int]] = None,
        manual_activations: bool = False,
        manual_batch_size: int = 8,
        manual_num_sequences: int = 4096,
    ) -> List[Feature]:
        suffix = "-n"
        if seed is not None:
            seed_everything(seed)

        corpus_tokens, corpus_strings = None, None
        if manual_activations and self.m is not None:
            logger.info("Manual activation mode enabled: skipping Neuronpedia activation lookup")
            logger.info(
                f"Loading corpus for manual activation collection "
                f"({manual_num_sequences} sequences from pile-uncopyrighted)..."
            )
            corpus_tokens, corpus_strings = load_pile_corpus(
                self.m.tokenizer,
                num_sequences=manual_num_sequences,
                max_length=128,
                dataset_name="monology/pile-uncopyrighted",
            )
            corpus_tokens = corpus_tokens.to(self.m.cfg.device)

        all_features: List[Feature] = []
        if self.with_sae:
            sae_types = self.sae_types or []
            sae_sizes = self.sae_sizes or []
            combos = [(t, s) for t in sae_types for s in sae_sizes]
            if not combos:
                raise ValueError("No SAE type/size combinations configured.")

            for combo_idx, (sae_type, sae_size) in enumerate(combos):
                combo_suffix = f"-{sae_size}-{sae_type}"

                # manual mode에서는 빈 activations feature도 일단 뽑아 와서
                # 뒤에서 manual collection으로 채운다
                sampling_threshold = 0 if manual_activations else threshold
                # Oversample 3x in manual mode to compensate for features
                # that won't meet the activation threshold
                sampling_amount = amount * 3 if manual_activations else amount

                features = self.get_feature_sample_sae(
                    amount=sampling_amount,
                    num_workers=num_workers,
                    base_seed=seed,
                    threshold=sampling_threshold,
                    sae_types=[sae_type],
                    sae_sizes=[sae_size],
                    prefer_manual=manual_activations,
                )
                if not features:
                    continue

                # manual mode: 전 feature를 manual activations로 채운다
                if manual_activations and corpus_tokens is not None:
                    from itertools import groupby

                    targets = features
                    targets.sort(key=lambda f: (f.sae_release or "", f.sae_id or ""))

                    # Pre-compute groups for tqdm and multi-GPU distribution
                    groups = []
                    for key, group in groupby(targets, key=lambda f: (f.sae_release, f.sae_id)):
                        groups.append((key, list(group)))

                    if devices is not None and len(devices) > 1:
                        # Multi-GPU manual activation collection
                        sae_groups = [
                            (release, sae_id, [f.feature for f in group_list])
                            for (release, sae_id), group_list in groups
                        ]
                        # Round-robin distribute SAE groups across GPUs
                        n_devices = len(devices)
                        shards = [[] for _ in range(n_devices)]
                        for i, sg in enumerate(sae_groups):
                            shards[i % n_devices].append(sg)

                        logger.info(
                            f"Multi-GPU manual activations: {len(sae_groups)} SAE groups "
                            f"across {n_devices} GPUs {devices}"
                        )

                        had_model = self.m is not None
                        if had_model:
                            del self.m
                            self.m = None
                            gc.collect()
                            torch.cuda.empty_cache()

                        ctx = mp.get_context("spawn")
                        manager = ctx.Manager()
                        result_dict = manager.dict()

                        processes = []
                        for rank, device_idx in enumerate(devices):
                            if not shards[rank]:
                                continue
                            p = ctx.Process(
                                target=_manual_activation_worker,
                                args=(device_idx, self.model_id, shards[rank],
                                      corpus_tokens.cpu(), corpus_strings,
                                      threshold, manual_batch_size, result_dict,
                                      self.hf_weights_id),
                            )
                            p.start()
                            processes.append(p)

                        for p in processes:
                            p.join()

                        for p in processes:
                            if p.exitcode != 0:
                                logger.warning(f"Manual activation worker (pid={p.pid}) exited with code {p.exitcode}")

                        # Apply results back to features
                        filled_total = 0
                        for (release, sae_id), group_list in groups:
                            for f in group_list:
                                key = (release, sae_id, f.feature)
                                if key in result_dict:
                                    f.activations = result_dict[key]
                                    filled_total += 1
                        logger.info(f"Multi-GPU manual activations: filled {filled_total}/{len(targets)} features")

                        if had_model:
                            self.load_model()
                            corpus_tokens = corpus_tokens.to(self.m.cfg.device)
                    else:
                        # Single-GPU manual activation collection
                        for (release, sae_id), group_list in tqdm(groups, desc="Manual activation SAEs"):
                            try:
                                sae = SAE.from_pretrained(
                                    release=release,
                                    sae_id=sae_id,
                                    device=str(self.m.cfg.device),
                                )
                                manual_acts = collect_manual_activations(
                                    model=self.m,
                                    sae=sae,
                                    feature_indices=[f.feature for f in group_list],
                                    corpus_tokens=corpus_tokens,
                                    corpus_strings=corpus_strings,
                                    top_k=threshold,
                                    batch_size=manual_batch_size,
                                )
                                filled = 0
                                for f in group_list:
                                    if f.feature in manual_acts and manual_acts[f.feature]:
                                        f.activations = manual_acts[f.feature]
                                        filled += 1
                                logger.info(
                                    f"Manual activations: filled {filled}/{len(group_list)} "
                                    f"features for SAE {sae_id}"
                                )
                            except Exception as e:
                                logger.warning(f"Manual activation collection failed for SAE {sae_id}: {e}")

                    before_count = len(features)
                    features = [f for f in features if f.activations and len(f.activations) >= threshold]
                    if before_count > len(features):
                        logger.info(
                            f"Post-manual threshold filter: {before_count} -> {len(features)} "
                            f"features (threshold={threshold})"
                        )

                    # Truncate back to `amount` per layer after oversampling
                    from collections import defaultdict
                    by_layer = defaultdict(list)
                    for f in features:
                        by_layer[f.layer].append(f)
                    features = []
                    for layer in sorted(by_layer):
                        features.extend(by_layer[layer][:amount])

                if seed is not None:
                    rng = random.Random(seed + combo_idx)
                    rng.shuffle(features)
                else:
                    random.shuffle(features)

                if enrich and kl_div_values:
                    self.enrich(features, kl_div_values, gen_max_tokens, devices=devices)

                if write:
                    self.write_features(features, out_file=out_file, suffix=combo_suffix, seed=seed)

                all_features.extend(features)

            suffix = ""
        else:
            features = self.get_feature_sample_no_sae(
                amount=amount,
                num_workers=num_workers,
                base_seed=seed,
                layers=layers,
            )
            if seed is not None:
                rng = random.Random(seed)
                rng.shuffle(features)
            else:
                random.shuffle(features)

            if enrich and kl_div_values:
                self.enrich(features, kl_div_values, gen_max_tokens, devices=devices)

            if write:
                self.write_features(features, out_file=out_file, suffix=suffix, seed=seed)
            all_features.extend(features)

        return all_features

    def enrich(
        self,
        features: List[Feature],
        kl_div_values: List[float],
        gen_max_tokens: int = 20,
        cleanup_interval: int = 50,
        force: bool = False,
        devices: Optional[List[int]] = None,
    ) -> None:
        if devices is not None and len(devices) > 1:
            self._enrich_multi_gpu(features, kl_div_values, gen_max_tokens, force, devices)
            return

        if self.m is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        torch.set_grad_enabled(False)
        tf_id = _resolve_tf_model_id(self.model_id)
        sae_cache = SAEGPUCache(
            device=DEVICE,
            model_name=self.model_id,
            max_cache_size=5,
            max_memory_gb=4.0,
            model_name_aliases=_expand_model_aliases(tf_id, self.model_name),
        )

        prompts = OUTPUT_METRIC_GENERATION_PROMPTS
        total_enriched = 0
        total_skipped = 0

        features = sort_features_by_layer(features)
        for f in tqdm(features, desc="Enriching features with steering cache"):
            existing_cache = getattr(f, "steering_cache", None)
            if existing_cache is not None and not force:
                cached_kl = set(existing_cache.get("steered_token_lists", {}).keys())
                requested_kl = {f"{'+' if kl >= 0 else ''}{kl}" for kl in kl_div_values}
                if requested_kl <= cached_kl and existing_cache.get("clean_token_lists"):
                    total_skipped += 1
                    continue

            sae = sae_cache.get_sae(f)
            try:
                cache = enrich_single_feature(
                    self.m, f, prompts,
                    sae=sae,
                    max_new_tokens=gen_max_tokens,
                    kl_div_values=kl_div_values,
                )
                f.steering_cache = cache
                total_enriched += 1

                if total_enriched % cleanup_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"Enriched {total_enriched} features; memory cleaned.")

            except Exception as e:
                logger.error(f"Error enriching feature {f}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        sae_cache.clear_cache()
        logger.info(f"Enrichment complete: {total_enriched} enriched, {total_skipped} skipped (already cached)")

    def _enrich_multi_gpu(
        self,
        features: List[Feature],
        kl_div_values: List[float],
        gen_max_tokens: int,
        force: bool,
        devices: List[int],
    ) -> None:
        to_enrich: List[Tuple[int, Feature]] = []
        for i, f in enumerate(features):
            existing = getattr(f, "steering_cache", None)
            if existing is not None and not force:
                cached_kl = set(existing.get("steered_token_lists", {}).keys())
                requested_kl = {f"{'+' if kl >= 0 else ''}{kl}" for kl in kl_div_values}
                if requested_kl <= cached_kl and existing.get("clean_token_lists"):
                    continue
            to_enrich.append((i, f))

        if not to_enrich:
            logger.info("All features already enriched, skipping")
            return

        to_enrich.sort(key=lambda item: (int(item[1].layer), item[1].feature))

        n_devices = len(devices)
        logger.info(f"Multi-GPU enrichment: {len(to_enrich)} features across {n_devices} GPUs {devices}")

        had_model = self.m is not None
        if had_model:
            del self.m
            self.m = None
            gc.collect()
            torch.cuda.empty_cache()

        chunk_size = (len(to_enrich) + n_devices - 1) // n_devices
        shards: List[List[Tuple[int, Feature]]] = [
            to_enrich[i * chunk_size: (i + 1) * chunk_size] for i in range(n_devices)
        ]

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        processes = []
        for rank, device_idx in enumerate(devices):
            if not shards[rank]:
                continue
            p = ctx.Process(
                target=_enrich_worker,
                args=(device_idx, self.model_id, shards[rank],
                      kl_div_values, gen_max_tokens, result_dict,
                      self.hf_weights_id),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for p in processes:
            if p.exitcode != 0:
                logger.warning(f"Worker (pid={p.pid}) exited with code {p.exitcode}")

        applied = 0
        for feat_idx, cache in result_dict.items():
            features[feat_idx].steering_cache = cache
            applied += 1

        skipped = len(to_enrich) - applied
        logger.info(
            f"Multi-GPU enrichment complete: {applied} enriched"
            + (f", {skipped} failed" if skipped else "")
        )

        if had_model:
            self.load_model()


MODEL_CONFIGS: Dict[str, Dict] = {
    "GEMMA-3-270M": dict(
        model_id="gemma-3-270m",
        model_name="gemma-3-270m",
        with_sae=True,
        sae_release_prefix="gemma-scope-2-270m-pt-",
        sae_sizes=["16k", "65k"],
        sae_types=["res"],
    ),
    "GEMMA-3-1B": dict(
        model_id="gemma-3-1b",
        model_name="gemma-3-1b",
        with_sae=True,
        sae_release_prefix="gemma-scope-2-1b-pt-",
        sae_sizes=["16k", "65k"],
        sae_types=["res"],
    ),
    "GEMMA-3-4B": dict(
        model_id="gemma-3-4b",
        model_name="gemma-3-4b",
        with_sae=True,
        sae_release_prefix="gemma-scope-2-4b-pt-",
        sae_sizes=["16k", "65k"],
        sae_types=["res"],
    ),
    "GPT-2-SM-V5": dict(
        model_id="gpt2-small",
        model_name="gpt2-small-v5",
        with_sae=True,
        sae_release_prefix="gpt2-small-",
        sae_sizes=["32k", "128k"],
        sae_types=["resid-post"],
    ),
    "QWEN-3-0.6B": dict(
        model_id="qwen3-0.6b",
        model_name="qwen3-0.6b",
        with_sae=True,
        sae_release_prefix="mwhanna-qwen3-0.6b-transcoders",
        sae_sizes=["lowl0"],
        sae_types=["transcoder"],
    ),
    "QWEN-3-1.7B": dict(
        model_id="qwen3-1.7b",
        model_name="qwen3-1.7b",
        with_sae=True,
        sae_release_prefix="mwhanna-qwen3-1.7b-transcoders",
        sae_sizes=["lowl0"],
        sae_types=["transcoder"],
    ),
    "QWEN-3-1.7B-BASE": dict(
        model_id="qwen3-1.7b-base",
        model_name="qwen3-1.7b-base",
        with_sae=True,
        sae_release_prefix="qwen-scope-3-1.7b-base-w32k-l50",
        sae_sizes=["32k"],
        sae_types=["res"],
        # TL only registers Qwen/Qwen3-1.7B; load base weights via hf_model=.
        hf_weights_id="Qwen/Qwen3-1.7B-Base",
    ),
    "QWEN-3-4B": dict(
        model_id="qwen3-4b",
        model_name="qwen3-4b",
        with_sae=True,
        sae_release_prefix="mwhanna-qwen3-4b-transcoders",
        sae_sizes=["hp"],
        sae_types=["transcoder"],
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "gemma3-270m": "GEMMA-3-270M",
    "gemma3-1b": "GEMMA-3-1B",
    "gemma3-4b": "GEMMA-3-4B",
    "gpt2-small": "GPT-2-SM-V5",
    "qwen3-0.6b": "QWEN-3-0.6B",
    "qwen3-1.7b": "QWEN-3-1.7B",
    "qwen3-4b": "QWEN-3-4B",
    "qwen3-1.7b-base": "QWEN-3-1.7B-BASE",
}


def build_model(alias: str) -> Model:
    if alias not in MODEL_ALIASES:
        raise KeyError(f"Unknown model alias '{alias}'. Choices: {', '.join(MODEL_ALIASES)}")
    config_name = MODEL_ALIASES[alias]
    config = MODEL_CONFIGS[config_name]
    return Model(**config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample features via the logic from feature_descriptions_pipeline_org.ipynb",
    )
    parser.add_argument(
        "--model",
        default="gpt2-small",
        choices=sorted(MODEL_ALIASES.keys()),
        help="Model alias to sample from.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=100,
        help="Number of features per layer to sample.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the sampled features to disk.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional custom output path for the pickle file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for console output.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to sample (e.g., '0,1,5'). Defaults to all layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of worker threads to use while sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Minimum number of positive max activations required for a feature to be selected.",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run steering enrichment after sampling (requires GPU). Stores precomputed steering caches on features.",
    )
    parser.add_argument(
        "--kl-div-values",
        type=str,
        default="0.25,0.5",
        help="KL divergence values for steering enrichment (comma-separated, default: 0.25,0.5)",
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=20,
        help="Max new tokens to generate per prompt during enrichment (default: 20)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="CUDA device indices for multi-GPU enrichment (e.g., '0,1,2,3'). Default: single GPU.",
    )
    parser.add_argument(
        "--manual-activations",
        action="store_true",
        help="Collect activations via model forward pass and prefer manual activations over Neuronpedia. Requires GPU.",
    )
    parser.add_argument(
        "--manual-batch-size",
        type=int,
        default=32,
        help="Batch size for manual activation forward passes (default: 8). Reduce for large models to avoid OOM.",
    )
    parser.add_argument(
        "--manual-num-sequences",
        type=int,
        default=16384,
        help="Number of corpus sequences for manual activation collection (default: 4096). Uses monology/pile-uncopyrighted via streaming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    if args.seed is not None:
        seed_everything(args.seed)
    if args.num_workers < 1:
        raise ValueError("--processes must be >= 1")

    model = build_model(args.model)
    model.load_model()
    layers = validate_layers(parse_layer_list(args.layers), model.m.cfg.n_layers if model.m else 0)

    if model.with_sae:
        specific_layers = model.get_specific_layers(layers=layers)
        logger.info(f"Preparing SAE metadata for {len(specific_layers)} layer/type/size combos")
        model.get_saes_info_specific_layers(specific_layers)

    if args.threshold < 0:
        raise ValueError("--threshold must be >= 0")

    kl_div_values = None
    devices = None
    if args.enrich:
        try:
            kl_div_values = [float(x.strip()) for x in args.kl_div_values.split(",") if x.strip()]
        except ValueError as e:
            logger.error(f"Failed to parse --kl-div-values: {e}")
            kl_div_values = [0.25, 0.5]
        if args.devices:
            devices = [int(d.strip()) for d in args.devices.split(",") if d.strip()]

    features = model.write_feature_sample(
        amount=args.amount,
        write=args.write,
        out_file=args.output,
        seed=args.seed,
        num_workers=args.num_workers,
        threshold=args.threshold,
        layers=layers,
        enrich=args.enrich,
        kl_div_values=kl_div_values,
        gen_max_tokens=args.gen_max_tokens,
        devices=devices,
        manual_activations=args.manual_activations,
        manual_batch_size=args.manual_batch_size,
        manual_num_sequences=args.manual_num_sequences,
    )
    logger.info(f"Sampled {len(features)} features total")


if __name__ == "__main__":
    main()
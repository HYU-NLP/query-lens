import os
import json
import random
import pickle
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence, Optional, Union, Iterable, Set

import numpy as np
import torch
import torch.distributed as dist
import datasets
from transformer_lens.utils import tokenize_and_concatenate

from sae_lens import SAE, HookedSAETransformer


def log_info(msg: str):
    """Simple rank-0-only logger."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


# ---------------------------- Distributed helpers ----------------------------
def set_seed(seed: Optional[int], *, worker_suffix: int = 0):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Base seed value. If None, no seeding is performed.
        worker_suffix: Optional offset to add to seed for multi-worker scenarios.
    """
    if seed is None:
        return
    try:
        base_seed = int(seed)
    except (TypeError, ValueError):
        log_info(f"Ignoring invalid seed value: {seed}")
        return
    final_seed = base_seed + int(worker_suffix)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize distributed training.

    Returns (rank, world_size, local_rank).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU / CPU mode
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distribute_batch(batch_size: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Distribute batch across processes, returning (start_idx, end_idx)."""
    batch_per_process = batch_size // world_size
    remainder = batch_size % world_size

    start_idx = rank * batch_per_process + min(rank, remainder)
    end_idx = start_idx + batch_per_process + (1 if rank < remainder else 0)

    return start_idx, end_idx


def gather_results(results: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    """Gather results (list of dicts) from all processes."""
    if not dist.is_initialized():
        return results

    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    flattened_results: List[Dict[str, Any]] = []
    for process_results in all_results:
        if process_results is not None:
            flattened_results.extend(process_results)

    return flattened_results


# ---------------------------- IO utils ----------------------------
def read_pkl_file(file_name: str):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def write_pkl_file(data, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def append_result_to_jsonl(result: Dict[str, Any], file_path: str):
    """Append a single result to a JSONL file (thread-safe with file locking)."""
    import fcntl

    # Convert any numpy/torch types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() > 1 else obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)

    serializable_result = convert_to_json_serializable(result)

    # Append to JSONL file with file locking
    with open(file_path, 'a') as f:
        # Acquire exclusive lock for writing
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(serializable_result, f)
            f.write('\n')
            f.flush()  # Ensure data is written immediately
        finally:
            # Release the file lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_processed_features(jsonl_file_path: str) -> set:
    """Load already processed feature indices from JSONL file."""
    if not os.path.exists(jsonl_file_path):
        return set()

    processed_indices = set()
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    if 'feature_index' in result:
                        processed_indices.add(result['feature_index'])
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
        log_info(f"Loaded {len(processed_indices)} already processed features from {jsonl_file_path}")
    except Exception as e:
        log_info(f"Error reading existing results file: {e}")
        return set()

    return processed_indices


def load_existing_layers_from_file(output_file: str, key: str = "layer_index") -> set:
    """Read completed layer indices (or other key values) from an existing JSONL file.
    
    Args:
        output_file: Path to JSONL file to read.
        key: Key to extract from each JSON entry (default: "layer_index").
    
    Returns:
        Set of integer values found for the specified key.
    """
    existing_values: set = set()
    if not os.path.exists(output_file):
        return existing_values
    try:
        with open(output_file, "r") as existing_file:
            for line in existing_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    log_info(
                        f"Failed to parse a line in output file while collecting {key} values; skipping line."
                    )
                    continue
                value = entry.get(key)
                if isinstance(value, int):
                    existing_values.add(value)
                elif value is not None:
                    try:
                        existing_values.add(int(value))
                    except (TypeError, ValueError):
                        continue
    except Exception as e:
        log_info(f"Error reading existing results file: {e}")
    return existing_values


def determine_device(preferred_device: Optional[str]) -> str:
    """Determine which accelerator/device string to use.
    
    Args:
        preferred_device: User-specified device string (e.g., "cuda:0", "mps", "cpu").
    
    Returns:
        Device string to use, with automatic fallback if preferred_device is None.
    """
    if preferred_device is not None:
        return preferred_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_device_list(devices_arg: Optional[List[str]]) -> List[str]:
    """Normalize --devices argument into a flat list of device strings.
    
    Args:
        devices_arg: List of device strings, possibly with comma-separated values.
    
    Returns:
        Flat list of normalized device strings.
    """
    if not devices_arg:
        return []
    devices: List[str] = []
    for entry in devices_arg:
        if not entry:
            continue
        parts = entry.split(",")
        for part in parts:
            device = part.strip()
            if device:
                devices.append(device)
    return devices


# ---------------------------- SAE helpers ----------------------------
def get_cfg_model_name(cfg):
    # v5 스타일 (cfg.model_name 직접 존재)도 호환
    if hasattr(cfg, "model_name"):
        return cfg.model_name
    # v6 스타일: metadata 안에 있음
    if getattr(cfg, "metadata", None) is not None and hasattr(cfg.metadata, "model_name"):
        return cfg.metadata.model_name
    return None

class SAEGPUCache:
    """GPU-based SAE cache with memory management."""

    def __init__(self, device: str, model_name: str, max_cache_size: int = 10, max_memory_gb: float = 8.0, dtype: Optional[torch.dtype] = None, model_name_aliases: Optional[set] = None):
        self.device = device
        self.model_name = model_name
        self.model_name_aliases = {model_name} | (model_name_aliases or set())
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.dtype = dtype
        self.cache: Dict[str, SAE] = {}  # sae_key -> SAE
        self.access_order: List[str] = []  # LRU tracking
        self.memory_usage = 0

    def _estimate_sae_memory(self, sae: SAE) -> int:
        """Estimate SAE memory usage in bytes."""
        # Rough estimation based on SAE parameters
        d_sae = sae.cfg.d_sae
        d_in = sae.cfg.d_in
        # W_enc: [d_in, d_sae], W_dec: [d_sae, d_in], b_enc: [d_sae], b_dec: [d_in]
        # Assuming bfloat16 (2 bytes per parameter)
        memory_bytes = (d_in * d_sae + d_sae * d_in + d_sae + d_in) * 2
        return memory_bytes

    def _evict_lru(self):
        """Evict least recently used SAE from cache."""
        if not self.access_order:
            return

        # Remove oldest accessed SAE
        oldest_key = self.access_order.pop(0)
        if oldest_key in self.cache:
            sae = self.cache[oldest_key]
            sae_memory = self._estimate_sae_memory(sae)
            self.memory_usage -= sae_memory
            del self.cache[oldest_key]
            log_info(f"Evicted SAE {oldest_key} from GPU cache (memory freed: {sae_memory/1024**2:.1f}MB)")

    def _check_memory_limit(self, new_sae_memory: int):
        """Check if we need to evict SAEs to stay within memory limit."""
        while (self.memory_usage + new_sae_memory > self.max_memory_bytes or
               len(self.cache) >= self.max_cache_size) and self.cache:
            self._evict_lru()

    def get_sae(self, f) -> Optional[SAE]:
        """Get SAE from cache or load if not present."""
        if getattr(f, "sae_release", None) is None or getattr(f, "sae_id", None) is None:
            return None

        sae_key = f"{f.sae_release}_{f.sae_id}"

        # Check if SAE is already in cache
        if sae_key in self.cache:
            # Move to end of access order (most recently used)
            self.access_order.remove(sae_key)
            self.access_order.append(sae_key)
            return self.cache[sae_key]

        # Load new SAE
        try:
            log_info(f"Loading SAE: {f.sae_release}/{f.sae_id} to GPU")
            sae, _, _ = SAE.from_pretrained(release=f.sae_release, sae_id=f.sae_id, device=self.device)
            sae.eval()
            
            # Convert dtype if specified
            if self.dtype is not None:
                sae = sae.to(dtype=self.dtype)
            
            # Verify SAE model matches expected model
            sae_model = get_cfg_model_name(sae.cfg)
            assert sae_model in self.model_name_aliases, f"SAE model name {sae_model} doesn't match current model {self.model_name}!"

            # Verify SAE size matches feature size
            if getattr(f, "size", "") != "" and not isinstance(f.size, str):
                assert sae.cfg.d_sae == f.size, f"SAE size {sae.cfg.d_sae} doesn't match current feature {f.size}!"

            # Estimate memory usage
            sae_memory = self._estimate_sae_memory(sae)

            # Check memory limits and evict if necessary
            self._check_memory_limit(sae_memory)

            # Add to cache
            self.cache[sae_key] = sae
            self.access_order.append(sae_key)
            self.memory_usage += sae_memory

            log_info(f"Successfully loaded SAE: {f.sae_release}/{f.sae_id} to GPU cache (memory: {sae_memory/1024**2:.1f}MB)")
            return sae

        except Exception as e:
            log_info(f"Error loading SAE for feature {f}: {e}")
            return None

    def clear_cache(self):
        """Clear all cached SAEs."""
        self.cache.clear()
        self.access_order.clear()
        self.memory_usage = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_info("Cleared SAE GPU cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'memory_usage_mb': self.memory_usage / 1024**2,
            'max_memory_mb': self.max_memory_bytes / 1024**2,
            'cached_saes': list(self.cache.keys())
        }


# ---------------------------- Model loading ----------------------------

def _toggle_base_suffix(name: str) -> str:
    if name.endswith("-Base"):
        return name[: -len("-Base")]
    return f"{name}-Base"


def _tl_known_names() -> set:
    """Set of every name transformer_lens accepts for from_pretrained().

    Includes official HF ids (dict keys) AND short aliases (dict values),
    since e.g. ``gpt2-small`` lives in the values list under key ``gpt2``.
    """
    import transformer_lens.loading_from_pretrained as L
    names = set(L.OFFICIAL_MODEL_NAMES)
    aliases = getattr(L, "MODEL_ALIASES", None)
    if isinstance(aliases, dict):
        names.update(aliases.keys())
        for v in aliases.values():
            if isinstance(v, (list, tuple)):
                names.update(v)
            elif isinstance(v, str):
                names.add(v)
    return names


def _resolve_tl_names(
    model_name: str,
    hf_weights_override: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Decide which name to hand to TL as the architecture key, and which (if
    any) HF repo to load actual weights from.

    Returns ``(tl_config_name, hf_weights_name)``:
      * ``tl_config_name`` — must be a TL-registered name; controls config/converter.
      * ``hf_weights_name`` — None means "let TL download by tl_config_name".
        A string means "load AutoModelForCausalLM from this repo and inject".

    Resolution order:
      1. Explicit ``hf_weights_override``: weights come from there. ``model_name``
         must already be TL-registered (else we try to toggle ``-Base``).
      2. ``model_name`` is TL-registered: direct load, no override.
      3. Auto-fallback: try toggling ``-Base`` suffix. If the toggled form is
         TL-registered, use it as the config and load weights from ``model_name``.
      4. Otherwise raise.
    """
    tl_known = _tl_known_names()

    if hf_weights_override is not None:
        if model_name in tl_known:
            return model_name, hf_weights_override
        sibling = _toggle_base_suffix(model_name)
        if sibling in tl_known:
            log_info(
                f"[load_hooked_transformer] {model_name!r} not in transformer_lens; "
                f"using sibling config {sibling!r} with explicit weights override {hf_weights_override!r}"
            )
            return sibling, hf_weights_override
        raise ValueError(
            f"transformer_lens has no config matching {model_name!r} (sibling tried: {sibling!r}). "
            "Pass a TL-registered model_name or extend the registry."
        )

    if model_name in tl_known:
        return model_name, None

    sibling = _toggle_base_suffix(model_name)
    if sibling in tl_known:
        log_info(
            f"[load_hooked_transformer] {model_name!r} not in transformer_lens registry; "
            f"using sibling config {sibling!r} and loading HF weights from {model_name!r}"
        )
        return sibling, model_name

    raise ValueError(
        f"transformer_lens has no config for {model_name!r} and no -Base sibling found. "
        "Pass a TL-registered name or supply hf_weights_override explicitly."
    )


def load_hooked_transformer(
    model_name: str,
    *,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    fold_processing: bool = True,
    hf_weights_override: Optional[str] = None,
) -> HookedSAETransformer:
    """Load a HookedSAETransformer, transparently handling the case where the
    desired HF repo (e.g. ``Qwen/Qwen3-1.7B-Base``) is not in transformer_lens'
    registry but a same-architecture sibling (``Qwen/Qwen3-1.7B``) is.

    Behavior:
      - If ``model_name`` is TL-known, loads it directly (the standard path).
      - Else, infers a sibling TL config (currently toggling the ``-Base``
        suffix) and injects the actual weights via ``hf_model=`` so the SAE
        works against the right checkpoint.
      - ``hf_weights_override`` forces the weights source regardless of
        ``model_name``.
      - ``fold_processing=True`` uses ``from_pretrained`` (LayerNorm folding,
        unembed centering, etc.). ``False`` uses ``from_pretrained_no_processing``
        — required by analysis code that needs raw weights/hooks.
    """
    from sae_lens import HookedSAETransformer  # local re-import for clarity

    tl_config_name, hf_weights_name = _resolve_tl_names(model_name, hf_weights_override)
    loader = (
        HookedSAETransformer.from_pretrained
        if fold_processing
        else HookedSAETransformer.from_pretrained_no_processing
    )

    if hf_weights_name is not None and hf_weights_name != tl_config_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log_info(
            f"[load_hooked_transformer] TL config={tl_config_name!r}, "
            f"HF weights={hf_weights_name!r}, fold_processing={fold_processing}"
        )
        hf_dtype = dtype if dtype is not None else "auto"
        hf_model = AutoModelForCausalLM.from_pretrained(hf_weights_name, torch_dtype=hf_dtype)
        tokenizer = AutoTokenizer.from_pretrained(hf_weights_name)
        kwargs: Dict[str, Any] = dict(
            hf_model=hf_model, tokenizer=tokenizer, device=device,
        )
        if dtype is not None:
            kwargs["dtype"] = dtype
        return loader(tl_config_name, **kwargs)

    log_info(f"[load_hooked_transformer] Loading {tl_config_name!r} on {device}")
    kwargs = dict(device=device)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return loader(tl_config_name, **kwargs)


def load_sae_for_feature(f, device: str = "cpu", model_name: str = "gemma-2-2b", model_name_aliases: Optional[set] = None) -> Optional[SAE]:
    """Load SAE for a specific feature (CPU), kept for backward compatibility."""
    valid_names = {model_name} | (model_name_aliases or set())
    try:
        sae, _, _ = SAE.from_pretrained(release=f.sae_release, sae_id=f.sae_id, device=device)
        sae.eval()

        # Verify SAE model matches expected model
        sae_model = get_cfg_model_name(sae.cfg)
        assert sae_model in valid_names, f"SAE model name {sae_model} doesn't match current model {model_name}!"

        # Verify SAE size matches feature size
        if getattr(f, "size", "") != "":
            assert sae.cfg.d_sae == f.size, f"SAE size {sae.cfg.d_sae} doesn't match current feature {f.size}!"

        return sae
    except Exception as e:
        log_info(f"Error loading SAE for feature {f}: {e}")
        return None


def load_pile_sample(model: HookedSAETransformer, device: str, num_samples: int = 512, max_length: int = 128, seed: Optional[int] = None) -> torch.Tensor:
    """Load pile dataset sample for analysis.
    
    Args:
        model: The model to use for tokenization
        device: Device to place the sample on
        num_samples: Number of samples to load
        max_length: Maximum sequence length
        seed: Optional seed for deterministic sampling. If None, uses current random state.
    """
    try:
        dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")

        pile = tokenize_and_concatenate(
            dataset,
            model.tokenizer,
            streaming=False,
            max_length=max_length,
            column_name="text",
            add_bos_token=False,
            num_proc=4
        )

        pile = pile[:-1000]["tokens"]
        
        # Use generator for deterministic sampling if seed is provided
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randint(0, len(pile), (num_samples,), generator=generator)
        else:
            indices = torch.randint(0, len(pile), (num_samples,))
        
        pile_sample = pile[indices].to(device)

        log_info(f"Loaded pile sample with shape: {pile_sample.shape}")
        return pile_sample

    except Exception as e:
        log_info(f"Error loading pile dataset: {e}")
        # Fallback to dummy data
        log_info("Using dummy data as fallback...")
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            return torch.randint(0, model.cfg.d_vocab, (num_samples, max_length), device=device, generator=generator)
        else:
            return torch.randint(0, model.cfg.d_vocab, (num_samples, max_length), device=device)


# ---------------------------- Feature classes ----------------------------
@dataclass
class Activation:
    id: str
    token_values: List[Tuple[str, float]]
    max_value: float
    min_value: float
    max_value_token_index: int = None
    loss_values: List[float] = None

    def get_tokens(self) -> List[str]:
        tokens = [tv[0] for tv in self.token_values]
        return tokens

    def get_values(self) -> List[float]:
        values = [tv[1] for tv in self.token_values]
        return values

    def get_tokens_str(self) -> str:
        return "".join(self.get_tokens())

    def get_max_token_values(self, w: int = 5):
        start = max(self.max_value_token_index - w, 0)
        end = min(self.max_value_token_index + w, len(self.token_values))
        return self.token_values[start:end]

    def get_max_tokens_str(self, w: int = 5) -> str:
        start = self.max_value_token_index - w
        end = self.max_value_token_index + w
        return "".join(self.get_tokens()[start:end])

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.max_value_token_index is not None:
            return f"Max={self.token_values[self.max_value_token_index]}, Sentence={self.get_max_token_values()}"
        return f"Tokens={self.get_tokens()}"

    def __hash__(self):
        return hash(str(self))


@dataclass
class Feature:
    model_id: str
    feature: int
    layer: str
    type: str
    activations: List[Activation] = None
    sae_id: str = None
    sae_release: str = None
    size: str = ""
    steering_cache: Optional[Dict] = None

    def get_pedia_dashboard_url(self, np_sae_id) -> str:
        return f"https://www.neuronpedia.org/{self.model_id}/{np_sae_id}/{self.feature}"

    def get_size_int(self):
        if self.size != "":
            return size_to_int(self.size)
        return 0

    def get_max_activating_examples(self, k: int = 5) -> List[Activation]:
        unique_activations = list(set(self.activations))
        sorted_activations = sorted(unique_activations, key=lambda x: x.max_value, reverse=True)
        return sorted_activations[:k]

    def __str__(self):
        return f"Feature {self.type}-{self.size}/{self.layer}/{self.feature}"

    def __hash__(self):
        return hash(str(self))


def size_to_int(size: str):
    """Converts an SAE size in string format to its integer value (e.g. "16k" -> 16384)"""
    if size == "":
        return 0
    assert size[-1] == 'k', "Invalid size string format"
    num = int(size[:len(size) - 1])
    raw_value = num * (2 ** 10)
    true_power = round(math.log2(raw_value))
    return 2 ** true_power


def sort_features_by_layer(all_features_sample):
    # 레이어를 정수로 변환하여 정렬하고, 같은 레이어 내에서는 feature 번호로 정렬
    sorted_features = sorted(all_features_sample, key=lambda f: (int(f.layer), f.feature))
    layer_counts = {}
    for f in sorted_features:
        layer_counts[f.layer] = layer_counts.get(f.layer, 0) + 1
    
    # log_info(f"Layer distribution: {layer_counts}")
    return sorted_features


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype"""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float': torch.float32,
        'half': torch.float16,
    }
    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower in dtype_map:
        return dtype_map[dtype_str_lower]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")

# Common "word-start / whitespace" markers used by various tokenizers
# - SentencePiece: ▁ (U+2581)
# - GPT2/RoBERTa byte-level BPE (sometimes in dumps): Ġ (U+0120)
# - Some BPE dumps also use "Ċ" for newline (U+010A)
_SP_SPACE = "\u2581"  # ▁
_BPE_SPACE = "\u0120"  # Ġ
_BPE_NEWLINE = "\u010A"  # Ċ

def normalize_token(tok: str) -> str:
    """
    Normalize tokenizer-specific whitespace markers into actual characters.

    - ▁foo  -> " foo"
    - Ġfoo  -> " foo"
    - Ċ     -> "\n"   (common in some byte-level BPE dumps)
    - Keeps everything else unchanged.

    Notes:
    - We intentionally replace markers wherever they appear; for most tokenizers they appear at the start,
      but replacing globally is harmless and simplifies handling.
    """
    if tok is None:
        return tok
    return (
        tok.replace(_SP_SPACE, " ")
           .replace(_BPE_SPACE, " ")
           .replace(_BPE_NEWLINE, "\n")
    )

def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """Vectorized helper."""
    return [normalize_token(t) for t in tokens]

def normalize_token_set(tokens: Iterable[str]) -> Set[str]:
    """Useful for fast membership checks."""
    return set(normalize_tokens(tokens))


def rmsnorm_pre_forward_fn(
    x: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RMSNormPre forward computation (no centering, no bias).

    Returns:
      normalized: x / sqrt(mean(x^2) + eps)   (cast to out_dtype if provided)
      scale:      sqrt(mean(x^2) + eps)       (float32)
    """
    x_f = x if x.dtype in (torch.float32, torch.float64) else x.to(torch.float32)
    scale = (x_f.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    normalized = x_f / scale
    if out_dtype is not None:
        normalized = normalized.to(out_dtype)
    return normalized, scale


def layernorm_pre_forward_fn(
    x: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LayerNormPre forward computation ('center and normalise' part).
    Works for shapes:
      [batch, pos, length] or [batch, pos, head_index, length]

    Returns:
      normalized: (x - mean) / sqrt(mean((x-mean)^2) + eps)   (cast to out_dtype if provided)
      scale:      sqrt(mean((x-mean)^2) + eps)                (float32)
    """
    x_f = x if x.dtype in (torch.float32, torch.float64) else x.to(torch.float32)
    x_centered = x_f - x_f.mean(dim=-1, keepdim=True)
    scale = (x_centered.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    normalized = x_centered / scale
    if out_dtype is not None:
        normalized = normalized.to(out_dtype)
    return normalized, scale
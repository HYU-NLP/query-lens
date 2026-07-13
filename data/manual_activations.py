"""manual_activations.py

Fallback module for collecting max-activating examples for SAE features via
model forward passes, used when Neuronpedia precomputed data is unavailable.

Provides:
    load_pile_corpus          – tokenise a sample of the Pile corpus
    collect_manual_activations – run inference and gather top-k activations
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import datasets
import torch
from loguru import logger
from tqdm import tqdm
from transformer_lens.utils import tokenize_and_concatenate
from util import Activation

def load_pile_corpus(
    tokenizer,
    num_sequences: int = 4096,
    max_length: int = 128,
    dataset_name: str = "monology/pile-uncopyrighted",
    seed: int = 42,
) -> Tuple[torch.Tensor, List[List[str]]]:
    """Load and tokenise a sample of the Pile corpus.

    Args:
        tokenizer:      A HuggingFace-compatible tokenizer.
        num_sequences:  Number of fixed-length token sequences to return.
        max_length:     Token sequence length (context window size).
        dataset_name:   HuggingFace dataset identifier.
        seed:           Random seed for shuffling and sampling.

    Returns:
        corpus_tokens:  LongTensor of shape (num_sequences, max_length).
        corpus_strings: List[List[str]] of shape (num_sequences, max_length),
                        each entry is the decoded string for that token with a
                        0.0 placeholder value (for later pairing with acts).
    """
    logger.info(f"Loading corpus from '{dataset_name}' (num_sequences={num_sequences}, max_length={max_length})")

    # Stream the dataset and collect enough raw documents.
    stream = datasets.load_dataset(dataset_name, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10000).take(num_sequences * 3)

    texts = [example["text"] for example in stream]
    raw_dataset = datasets.Dataset.from_dict({"text": texts})

    # Tokenise and concatenate into fixed-length chunks.
    tokenised = tokenize_and_concatenate(
        raw_dataset,
        tokenizer,
        streaming=False,
        max_length=max_length,
        column_name="text",
        add_bos_token=False,
        num_proc=4,
    )

    total_sequences = len(tokenised)
    if total_sequences < num_sequences:
        logger.warning(
            f"Only {total_sequences} sequences available after tokenisation; "
            f"requested {num_sequences}."
        )
        num_sequences = total_sequences

    # Randomly sample num_sequences rows.
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randint(0, total_sequences, (num_sequences,), generator=generator).tolist()

    all_tokens = tokenised["tokens"]  # HuggingFace Dataset column
    corpus_tokens = torch.stack([torch.tensor(all_tokens[i]) for i in indices])  # (num_sequences, max_length)

    # Build per-token string lists with 0.0 placeholder values.
    corpus_strings: List[List[str]] = []
    for seq_idx in range(num_sequences):
        token_ids = corpus_tokens[seq_idx].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
        corpus_strings.append(token_strs)

    logger.info(f"Corpus loaded: {corpus_tokens.shape[0]} sequences x {corpus_tokens.shape[1]} tokens")
    return corpus_tokens, corpus_strings


def collect_manual_activations(
    model,
    sae,
    feature_indices: List[int],
    corpus_tokens: torch.Tensor,
    corpus_strings: List[List[str]],
    top_k: int = 20,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict[int, List[Activation]]:
    """Collect top-k max-activating corpus examples for each requested SAE feature.

    Args:
        model:           HookedSAETransformer instance.
        sae:             SAE instance whose hook point is used for residual stream
                         extraction.
        feature_indices: List of SAE feature indices to collect activations for.
        corpus_tokens:   LongTensor of shape (num_sequences, seq_len).
        corpus_strings:  Decoded token strings, shape (num_sequences, seq_len).
        top_k:           Number of top examples to keep per feature.
        batch_size:      Number of sequences processed per forward pass.
        device:          Torch device string; inferred from model if not provided.

    Returns:
        Dict mapping each feature index to a list of up to top_k Activation
        objects sorted by descending max activation value.
    """
    device = device or str(model.cfg.device)
    hook_name: str = sae.cfg.metadata.hook_name

    logger.info(
        f"Collecting manual activations for {len(feature_indices)} features "
        f"using hook '{hook_name}' on device '{device}'"
    )

    # Initialise per-feature min-heaps (keyed by max_val for top-k selection).
    heaps: Dict[int, list] = {fi: [] for fi in feature_indices}

    num_sequences = corpus_tokens.shape[0]
    n_batches = (num_sequences + batch_size - 1) // batch_size
    batches_failed = 0

    for batch_idx in tqdm(range(n_batches), desc="Manual activation batches", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_sequences)
        batch_tokens = corpus_tokens[start:end].to(device)

        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
                resid = cache[hook_name]          # (batch, seq_len, d_model)
                feat_acts = sae.encode(resid)     # (batch, seq_len, d_sae)

            for fi in feature_indices:
                acts = feat_acts[:, :, fi]        # (batch, seq_len)
                for local_idx in range(acts.shape[0]):
                    global_idx = start + local_idx
                    seq_acts = acts[local_idx]    # (seq_len,)
                    max_val = seq_acts.max().item()

                    if max_val <= 0:
                        continue

                    # Heap entry: (max_val, global_seq_idx, per_token_acts)
                    entry = (max_val, global_idx, seq_acts.cpu().tolist())

                    if len(heaps[fi]) < top_k:
                        heapq.heappush(heaps[fi], entry)
                    elif max_val > heaps[fi][0][0]:
                        heapq.heapreplace(heaps[fi], entry)

            del cache, resid, feat_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Manual activation batch {batch_idx}/{n_batches} failed: {e}")
            batches_failed += 1
            continue

    # Convert heaps to sorted Activation objects.
    results: Dict[int, List[Activation]] = {}
    for fi in feature_indices:
        sorted_entries = sorted(heaps[fi], key=lambda x: x[0], reverse=True)
        activations: List[Activation] = []
        for rank, (max_val, global_idx, per_token_acts) in enumerate(sorted_entries):
            tokens = corpus_strings[global_idx]
            token_values = list(zip(tokens, per_token_acts))
            max_token_idx = per_token_acts.index(max(per_token_acts))
            activations.append(
                Activation(
                    id=f"manual-{fi}-{rank}",
                    token_values=token_values,
                    max_value=max_val,
                    min_value=min(per_token_acts),
                    max_value_token_index=max_token_idx,
                )
            )
        results[fi] = activations

    n_with_acts = sum(1 for acts in results.values() if acts)
    logger.info(
        f"Manual activation collection: {n_with_acts}/{len(feature_indices)} features collected, "
        f"{batches_failed} batches failed"
    )
    return results

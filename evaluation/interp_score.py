from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.visualize import visualize_interp_results

try:
    # OpenAI SDK v1 style import
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You will be given a ranked list of vocabulary tokens that are most strongly
promoted in the output (or most strongly activating on the input) of a single
sparse feature in a language model. Your job is to judge how interpretable
this feature is.

Do two things:
1) Meaning summary: name the single dominant theme that the tokens support.
2) Interpretability score (0–10): rate the *topical concentration* of the
   token list — i.e., what fraction of tokens directly support one coherent
   theme.

How to read the tokens
- Tokens may be subword/BPE pieces (e.g., "ansas" from "Kansas", " Direct"
  from "Director"). Consider the canonical word or concept the piece belongs
  to before judging coherence.
- Multiple languages or scripts referring to the same concept (e.g., "food"
  in English, Chinese, Thai, Korean) count as ONE coherent theme. Do not
  penalize multilingual coverage; treat it the same as monolingual coverage
  of the same concept.
- Tokens are typically rank-ordered by promotion strength. Earlier tokens
  carry slightly more weight, but treat the full list when computing
  concentration.
- If two coherent themes coexist, pick the dominant one for Summary, but do
  not lower the score solely because more than one theme is present — judge
  on coverage of the dominant theme.

Scoring rubric (use the % of tokens directly supporting the dominant theme)
- 9–10: ≥80% of tokens directly support a single, specific theme. Off-topic
        tokens are rare and look like residual noise.
- 7–8 : 60–79% support; theme is clear; noticeable but minority off-topic
        tokens.
- 4–6 : 40–59% support; theme is identifiable but ~half the list is mixed
        or off-topic.
- 2–3 : 20–39% support; weak clustering; theme is plausible but much of the
        list is unrelated.
- 0–1 : <20% support, or no recognizable specialized theme. Tokens look
        generic, scattered, or like punctuation/format noise.

Calibration examples (do not echo these in your output)

Example A — Score 9
Tokens: " meal,  meals,  food,  foods,  cuisine,  gastronomy, 美味, อาหาร,
 음식,  comida,  Mahlzeit,  cibo,  repas,  makanan,  yemek,  meal time,
 dining,  edible,  nourishment,  appetite,  snack,  feast,  buffet,
 culinary,  diet"
Summary: "Food and meals across multiple languages."
Explanation: "Almost every token names food, a meal, or eating; the
multilingual variants describe the same concept."
→ 9

Example B — Score 6
Tokens: " manager,  Director,  supervisor,  Manager,  director,  CEO,
 chief,  Co,  ord, inate,  Sup, ervis, or,  staff,  team,  office,
 colleague,  the,  and,  with,  to,  of,  in,  on, ed"
Summary: "Senior management / leadership roles."
Explanation: "About half the tokens denote leadership/management roles; the
remainder are common function words and BPE fragments unrelated to the
theme."
→ 6

Example C — Score 1
Tokens: " شناس, 器件,  पोलिसांनी, ඥ, chaften, நபியே,  tuần,  পোলিশ,
 terang,  Reichs,  ärger, '\\u30af',  ಕಾಯ್ದ,  schaft, ',  ',  и,  '..',  '–',
 ﬁ,  '“', ',', '|', '#', '$"
Summary: "None"
Explanation: "Tokens are scattered across unrelated languages, punctuation,
and format pieces with no recognizable single theme."
→ 1

Rules
- Be strict. Topical concentration, not confidence in your guess, drives the
  score.
- If no specific theme is supported, set Summary to "None" and give a score
  in 0–1.
- Do not infer intent, sentiment, or specific entity identities beyond what
  the tokens directly support.

Output
Return RAW JSON only. No markdown fences, no commentary outside the JSON.

{
  "Score": <integer 0 to 10>,
  "Summary": "<one-sentence specific meaning/function or 'None'>",
  "Explanation": "<one short sentence: which tokens support the theme and roughly what fraction of the list does>"
}
"""


class Explainer:
    def __init__(
        self,
        remote: bool,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.remote = remote
        if remote:
            if OpenAI is None:
                raise RuntimeError("OpenAI client not available; install and configure the OpenAI SDK.")
            self.client = OpenAI()
            self.remote_model = model
        else:
            raise NotImplementedError("Local inference is not implemented in this module.")

    def __call__(self, prompts: List[Dict]):
        if not self.remote:
            raise NotImplementedError("Only remote mode is supported here.")
        completion = self.client.chat.completions.create(model=self.remote_model, messages=prompts, prompt_cache_key=f"interp_system_prompt")
        return (completion.choices[0].message.content or "", completion)

def _build_user_prompt(tokens: str) -> str:
    return (
        f"Tokens: {tokens}"
    )


def _extract_usage_from_completion(completion: Any) -> Dict[str, Optional[int]]:
    usage: Dict[str, Optional[int]] = {"input_tokens": None, "output_tokens": None, "cached_tokens": None}
    try:
        u = getattr(completion, "usage", None)
        if u is None:
            return usage
        if hasattr(u, "input_tokens") or (isinstance(u, dict) and "input_tokens" in u):
            input_tokens = getattr(u, "input_tokens", None) if not isinstance(u, dict) else u.get("input_tokens")
            output_tokens = getattr(u, "output_tokens", None) if not isinstance(u, dict) else u.get("output_tokens")
            details = getattr(u, "input_tokens_details", None) if not isinstance(u, dict) else u.get("input_tokens_details")
            cached_tokens = None
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", None) if not isinstance(details, dict) else details.get("cached_tokens")
            usage["input_tokens"] = input_tokens
            usage["output_tokens"] = output_tokens
            usage["cached_tokens"] = cached_tokens
            return usage
        prompt_tokens = getattr(u, "prompt_tokens", None) if not isinstance(u, dict) else u.get("prompt_tokens")
        completion_tokens = getattr(u, "completion_tokens", None) if not isinstance(u, dict) else u.get("completion_tokens")
        usage["input_tokens"] = prompt_tokens
        usage["output_tokens"] = completion_tokens
        usage["cached_tokens"] = None
        return usage
    except Exception:
        return usage


def _get_json_response(sys_prompt: str, explainer: Explainer, user_prompt: str) -> tuple[Dict[str, Any], Dict[str, Optional[int]]]:
    explanation_text, completion = explainer(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = _extract_usage_from_completion(completion)

    # Try to parse JSON response; if it fails, return raw content
    try:
        json_content = (
            explanation_text.strip("json").strip("`").removeprefix("json\n").removesuffix("\n")
        )
        parsed = json.loads(json_content)
        if isinstance(parsed, dict):
            return parsed, usage
    except Exception:
        pass
    return {"raw": explanation_text}, usage


def interp(tokens: str, explainer: Explainer) -> tuple[Dict[str, Any], Dict[str, Optional[int]]]:
    if not tokens:
        return {"error": "empty tokens"}, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    user_prompt = _build_user_prompt(tokens)
    return _get_json_response(SYSTEM_PROMPT, explainer, user_prompt)


def _is_valid_value(val: Any) -> bool:
    """Check if a value is valid (not None, not NaN, not empty string)."""
    if val is None:
        return False
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return False
    except Exception:
        pass
    if isinstance(val, str) and (val.lower() == "nan" or val == ""):
        return False
    return True


def _parse_token_list(value: Any) -> List[str]:
    if not _is_valid_value(value):
        return []
    if isinstance(value, list):
        return [str(tok) for tok in value if tok is not None]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(tok) for tok in parsed if tok is not None]
    except Exception:
        pass
    return []


def _parse_token_list_of_lists(value: Any) -> List[List[str]]:
    if not _is_valid_value(value):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    normalized: List[List[str]] = []
    for entry in parsed:
        if isinstance(entry, list):
            normalized.append([str(tok) for tok in entry if tok is not None])
        elif entry is not None:
            normalized.append([str(entry)])
    return normalized


def _format_tokens(tokens: Optional[List[str]]) -> str:
    if not tokens:
        return ""
    return ", ".join(tokens)

def _select_median_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the result with the median score from a list of results.
    If all scores are the same or there are two median values, randomly select one.
    """
    if not results:
        return {"error": "no results"}
    
    # Extract scores with their indices
    score_data = []
    for idx, res in enumerate(results):
        score = res.get("Score")
        if score is not None and "error" not in res:
            try:
                score_data.append((int(score), idx))
            except (ValueError, TypeError):
                pass
    
    if not score_data:
        # If no valid scores, return the first result
        return results[0]
    
    # Sort by score
    score_data.sort(key=lambda x: x[0])
    scores = [s[0] for s in score_data]
    
    # Find median
    median_score = statistics.median(scores)
    
    # Find all results with median score
    median_indices = [idx for score, idx in score_data if score == median_score]
    median_results = [results[idx] for idx in median_indices]
    
    # If all scores are the same or we have multiple median results, randomly select one
    if len(median_results) > 1 or len(set(scores)) == 1:
        return random.choice(median_results)
    
    return median_results[0]

def process_row(
    row_dict: Dict[str, Any],
    model: str = "gpt-4o-mini",
    token_source: str = "top",
    num_queries: int = 3,
    fallback_entry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single row. This function runs in a worker process.
    Creates its own Explainer instance.

    Reads unified format: top_tokens / bottom_tokens (one per JSONL entry).
    """
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    explainer = Explainer(remote=True, model=model)

    feature_index_val = row_dict.get("feature_index")
    feature_str_val = row_dict.get("feature_str")

    try:
        feature_key = str(int(feature_index_val)) if _is_valid_value(feature_index_val) else None
    except Exception:
        feature_key = str(feature_index_val) if _is_valid_value(feature_index_val) else None

    feature_str = str(feature_str_val) if _is_valid_value(feature_str_val) else ""

    if not feature_key:
        return None

    # Read unified token field
    token_field = "top_tokens" if token_source == "top" else "bottom_tokens"

    token_list = _parse_token_list(row_dict.get(token_field))
    tokens_str = _format_tokens(token_list) if token_list else ""

    # Optional fallback result
    fallback_tokens = None
    if fallback_entry and (fallback_entry.get("token_source") is None or fallback_entry.get("token_source") == token_field):
        fallback_tokens = fallback_entry.get("tokens")

    # Process tokens - query n times and select median
    query_results = []
    usage_sum = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    try:
        if fallback_tokens and isinstance(fallback_tokens, dict):
            tokens_str = str(fallback_tokens.get("tokens") or "")
            res = fallback_tokens.get("result", {"error": "missing result"})
        elif tokens_str:
            for _ in range(num_queries):
                try:
                    res, usage = interp(tokens_str, explainer=explainer)
                    query_results.append(res)
                    usage_sum["input_tokens"] += int(usage.get("input_tokens") or 0)
                    usage_sum["output_tokens"] += int(usage.get("output_tokens") or 0)
                    usage_sum["cached_tokens"] += int(usage.get("cached_tokens") or 0)
                except Exception as e:
                    query_results.append({"error": str(e)})
            res = _select_median_result(query_results)
        else:
            res = {"error": "empty tokens"}
    except Exception as e:
        res = {"error": str(e)}

    return {
        "line_obj": {
            "feature_index": feature_key,
            "feature_str": feature_str,
            "token_source": token_field,
            "tokens": {"tokens": tokens_str, "result": res},
        },
        "usage": usage_sum,
    }

def format_output_filename_with_source(base_path: str, token_source: Optional[str]) -> str:
    """Append token source information to the output filename when provided."""
    if not token_source:
        return base_path
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{token_source}{ext}"
    return os.path.join(directory, new_filename)


def load_existing_interp_results(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load existing interpretation results from a JSONL file.
    Returns mapping keyed by "<feature_key>||<token_source>" where feature_key can
    be feature_index or feature_str (as string).
    """
    if not path:
        return {}
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token_source = entry.get("token_source")
                if not token_source:
                    continue
                keys = []
                feature_index = entry.get("feature_index")
                if feature_index is not None:
                    try:
                        feature_key = str(int(feature_index))
                    except Exception:
                        feature_key = str(feature_index)
                    keys.append(f"{feature_key}||{token_source}")
                feature_str = entry.get("feature_str")
                if feature_str is not None:
                    keys.append(f"{str(feature_str)}||{token_source}")
                for k in keys:
                    existing[k] = entry
    except Exception as e:
        print(f"Warning: failed to load fallback results from {path}: {e}", file=sys.stderr)
    return existing


def _run_visualization_if_requested(out_path: str, visualize_flag: bool) -> None:
    """Generate visualization and summary when requested and results file exists."""
    if not visualize_flag:
        return
    if not os.path.exists(out_path):
        print(f"Visualization skipped: results file not found at {out_path}", file=sys.stderr)
        return

    try:
        base, _ = os.path.splitext(os.path.abspath(out_path))
        plot_path = f"{base}.png"
        summary_path = os.path.join(os.path.dirname(base), "summary.json")

        summary_stats = visualize_interp_results(
            jsonl_path=out_path,
            output_path=plot_path,
            try_positive_fallback=True,
        )

        existing = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                    if not isinstance(existing, dict):
                        existing = {}
            except Exception:
                existing = {}

        existing["interpretation"] = summary_stats
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        print(f"Saved interpretation visualization to {plot_path}")
        print(f"Saved interpretation summary to {summary_path}")
    except Exception as e:
        print(f"Visualization step failed: {e}", file=sys.stderr)
    return

def main(argv: Optional[List[str]] = None) -> int:
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    parser = argparse.ArgumentParser(description="Compute interpretation from tokens in a JSONL file using configured layer tokens.")
    parser.add_argument("jsonl", help="Path to input JSONL file")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write JSON results (default: <jsonl>_interp.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Explainer weak model name (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--token-source",
        choices=["top", "bottom"],
        default="top",
        help="Token list to sample from when building prompts (default: top).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1,
        help="Number of queries to make for each token set. The result with median score will be selected (default: 1).",
    )
    parser.add_argument(
        "--fallback-results",
        default=None,
        help="Optional path to existing interpretation results to reuse first/positive outputs when available.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate interpretation plot and summary.json entry after completion.",
    )
    args = parser.parse_args(argv)
    try:
        df = pd.read_json(args.jsonl, lines=True)
    except ValueError as exc:
        print(f"Failed to parse JSONL file: {exc}", file=sys.stderr)
        return 1

    # Validate expected columns
    base_required = ["feature_index", "feature_str"]
    missing_base = [col for col in base_required if col not in df.columns]
    if missing_base:
        print(f"Missing required columns: {', '.join(missing_base)}", file=sys.stderr)
        return 1
    token_columns = [
        "top_tokens",
        "bottom_tokens",
    ]
    if not any(col in df.columns for col in token_columns):
        print(
            "Missing required token columns: top_tokens or bottom_tokens",
            file=sys.stderr,
        )
        return 1

    # JSON logging style: append-per-line with file locking and resume support
    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        jsonl_dir = os.path.dirname(os.path.abspath(args.jsonl))
        jsonl_base = os.path.splitext(os.path.basename(args.jsonl))[0]
        out_path = os.path.join(jsonl_dir, f"{jsonl_base}_interp_{args.model}.jsonl")
    out_path = format_output_filename_with_source(out_path, args.token_source)

    # Load fallback interpretation results (if provided)
    fallback_results = load_existing_interp_results(getattr(args, "fallback_results", None))
    token_field_for_lookup = "top_tokens" if args.token_source == "top" else "bottom_tokens"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Helper: append with file lock (mirrors feature_entropy_analysis.py behavior)
    def append_result_to_jsonl(result: Dict[str, Any], file_path: str) -> None:
        import fcntl

        def to_jsonable(obj: Any):
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if isinstance(obj, list):
                return [to_jsonable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}
            return str(obj)

        serializable = to_jsonable(result)
        with open(file_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(serializable, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Helper: load processed feature indices to resume
    def load_processed_features(jsonl_file_path: str) -> set:
        if not os.path.exists(jsonl_file_path):
            return set()
        processed = set()
        try:
            with open(jsonl_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "feature_index" in obj:
                        processed.add(obj["feature_index"])  # may be int or str; we compare like-for-like below
        except Exception:
            return set()
        return processed

    processed_indices = load_processed_features(out_path)

    # Prepare rows to process (filter out already processed ones)
    rows_to_process = []
    for idx, row in df.iterrows():
        feature_index_val = row.get("feature_index")
        try:
            feature_key = str(int(feature_index_val)) if pd.notna(feature_index_val) else None
        except Exception:
            feature_key = str(feature_index_val) if feature_index_val is not None else None
        
        if not feature_key:
            continue
        
        # Resume support: skip if already processed
        if feature_key in processed_indices or (
            feature_index_val in processed_indices
        ):
            continue
        
        # Convert row to dict for worker process
        row_dict = row.to_dict()
        feature_str_val = row_dict.get("feature_str")
        feature_str = str(feature_str_val) if _is_valid_value(feature_str_val) else None

        fallback_entry = None
        lookup_keys = []
        lookup_keys.append(f"{feature_key}||{token_field_for_lookup}")
        if feature_str:
            lookup_keys.append(f"{feature_str}||{token_field_for_lookup}")
        for lk in lookup_keys:
            if lk in fallback_results:
                fallback_entry = fallback_results[lk]
                break

        rows_to_process.append({"row": row_dict, "fallback": fallback_entry})
    
    if not rows_to_process:
        print("All rows have already been processed.")
        _run_visualization_if_requested(out_path, args.visualize)
        return 0
    
    print(f"Processing {len(rows_to_process)} rows with multiprocessing...")
    
    # Determine number of workers
    num_workers = args.workers
    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    
    print(f"Using {num_workers} worker processes")
    
    # Process rows in parallel
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    # Pricing table (USD per 1M tokens)
    PRICES = {
        "gpt-5": {"input_per_m": 1.25, "cached_input_per_m": 0.125, "output_per_m": 10.0},
        "gpt-5-mini": {"input_per_m": 0.25, "cached_input_per_m": 0.025, "output_per_m": 2.0},
        "gpt-5-nano": {"input_per_m": 0.05, "cached_input_per_m": 0.005, "output_per_m": 0.40},
        "gpt-5-chat-latest": {"input_per_m": 1.25, "cached_input_per_m": 0.125, "output_per_m": 10.0},
        "gpt-5-codex": {"input_per_m": 1.25, "cached_input_per_m": 0.125, "output_per_m": 10.0},
        "gpt-5-pro": {"input_per_m": 15.0, "cached_input_per_m": None, "output_per_m": 120.0},
        "gpt-4.1": {"input_per_m": 2.0, "cached_input_per_m": 0.50, "output_per_m": 8.0},
        "gpt-4.1-mini": {"input_per_m": 0.40, "cached_input_per_m": 0.10, "output_per_m": 1.60},
        "gpt-4.1-nano": {"input_per_m": 0.10, "cached_input_per_m": 0.025, "output_per_m": 0.40},
        "gpt-4o": {"input_per_m": 2.50, "cached_input_per_m": 1.25, "output_per_m": 10.0},
        "gpt-4o-2024-05-13": {"input_per_m": 5.0, "cached_input_per_m": None, "output_per_m": 15.0},
        "gpt-4o-mini": {"input_per_m": 0.15, "cached_input_per_m": 0.075, "output_per_m": 0.60},
    }

    def _select_price_for_model(name: str):
        if name in PRICES:
            return name, PRICES[name]
        for k in PRICES.keys():
            if name.startswith(k):
                return k, PRICES[k]
        return "gpt-4o-mini", PRICES["gpt-4o-mini"]

    try:
        model_name = args.model or "gpt-4o-mini"
    except Exception:
        model_name = "gpt-4o-mini"
    model_key, price = _select_price_for_model(model_name)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {
            executor.submit(
                process_row,
                item["row"],
                model_name,
                args.token_source,
                args.num_queries,
                item.get("fallback"),
            ): item["row"]
            for item in rows_to_process
        }
        pbar = tqdm(as_completed(future_to_row), total=len(rows_to_process), desc="Interpreting")
        for future in pbar:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    usage = result.get("usage", {})
                    total_input_tokens += int(usage.get("input_tokens") or 0)
                    total_output_tokens += int(usage.get("output_tokens") or 0)
                    total_cached_tokens += int(usage.get("cached_tokens") or 0)
                    cached_rate = price.get("cached_input_per_m")
                    if cached_rate is None:
                        cached_rate = price["input_per_m"]
                    uncached_input = max(total_input_tokens - total_cached_tokens, 0)
                    cost_input_uncached = (uncached_input / 1_000_000.0) * price["input_per_m"]
                    cost_input_cached = (total_cached_tokens / 1_000_000.0) * cached_rate
                    cost_output = (total_output_tokens / 1_000_000.0) * price["output_per_m"]
                    est_cost = cost_input_uncached + cost_input_cached + cost_output
                    pbar.set_postfix(
                        tokens=f"in:{total_input_tokens} out:{total_output_tokens} cache:{total_cached_tokens}",
                        cost=f"${est_cost:.6f}",
                        model=model_key,
                    )
            except Exception as e:
                row_dict = future_to_row[future]
                feature_index_val = row_dict.get("feature_index")
                try:
                    feature_key = str(int(feature_index_val)) if pd.notna(feature_index_val) else None
                except Exception:
                    feature_key = str(feature_index_val) if feature_index_val is not None else None
                print(f"Error processing feature {feature_key}: {e}", file=sys.stderr)
    
    # Write all results to file
    for result in results:
        append_result_to_jsonl(result["line_obj"], out_path)
    
    cached_rate = price.get("cached_input_per_m")
    if cached_rate is None:
        cached_rate = price["input_per_m"]
    uncached_input = max(total_input_tokens - total_cached_tokens, 0)
    cost_input_uncached = (uncached_input / 1_000_000.0) * price["input_per_m"]
    cost_input_cached = (total_cached_tokens / 1_000_000.0) * cached_rate
    cost_output = (total_output_tokens / 1_000_000.0) * price["output_per_m"]
    est_cost = cost_input_uncached + cost_input_cached + cost_output
    print(f"Wrote JSONL results to: {out_path}")
    print(f"Processed {len(results)} rows successfully.")
    print(
        f"Token usage — input: {total_input_tokens}, output: {total_output_tokens}, cached: {total_cached_tokens}"
    )
    print(f"Estimated cost ({model_key}): ${est_cost:.6f}")

    _run_visualization_if_requested(out_path, args.visualize)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import os
import json
import random
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

import gradio as gr

# =========================
# Display labels
# =========================
PREFIX_LABELS = {
    "first": "Logit Lens",
    "positive": "Token Change",
    "final": "Query Lens",
    "last": "Query Lens",  # interpretation uses "last"
}

def display_prefix(prefix: str) -> str:
    return PREFIX_LABELS.get(prefix, prefix.replace("_", " ").title())

# =========================
# Paths
# =========================
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.environ.get(
    "QL_EXPERIMENT_DIR",
    os.path.join(_REPO_ROOT, "experiments", "features-sample-gpt2-small-v5-32k-resid-post-s17", "jvp"),
)

JVP_PATH = os.path.join(BASE_PATH, "feature_jvp_analysis.jsonl")
ACTIVATION_PATH = os.path.join(BASE_PATH, "feature_activation_results.json")
STEERING_PATH = os.path.join(BASE_PATH, "feature_steering_results.jsonl")
INTERP_PATH = os.path.join(BASE_PATH, "feature_jvp_analysis_interp_gpt-5-nano_top.jsonl")


# =========================
# Indexing utilities (fast lookup for JSONL)
# =========================
def build_jsonl_offset_index(
    path: str,
    key: str = "feature_str",
) -> Dict[str, int]:
    """
    Build a dict: feature_str -> file_offset (byte position of the line).
    So later we can f.seek(offset) and read the full line quickly.
    """
    offsets: Dict[str, int] = {}
    if not os.path.exists(path):
        return offsets
    with open(path, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get(key)
            if isinstance(k, str) and k:
                offsets[k] = pos
    return offsets

def build_layer_map_and_jvp_offsets(
    jvp_path: str,
) -> Tuple[Dict[int, List[str]], Dict[str, int], List[str]]:
    """
    From JVP JSONL, build:
      - layer_map: layer(int) -> [feature_str, ...]
      - jvp_offsets: feature_str -> file_offset
      - feature_list_in_order: feature_str list in file order (global index)
    """
    layer_map: Dict[int, List[str]] = defaultdict(list)
    offsets: Dict[str, int] = {}
    feature_list: List[str] = []

    if not os.path.exists(jvp_path):
        return dict(layer_map), offsets, feature_list

    with open(jvp_path, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            feature_str = obj.get("feature_str")
            layer = obj.get("layer")
            if not isinstance(feature_str, str) or not feature_str:
                continue
            if not isinstance(layer, int):
                # fallback: sometimes stored as str
                try:
                    layer = int(layer)
                except Exception:
                    continue

            offsets[feature_str] = pos
            layer_map[layer].append(feature_str)
            feature_list.append(feature_str)

    return dict(layer_map), offsets, feature_list

def read_jsonl_entry_by_offset(path: str, offset: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offset)
        line = f.readline()
    line = (line or "").strip()
    if not line:
        return None
    # Fix: if line is "", avoid JSON parse error and do not return {}
    if line == "":
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

# =========================
# Load activation JSON once
# =========================
def load_activation_features(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            if not raw.strip():
                return {}
            data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        feats = data.get("features", {})
        return feats if isinstance(feats, dict) else {}
    except Exception:
        return {}

# =========================
# Rendering helpers (Markdown)
# =========================
def pretty_tokens(tokens, limit=30) -> str:
    if not tokens:
        return "(none)"

    def fmt_one(t) -> str:
        # backtick 깨지는 경우 대비
        s = str(t).replace("`", "\\`")
        return f"`{s}`"

    if isinstance(tokens, list):
        out = ", ".join(fmt_one(t) for t in tokens[:limit])
        if len(tokens) > limit:
            out += " ..."
        return out

    # list가 아니면 단일 값도 backtick 처리
    return fmt_one(tokens)

def md_block(title: str, body: str) -> str:
    return f"### {title}\n\n{body}\n"

def format_score(val: Any) -> str:
    """
    Render numeric-like values with two decimal places.
    Falls back to string conversion if not a number.
    """
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    if isinstance(val, str):
        try:
            num = float(val)
            return f"{num:.2f}"
        except Exception:
            return val
    return str(val)

###########################
# ADAPTED RENDER_JVP FOR NEW FORMAT
###########################
def render_jvp(jvp: Optional[Dict[str, Any]]) -> str:
    if not jvp:
        return md_block("JVP Analysis", "_No JVP entry found._")

    lines = []
    # Top tokens for first/positive/final
    lines.append(f"- **{display_prefix('first')} top tokens**: {pretty_tokens(jvp.get('first_top_tokens'))}")
    lines.append(f"- **{display_prefix('positive')} top tokens**: {pretty_tokens(jvp.get('positive_top_tokens'))}")
    lines.append(f"- **{display_prefix('final')} top tokens**: {pretty_tokens(jvp.get('final_top_tokens'))}")

    lines.append(f"- **{display_prefix('first')} bottom tokens**: {pretty_tokens(jvp.get('first_bottom_tokens'))}")
    lines.append(f"- **{display_prefix('positive')} bottom tokens**: {pretty_tokens(jvp.get('positive_bottom_tokens'))}")
    lines.append(f"- **{display_prefix('final')} bottom tokens**: {pretty_tokens(jvp.get('final_bottom_tokens'))}")

    # Per-layer top/bottom tokens (list of list of tokens)
    layer_top_tokens = jvp.get("layer_top_tokens") or []
    layer_bottom_tokens = jvp.get("layer_bottom_tokens") or []

    # Layer index info
    layer = jvp.get("layer", None)
    feature_index = jvp.get("feature_index", None)
    feature_str = jvp.get("feature_str", None)
    rank = jvp.get("rank", None)

    # Header summary if present
    meta_items = []
    if feature_str is not None:
        meta_items.append(f"**feature_str**: `{feature_str}`")
    if feature_index is not None:
        meta_items.append(f"**feature_index**: `{feature_index}`")
    if layer is not None:
        meta_items.append(f"**layer**: `{layer}`")
    if rank is not None:
        meta_items.append(f"**rank**: `{rank}`")
    if meta_items:
        lines.insert(0, "  \n".join(meta_items))

    # Layer-wise token display
    if layer_top_tokens or layer_bottom_tokens:
        layer_token_lines = []
        lines.append("\n**Per-layer Top/Bottom Tokens**:")
        max_layers = max(len(layer_top_tokens), len(layer_bottom_tokens))
        for i in range(max_layers):
            top = layer_top_tokens[i] if i < len(layer_top_tokens) else []
            bottom = layer_bottom_tokens[i] if i < len(layer_bottom_tokens) else []
            lines.append(f"- Layer {i}:  \n"
                         f"  - Top: {pretty_tokens(top)}  \n"
                         f"  - Bottom: {pretty_tokens(bottom)}"
            )
    else:
        lines.append("- layer_top_tokens: (none)")

    return md_block("JVP Analysis", "\n".join(lines))


###########################

def render_activation(act: Optional[Dict[str, Any]]) -> str:
    if not act:
        return md_block("Activation Results", "_No activation entry found._")

    lines = []
    lines.append(f"- **{display_prefix('first')}**: `{format_score(act.get('first_score'))}`")
    lines.append(f"- **{display_prefix('positive')}**: `{format_score(act.get('positive_score'))}`")
    lines.append(f"- **{display_prefix('final')}**: `{format_score(act.get('final_score'))}`")

    max_act_strs = act.get("max_activating_strs", {})
    if isinstance(max_act_strs, dict) and max_act_strs:
        lines.append("\n**max_activating_strs**:")

        # 토큰 개수 제한 (원하면 숫자만 바꾸면 됨)
        for tok, strs in list(max_act_strs.items())[:20]:
            lines.append(f"\n- `{tok!r}`")

            if isinstance(strs, (list, tuple, set)):
                items = list(strs)
            else:
                items = [strs]

            # 각 토큰에서 보여줄 예시 개수 제한 (원하면 바꾸면 됨)
            for s in items[:5]:
                if s is None:
                    continue
                s = str(s).strip().replace("\n", "\\n")
                # 너무 길면 컷
                preview = s[:300] + ("..." if len(s) > 300 else "")
                lines.append(f"  - {preview}")
    else:
        lines.append("\n- **max_activating_strs**: (none)")

    return md_block("Activation Results", "\n".join(lines))


def render_steering(steer: Optional[Dict[str, Any]]) -> str:
    if not steer:
        return md_block("Steering Results", "_No steering entry found._")

    completions = steer.get("completions", {}) or {}

    def esc_md(s: str) -> str:
        s = str(s)
        return s.replace("`", "\\`")

    def fmt_prefix(p: str) -> str:
        return f"{esc_md(p)}"

    def fmt_text_block(text: str, max_chars: int = 600) -> str:
        t = (text or "").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + " ..."
        t = esc_md(t)
        return f"\n\n```text\n{t}\n```\n"

    def get_mapping_at_path(root: Any, path: List[str]) -> Optional[Dict[str, str]]:
        cur = root
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        if isinstance(cur, dict) and all(isinstance(v, str) for v in cur.values()):
            return cur
        return None

    def find_prefix_text(root: Any, prefix: str) -> Optional[str]:
        if isinstance(root, dict):
            if prefix in root and isinstance(root[prefix], str):
                return root[prefix]
            for v in root.values():
                out = find_prefix_text(v, prefix)
                if out is not None:
                    return out
        elif isinstance(root, list):
            for v in root:
                out = find_prefix_text(v, prefix)
                if out is not None:
                    return out
        return None

    def render_examples_section(
        title: str,
        prefixes: List[str],
        comp_root: Any,
        max_items: int = 10,
    ) -> List[str]:
        out: List[str] = []
        if not prefixes:
            return out

        shown = 0
        for p in prefixes:
            if shown >= max_items:
                break
            text = find_prefix_text(comp_root, p)
            if not isinstance(text, str) or not text.strip():
                continue
            out.append(f"- {fmt_prefix(p)}")
            out.append(fmt_text_block(text))
            shown += 1

        if shown > 0:
            out.insert(0, f"**{title}**")
        return out

    def layer_header(name: str, layer_obj: Dict[str, Any]) -> str:
        stats = layer_obj.get("stats", {}) if isinstance(layer_obj, dict) else {}
        label = display_prefix(name.split("_", 1)[0])  # first/positive/final
        return (
            f"### {label}\n"
            f"- output_score=`{format_score(stats.get('output_score'))}` \n"
        )

    def sorted_kl_keys(d: Dict[str, Any]) -> List[str]:
        def keyfn(x: str):
            s = str(x).replace("+", "")
            try:
                return float(s)
            except Exception:
                return s
        return sorted(list(d.keys()), key=keyfn)

    lines: List[str] = []

    for name in ("first_layer", "positive_layer", "final_layer"):
        layer_obj = steer.get(name, {}) or {}
        if not isinstance(layer_obj, dict):
            continue

        lines.append(layer_header(name, layer_obj))

        mp = layer_obj.get("membership_prefixes", {}) or {}
        if not isinstance(mp, dict):
            # lines.append("\n_Interesting generations: (membership_prefixes missing)_\n")
            lines.append("")
            continue

        # --------
        # Clean: 출력하지 않음
        # --------

        # --------
        # Steered only
        # --------
        steered_mp = mp.get("steered", {}) or {}
        steered_blocks: List[str] = []

        if isinstance(steered_mp, dict):
            for side in ("top_tokens", "bottom_tokens"):
                side_map = steered_mp.get(side, {}) or {}
                if not isinstance(side_map, dict):
                    continue

                for kl in sorted_kl_keys(side_map):
                    prefixes = side_map.get(kl, [])
                    if not isinstance(prefixes, list) or not prefixes:
                        continue

                    # Try likely completion paths first; fallback to global search.
                    candidates = [
                        ["steered", side, kl],
                        ["steered", kl, side],
                        [side, kl],
                        [kl, side],
                        [kl],
                    ]

                    comp_root = None
                    for path in candidates:
                        m = get_mapping_at_path(completions, path)
                        if m is not None:
                            comp_root = m
                            break
                    if comp_root is None:
                        comp_root = completions  # fallback: search everywhere by prefix

                    steered_blocks += render_examples_section(
                        f"{kl}",
                        prefixes,
                        comp_root,
                        max_items=10,
                    )

        if steered_blocks:
            # lines.append("\n**Interesting generations (overlap >= 1)**\n")
            lines.extend(steered_blocks)
            lines.append("")
        else:
            # lines.append("\n_Interesting generations: (none)_\n")
            lines.append("")

    return md_block("Steering Results", "\n".join(lines).strip())

def render_interp(interp: Optional[Dict[str, Any]]) -> str:
    if not interp:
        return md_block("Interpretation", "_No interpretation entry found._")

    lines = []
    for key in ("first_tokens", "positive_tokens", "last_tokens"):
        block = interp.get(key)
        if not isinstance(block, dict):
            continue
        result = block.get("result", {})
        if not isinstance(result, dict):
            continue
        score = result.get("Score")
        summary = result.get("Summary")
        label = display_prefix(key.split("_", 1)[0])
        lines.append(f"- **{label}**: score=`{format_score(score)}`")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"  - {summary.strip()}")

    if not lines:
        lines.append("_No token-set results found._")

    return md_block("Interpretation", "\n".join(lines))

# =========================
# Global init (index once)
# =========================
LAYER_MAP, JVP_OFFSETS, FEATURE_LIST = build_layer_map_and_jvp_offsets(JVP_PATH)
STEER_OFFSETS = build_jsonl_offset_index(STEERING_PATH)
INTERP_OFFSETS = build_jsonl_offset_index(INTERP_PATH)
ACTIVATION_FEATURES = load_activation_features(ACTIVATION_PATH)

ALL_LAYERS = sorted(LAYER_MAP.keys())
LAYER_CHOICES = ["ALL"] + [str(x) for x in ALL_LAYERS]

def get_global_index(feature_str: str) -> Optional[int]:
    # O(n) lookup, but n ~ 32k이면 충분히 가볍습니다.
    # 더 빠르게 하려면 dict(feature_str->idx)로 캐시하면 됩니다.
    try:
        return FEATURE_LIST.index(feature_str)
    except ValueError:
        return None

def fetch_all(feature_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    jvp = None
    if feature_str in JVP_OFFSETS:
        jvp = read_jsonl_entry_by_offset(JVP_PATH, JVP_OFFSETS[feature_str])

    act = ACTIVATION_FEATURES.get(feature_str)

    steer = None
    if feature_str in STEER_OFFSETS:
        steer = read_jsonl_entry_by_offset(STEERING_PATH, STEER_OFFSETS[feature_str])

    interp = None
    if feature_str in INTERP_OFFSETS:
        interp = read_jsonl_entry_by_offset(INTERP_PATH, INTERP_OFFSETS[feature_str])

    return jvp, act, steer, interp

def pick_random_feature(layer_choice: str) -> str:
    if layer_choice == "ALL":
        if not FEATURE_LIST:
            return ""
        return random.choice(FEATURE_LIST)

    try:
        layer = int(layer_choice)
    except Exception:
        return ""

    feats = LAYER_MAP.get(layer, [])
    if not feats:
        return ""
    return random.choice(feats)

def ui_random(layer_choice: str):
    feature_str = pick_random_feature(layer_choice)
    if not feature_str:
        return "(none)", "", "No feature found.", {}, {}, {}

    jvp, act, steer, interp = fetch_all(feature_str)
    idx = get_global_index(feature_str)
    header = f"## Feature\n- **feature_str**: {feature_str}\n- **global_index**: {idx}\n"
    md = header + "\n" + render_jvp(jvp) + render_interp(interp)+ render_activation(act) + render_steering(steer)
    # Always return dicts for JSON - never blank string!
    return feature_str, str(idx) if idx is not None else "", md, jvp or {}, act or {}, steer or {}

def ui_load_by_feature(feature_str: str):
    feature_str = (feature_str or "").strip()
    if not feature_str:
        return "", "", "Please input feature_str.", {}, {}, {}

    jvp, act, steer, interp = fetch_all(feature_str)
    idx = get_global_index(feature_str)
    header = f"## Feature\n- **feature_str**: {feature_str}\n- **global_index**: {idx}\n"
    md = header + "\n" + render_jvp(jvp) + render_interp(interp) + render_activation(act) + render_steering(steer)
    # Always return dicts for JSON - never blank string!
    return feature_str, str(idx) if idx is not None else "", md, jvp or {}, act or {}, steer or {}

def ui_load_by_index(idx_str: str):
    idx_str = (idx_str or "").strip()
    try:
        idx = int(idx_str)
    except Exception:
        return "", "", "Index must be an integer.", {}, {}, {}

    if idx < 0 or idx >= len(FEATURE_LIST):
        return "", "", f"Index out of range (0..{len(FEATURE_LIST)-1}).", {}, {}, {}

    feature_str = FEATURE_LIST[idx]
    return ui_load_by_feature(feature_str)

# =========================
# Gradio App
# =========================
CUSTOM_CSS = """
.gradio-container {
  font-size: 18px !important;   /* 전체 기본 글씨 크기 */
  line-height: 1.5;
}

/* Markdown 영역(출력) 더 키우고 싶으면 */
.gradio-container .prose {
  font-size: 18px !important;
}
.gradio-container .prose h1 { font-size: 32px !important; }
.gradio-container .prose h2 { font-size: 26px !important; }
.gradio-container .prose h3 { font-size: 22px !important; }

/* Textbox/Dropdown/버튼 */
.gradio-container label { font-size: 16px !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  font-size: 16px !important;
}
.gradio-container button { font-size: 16px !important; }
"""

with gr.Blocks(title="Feature Lens Viewer", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Feature Lens Viewer\n"
        "레이어 선택 후 Random 버튼을 누르면 해당 레이어에서 임의 feature를 뽑아 결과를 보여줍니다."
    )

    with gr.Row():
        layer_dd = gr.Dropdown(choices=LAYER_CHOICES, value="ALL", label="Layer")
        random_btn = gr.Button("Random", variant="primary")

    with gr.Row():
        feature_in = gr.Textbox(label="feature_str (manual)", placeholder="paste feature_str here")
        load_feature_btn = gr.Button("Load by feature_str")

    with gr.Row():
        idx_in = gr.Textbox(label="global index (manual)", placeholder="e.g., 559")
        load_idx_btn = gr.Button("Load by index")

    with gr.Row():
        feature_out = gr.Textbox(label="Selected feature_str", interactive=False)
        idx_out = gr.Textbox(label="Selected global index", interactive=False)

    md_out = gr.Markdown()

    with gr.Accordion("Raw JSON (optional)", open=False):
        jvp_json = gr.JSON(label="JVP entry")
        act_json = gr.JSON(label="Activation entry")
        steer_json = gr.JSON(label="Steering entry")

    random_btn.click(
        fn=ui_random,
        inputs=[layer_dd],
        outputs=[feature_out, idx_out, md_out, jvp_json, act_json, steer_json],
    )

    load_feature_btn.click(
        fn=ui_load_by_feature,
        inputs=[feature_in],
        outputs=[feature_out, idx_out, md_out, jvp_json, act_json, steer_json],
    )

    load_idx_btn.click(
        fn=ui_load_by_index,
        inputs=[idx_in],
        outputs=[feature_out, idx_out, md_out, jvp_json, act_json, steer_json],
    )

if __name__ == "__main__":
    # share=True 하면 외부 공유 URL도 뜹니다 (환경에 따라 막힐 수 있음).
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
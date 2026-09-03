# nexus_gui.py
import streamlit as st
import json
import os
import re
import math
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass

from upload_utils import (
    get_ollama_models,
    query_ollama,
    read_image_any_format,
    convert_to_preview,
    extract_metadata_from_tiff,
    extract_metadata_from_json_header,
    make_json_safe,
    extract_json_from_response,
    build_nexus_file,
)

# =========================================
# Constants
# =========================================
PER_KEY_CANDIDATE_LIMIT = 50
_WORD_RE = re.compile(r"[a-z0-9_]+")

# Deterministic / Hybrid thresholds
CONF_MIN = 0.78
LOGRATIO_MIN = 0.55

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
env_root = os.getenv("NEXUS_DEFINITIONS_ROOT", "").strip()
DEF_ROOT = Path(env_root) if env_root else (BASE_DIR / "nexus_definitions")
APP_DIRS = [DEF_ROOT / "applications", DEF_ROOT / "contributed_definitions"]

EXAMPLE_MAPPINGS_DIR = BASE_DIR / "example_mappings"

NON_MAPPABLE_ROOT_KEYS = {
    "50431",  # TESCAN binary metadata container; VendorKV holds parsed fields.
    "BitsPerSample",
    "Compression",
    "Orientation",
    "PlanarConfiguration",
    "RowsPerStrip",
    "StripByteCounts",
    "StripOffsets",
    "VendorKV_sources",
}

# -----------------------------------------
# (A) General fallback “stash” for unmapped metadata
# -----------------------------------------
# These are typical TIFF/EXIF-ish file/encoding keys that should NOT be forced into
# instrument fabrication/model/vendor fields (a common failure mode in purely lexical matching).
ENCODING_KEYS = {
    "bitspersample",
    "compression",
    "photometricinterpretation",
    "stripbytecounts",
    "stripoffsets",
    "orientation",
    "rowspersstrip",
    "samplesperpixel",
    "planarconfiguration",
    "software",
}

# If one of these “encoding” keys maps outside these safe zones, we stash instead.
SAFE_ZONES_FOR_ENCODING = (
    "nximage",
    "nxdata",
    "nxprocess",
    "nxnote",
    "nxcollection",
    "nxentry",  # allow start_time etc
)


def _find_best_stash_field(allowed_paths: list[str]) -> str:
    """
    Find a *schema-valid* field path (must exist in allowed_paths) where we can store
    a JSON blob of unmapped metadata, without inventing new NXDL paths.

    Preference:
      1) NXnote/(data|description|context|comment|text)
      2) NXprocess/.../(description|comment|note-like leaf)
      3) any leaf that looks like a free-text sink (description/comment/context)
    """
    if not allowed_paths:
        return ""

    aps = [p for p in allowed_paths if isinstance(p, str)]
    aps_l = [(p, p.lower()) for p in aps]

    # Helper: choose shortest path among matches (often more canonical)
    def pick(matches: list[str]) -> str:
        if not matches:
            return ""
        matches.sort(key=lambda x: (len(x), x))
        return matches[0]

    # 1) Strong preference: NXnote leaf fields
    note_leaf_priority = ("/data", "/description", "/context", "/comment", "/text", "/type", "/title")
    note_matches = []
    for p, pl in aps_l:
        if "nxnote" in pl:
            for suf in note_leaf_priority:
                if pl.endswith(suf):
                    note_matches.append(p)
                    break
    chosen = pick(note_matches)
    if chosen:
        return chosen

    # 2) NXprocess sinks
    proc_leaf_priority = ("/description", "/comment", "/context", "/type", "/title")
    proc_matches = []
    for p, pl in aps_l:
        if "nxprocess" in pl:
            for suf in proc_leaf_priority:
                if pl.endswith(suf):
                    proc_matches.append(p)
                    break
    chosen = pick(proc_matches)
    if chosen:
        return chosen

    # 3) NXcollection sinks
    coll_leaf_priority = ("/data", "/description", "/comment", "/context")
    coll_matches = []
    for p, pl in aps_l:
        if "nxcollection" in pl:
            for suf in coll_leaf_priority:
                if pl.endswith(suf):
                    coll_matches.append(p)
                    break
    chosen = pick(coll_matches)
    if chosen:
        return chosen

    # 4) Last-resort generic sinks anywhere
    generic_priority = ("/description", "/comment", "/context")
    generic_matches = []
    for p, pl in aps_l:
        for suf in generic_priority:
            if pl.endswith(suf):
                generic_matches.append(p)
                break
    return pick(generic_matches)


def _should_stash_encoding_key(meta_key: str, proposed_path: str) -> bool:
    """
    Minimal concrete suggestion:
    - For file encoding-ish keys, avoid accepting mappings into unrelated instrument fields.
    - If the proposed path is not in safe zones, stash.
    """
    k = (meta_key or "").strip().lower()
    if k not in ENCODING_KEYS:
        return False
    pl = (proposed_path or "").lower()
    if not pl:
        return True
    return not any(z in pl for z in SAFE_ZONES_FOR_ENCODING)


# =========================================
# JSON extraction helper
# =========================================
def robust_json_from_text(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    candidate = m.group(0)
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", candidate)
    candidate = re.sub(r",\s*(?=[}\]])", "", candidate)
    decoder = json.JSONDecoder(strict=False)
    return decoder.decode(candidate)


# =========================================
# Prompt builders (instrument & experiment)
# =========================================
def build_instrument_detection_prompt(pretty_metadata: str) -> str:
    return (
        """
You are a FAIRmat metadata assistant.

Your task is to identify the instrument name or model used to generate the following metadata:

"""
        + pretty_metadata
        + """
Guidelines:
- Focus on identifying a specific **instrument make and model** (e.g., "TESCAN Vega3", "FEI Nova NanoSEM 650", "Bruker Dimension Icon").
- If only the manufacturer is mentioned, return that.
- If the metadata does not contain any instrument information, return "unknown".

Strict output format:
- Respond ONLY with a **single valid JSON object**.
- ❌ DO NOT include any text, explanation, markdown, or formatting.
- ✅ ONLY return JSON.
- Keep the value under 80 characters; do NOT include firmware/software version numbers.
- Return exactly one line of JSON.

✅ Example valid responses:
{"instrument":"TESCAN Vega3"}

or if missing:
{"instrument":"unknown"}
"""
    )


def build_experiment_type_prompt(pretty_metadata: str, instrument_name: str) -> str:
    return (
        """
You are a FAIRmat metadata assistant.

Your task is to determine the most likely **experiment type** from the metadata.

Instrument:
"""
        + str(instrument_name)
        + """

Metadata:
"""
        + pretty_metadata
        + """
Valid experiment types (choose exactly one):
SEM, TEM, STEM, AFM, XPS, XAS, ARPES, Raman, Ellipsometry, Transport, STXM, Tomography, MX, APT, Other

Rules:
- Decide based on instrument model, detectors, beam/voltage/current, or modality terms.
- If electron microscopy, distinguish SEM vs TEM vs STEM.
- If spectroscopy → XPS, XAS, Raman, ARPES, Ellipsometry.
- If unclear, return "Other".
- Keep the value under 40 characters.
- Respond strictly in one-line JSON format.

Correct response examples:
{"experiment":"SEM"}
{"experiment":"Other"}

❌ Do NOT include explanations, markdown, or extra text.
✅ Return only one valid JSON object.
"""
    )


# =========================================
# Tokenisation helper
# =========================================
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tok(text: str) -> list[str]:
    return _WORD_RE.findall(_norm(text))


# =========================================
# Canonicalize NeXus paths
# =========================================
def _canonicalize_example_path(p: str) -> str:
    if not isinstance(p, str):
        return ""
    s = p.strip()
    if s.startswith("/ENTRY"):
        s = "NXentry" + s[len("/ENTRY") :]
    elif s.startswith("ENTRY"):
        s = "NXentry" + s[len("ENTRY") :]
    elif s.startswith("/NXentry"):
        s = s[1:]
    if s.startswith("/"):
        s = s[1:]
    return s


def _normalize_path(p: str) -> str:
    if not isinstance(p, str):
        return ""
    return p[1:] if p.startswith("/") else p


# =========================================
# 5-cluster routing
# =========================================
CLUSTERS = {
    "C1_IMAGE_ACQ": {
        "label": "Image & acquisition",
        "path_hints": [
            "/image_set/",
            "/image_em",
            "/NXimage",
            "/NXdata",
            "/imaging",
            "pixel",
            "axis_",
            "shape_",
            "field_of_view",
            "magnification",
            "zoom",
            "exposure",
        ],
        "key_hints": [
            "image",
            "img",
            "width",
            "height",
            "length",
            "pixel",
            "resolution",
            "dpi",
            "fov",
            "field",
            "magn",
            "zoom",
            "frame",
            "exposure",
            "scan",
        ],
    },
    "C2_BEAM_OPTICS": {
        "label": "Beam & optics",
        "path_hints": [
            "/ebeam/",
            "accelerating_voltage",
            "beam_current",
            "probe_current",
            "gun",
            "lens",
            "aperture",
            "spot",
        ],
        "key_hints": ["hv", "eht", "voltage", "kv", "beam", "probe", "current", "spot", "aperture", "lens"],
    },
    "C3_STAGE_GEOM": {
        "label": "Stage & geometry",
        "path_hints": ["/stage_lab/", "working_distance", "tilt", "rotation", "/position/"],
        "key_hints": ["wd", "working", "distance", "stage", "tilt", "rotate", "rotation", "x", "y", "z"],
    },
    "C4_DETECTOR_SIGNAL": {
        "label": "Detector & signal",
        "path_hints": ["detector_em", "detector", "signal", "bse", "se", "etd", "inlens", "mode", "channel"],
        "key_hints": ["detector", "signal", "bse", "se", "etd", "inlens", "channel", "mode"],
    },
    "C5_PROV_PROC_INSTR": {
        "label": "Provenance / processing / instrument",
        "path_hints": [
            "/process",
            "compression",
            "gamma",
            "brightness",
            "contrast",
            "software",
            "vendor",
            "model",
            "manufacturer",
            "/instrument",
            "/user",
            "start_time",
        ],
        "key_hints": [
            "vendor",
            "model",
            "instrument",
            "system",
            "software",
            "user",
            "operator",
            "date",
            "time",
            "timestamp",
            "compression",
            "gamma",
            "contrast",
            "brightness",
            "comment",
            "raw",
            "metadata",
            "tag",
        ],
    },
}


def _key_tokens_for_cluster(key: str) -> set[str]:
    k = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))
    return set(_tok(k))


def _metadata_key_aliases(key: str) -> tuple[str, ...]:
    full_key = str(key).strip().lower()
    leaf_key = full_key.rsplit(".", 1)[-1]
    return tuple(dict.fromkeys((full_key, leaf_key)))


def similarity_score_tokens(q_tokens: set[str], ex_tokens: set[str]) -> float:
    if not q_tokens or not ex_tokens:
        return 0.0
    inter = len(q_tokens & ex_tokens)
    union = len(q_tokens | ex_tokens)
    return (inter / union) + 0.15 * inter


def _example_cluster_from_path(nx_path: str) -> str:
    p = (nx_path or "").lower()
    for cid, cinfo in CLUSTERS.items():
        for h in cinfo["path_hints"]:
            if h.lower() in p:
                return cid
    return "C5_PROV_PROC_INSTR"


def _example_cluster_from_key(key: str) -> str | None:
    kt = _key_tokens_for_cluster(key)
    for cid, cinfo in CLUSTERS.items():
        if any(h in kt for h in cinfo["key_hints"]):
            return cid
    return None


def load_all_example_pairs_from_dir(example_dir: Path, definition_prefix: str = "NXem") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not example_dir.exists():
        return pairs
    for p in example_dir.glob(f"{definition_prefix}_*.json"):
        try:
            raw = json.load(open(p, "r", encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            for k, info in raw.items():
                if isinstance(info, dict) and "nx_path" in info:
                    nxp = _canonicalize_example_path(str(info["nx_path"]))
                    if nxp:
                        pairs.append((str(k), nxp))
        except Exception:
            continue
    return pairs


def build_example_index(example_pairs: list[tuple[str, str]]) -> dict:
    exact_key_to_paths: dict[str, Counter] = {}
    exact_key_to_clusters: dict[str, Counter] = {}
    cluster_examples: dict[str, list[tuple[str, str]]] = {cid: [] for cid in CLUSTERS.keys()}

    for key, nxp in example_pairs:
        kl = key.strip().lower()
        c_from_path = _example_cluster_from_path(nxp)
        c_from_key = _example_cluster_from_key(key)
        cluster = c_from_key or c_from_path

        exact_key_to_paths.setdefault(kl, Counter())[nxp] += 1
        exact_key_to_clusters.setdefault(kl, Counter())[cluster] += 1
        cluster_examples.setdefault(cluster, []).append((key, nxp))

    exact_key_map: dict[str, str] = {}
    key_to_cluster: dict[str, str] = {}
    for kl, cnt in exact_key_to_paths.items():
        exact_key_map[kl] = cnt.most_common(1)[0][0]
    for kl, cnt in exact_key_to_clusters.items():
        key_to_cluster[kl] = cnt.most_common(1)[0][0]

    return {"exact_key_map": exact_key_map, "key_to_cluster": key_to_cluster, "cluster_examples": cluster_examples}


def route_cluster(index: dict, key: str) -> tuple[str, str]:
    for alias in _metadata_key_aliases(key):
        if alias in index.get("key_to_cluster", {}):
            return index["key_to_cluster"][alias], "exact-key (examples)"
    c_hint = _example_cluster_from_key(key)
    if c_hint:
        return c_hint, "key-hints"
    return "C5_PROV_PROC_INSTR", "fallback"


def pick_similar_examples(index: dict, cluster_id: str, key: str, top_n: int = 6) -> list[tuple[str, str, float]]:
    q_tokens = _key_tokens_for_cluster(_metadata_key_aliases(key)[-1])
    candidates = index.get("cluster_examples", {}).get(cluster_id, [])
    scored: list[tuple[str, str, float]] = []
    for ex_key, ex_path in candidates:
        s = similarity_score_tokens(q_tokens, _key_tokens_for_cluster(ex_key))
        if s > 0:
            scored.append((ex_key, ex_path, s))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]


def filter_paths_by_cluster(allowed_paths: list[str], cluster_id: str) -> list[str]:
    if not allowed_paths:
        return []
    hints = [h.lower() for h in CLUSTERS[cluster_id]["path_hints"]]
    out = []
    for p in allowed_paths:
        pl = p.lower()
        if "nxentry" not in pl:
            continue
        if any(h in pl for h in hints):
            out.append(p)
    return out if len(out) >= 30 else allowed_paths


# =========================================
# LLM prompt for mapping (with examples)
# =========================================
def build_single_key_mapping_prompt(
    definition: str,
    selected_key: str,
    value_preview: str,
    allowed_subset: list[str],
    metadata_context_snippet: str,
    example_pairs: list[tuple[str, str]],
) -> str:
    allowed_text = "\n".join("- " + p for p in allowed_subset) if allowed_subset else "- (no paths)"

    if example_pairs:
        ex_lines = "\n".join([f'- "{k}" → "{p}"' for k, p in example_pairs])
        examples_block = f"""
Similar keys from curated examples (key → path):
{ex_lines}

Use these as guidance. Prefer choosing the best candidate from the allowed list.
"""
    else:
        examples_block = "No curated examples found for a similar key.\n"

    return f"""
You are a FAIRmat metadata assistant.

Goal:
Given ONE metadata key and its value, pick the best matching NeXus path from the allowed list.

Guidance:
- Prefer choosing the best candidate over returning "".
- Return "" only if none of the allowed paths are meaningfully related.

Definition:
{definition}

Metadata context (truncated):
{metadata_context_snippet}

Key:
- "{selected_key}"
- value preview: {value_preview}

{examples_block}

Allowed NeXus paths (choose one; do NOT invent):
{allowed_text}

Return ONLY one JSON object:
{{
  "mapping": {{"{selected_key}": "<one_allowed_path_or_empty>"}}
}}
"""


# =========================================
# Lexical ranking of NeXus paths (exponential)
# =========================================
def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)} if len(tokens) >= 2 else set()


def _split_path_parts(path: str) -> list[str]:
    return [p for p in path.split("/") if p]


def _build_path_doc_tokens(path: str) -> list[str]:
    parts = _split_path_parts(path)
    leaf = parts[-1].lower() if parts else ""
    toks = []
    for seg in parts:
        toks.extend(_tok(seg))
    if leaf:
        toks.extend(_tok(leaf))
    return toks


def build_path_index(allowed_paths: list[str]) -> list[dict]:
    index = []
    for p in allowed_paths:
        if not p or len(p) >= 220:
            continue
        parts = _split_path_parts(p)
        leaf = parts[-1].lower() if parts else ""
        tokens = _build_path_doc_tokens(p)
        entry = {"path": p, "leaf": leaf, "tokens": tokens, "bigrams": _bigrams(tokens), "depth": len(parts)}
        index.append(entry)
    return index


def make_query_tokens(key: str, value_preview: str, sibling_keys: list[str]) -> tuple[list[str], set[tuple[str, str]]]:
    sib_snip = " ".join(sibling_keys[:6]) if sibling_keys else ""
    q_text = f"{key} {value_preview} {sib_snip}"
    q_tokens = _tok(q_text)
    q_bigrams = _bigrams(q_tokens)
    return q_tokens, q_bigrams


def _lexical_score(query_tokens: list[str], query_bigrams: set[tuple[str, str]], entry: dict) -> float:
    doc_tokens = entry["tokens"]
    doc_bigrams = entry["bigrams"]
    leaf = entry["leaf"].lower()
    depth = entry["depth"]
    path_l = entry["path"].lower()

    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)

    uni = 0
    for t, qc in q_counts.items():
        if t in d_counts:
            uni += min(qc, d_counts[t])

    bi = len(query_bigrams & doc_bigrams)

    sub_hits = 0.0
    for t in set(query_tokens):
        if len(t) >= 3:
            if t in leaf:
                sub_hits += 1.0
            elif t in path_l:
                sub_hits += 0.5

    A, B, C, D = 1.10, 1.80, 0.75, 0.08
    linear_score = A * uni + B * bi + C * sub_hits + D * depth
    if linear_score > 20.0:
        linear_score = 20.0
    return math.exp(linear_score)


def _get_path_index_for_pool(allowed_paths: list[str], cache_key: str) -> list[dict]:
    idx_key = f"_path_index::{cache_key}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = build_path_index(allowed_paths)
    return st.session_state[idx_key]


def prioritize_paths_lexical(
    allowed_paths: list[str],
    key: str,
    value_preview: str,
    sibling_keys: list[str],
    cache_key: str,
    top_k: int = PER_KEY_CANDIDATE_LIMIT,
    show_debug: bool = False,
) -> list[str]:
    path_index = _get_path_index_for_pool(allowed_paths, cache_key)
    q_tokens, q_bigrams = make_query_tokens(key, value_preview, sibling_keys)

    scored = []
    for entry in path_index:
        s = _lexical_score(q_tokens, q_bigrams, entry)
        scored.append((entry["path"], s, entry["depth"]))

    scored.sort(key=lambda t: (t[1], t[2], -len(t[0])), reverse=True)
    top = scored[:top_k]

    if show_debug:
        st.markdown(f"**{key} — top {len(top)} lexical candidates**")
        for i, (p, sc, depth) in enumerate(top, start=1):
            st.write(f"{i:02d}. {p} — score={sc:.2f}, depth={depth}")

    return [p for p, _, _ in top]


# =========================================
# Deterministic + hybrid mapping (minimize LLM)
# =========================================
def prioritize_paths_lexical_scored(
    allowed_paths: list[str],
    key: str,
    value_preview: str,
    sibling_keys: list[str],
    cache_key: str,
    top_k: int = PER_KEY_CANDIDATE_LIMIT,
) -> list[tuple[str, float, int]]:
    path_index = _get_path_index_for_pool(allowed_paths, cache_key)
    q_tokens, q_bigrams = make_query_tokens(key, value_preview, sibling_keys)

    scored: list[tuple[str, float, int]] = []
    for entry in path_index:
        s = float(_lexical_score(q_tokens, q_bigrams, entry))
        scored.append((entry["path"], s, int(entry["depth"])))

    scored.sort(key=lambda t: (t[1], t[2], -len(t[0])), reverse=True)
    return scored[:top_k]


def compute_confidence_from_scores(s1: float, s2: float) -> tuple[float, float]:
    if s1 <= 0:
        return 0.0, 0.0
    s2 = max(s2, 1e-12)
    log_ratio = math.log(s1 / s2)
    conf = 0.50 + 0.25 * min(2.0, max(0.0, log_ratio))
    conf = float(max(0.0, min(0.99, conf)))
    return conf, float(log_ratio)


def deterministic_pick_path(
    aps_list: list[str],
    index: dict,
    definition: str,
    key: str,
    value_preview: str,
    sibling_keys: list[str],
) -> dict:
    cluster_id, cluster_reason = route_cluster(index, key)
    cluster_paths = filter_paths_by_cluster(aps_list, cluster_id)
    cache_key = f"{definition}::{cluster_id}"

    scored = prioritize_paths_lexical_scored(
        allowed_paths=cluster_paths,
        key=key,
        value_preview=value_preview,
        sibling_keys=sibling_keys,
        cache_key=cache_key,
        top_k=PER_KEY_CANDIDATE_LIMIT,
    )

    if not scored:
        return {
            "chosen": "",
            "confidence": 0.0,
            "log_ratio": 0.0,
            "cluster_id": cluster_id,
            "cluster_reason": cluster_reason,
            "candidates": [],
        }

    p1, s1, d1 = scored[0]
    s2 = scored[1][1] if len(scored) > 1 else (s1 / 1.5)
    conf, log_ratio = compute_confidence_from_scores(s1, s2)

    return {
        "chosen": p1,
        "confidence": conf,
        "log_ratio": log_ratio,
        "cluster_id": cluster_id,
        "cluster_reason": cluster_reason,
        "candidates": [{"path": p, "score": sc, "depth": dep} for (p, sc, dep) in scored[:5]],
    }


# =========================================
# Helpers for mapping
# =========================================
def flatten_metadata_keys(data, parent=""):
    keys = []
    if isinstance(data, dict):
        for k, v in data.items():
            if not parent and str(k) in NON_MAPPABLE_ROOT_KEYS:
                continue
            path = f"{parent}.{k}" if parent else k
            keys.extend(flatten_metadata_keys(v, path))
    elif isinstance(data, list):
        keys.append(parent)
    else:
        keys.append(parent)
    return list(dict.fromkeys(keys))


def get_metadata_value(data: dict, dotted_key: str):
    node = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def get_value_preview(data: dict, dotted_key: str, max_len: int = 240) -> str:
    try:
        s = repr(get_metadata_value(data, dotted_key))
    except Exception:
        s = "None"
    s = "None" if s is None else str(s)
    return (s[:max_len] + "…") if len(s) > max_len else s


# =========================================
# NXDL filesystem + XML parsing
# =========================================
EXPERIMENT_TO_NX = {
    "SEM": "NXem",
    "TEM": "NXem",
    "STEM": "NXem",
    "AFM": "NXafm",
    "XPS": "NXxps",
    "XAS": "NXxas",
    "ARPES": "NXarpes",
    "Raman": "NXraman",
    "Ellipsometry": "NXellipsometry",
    "STXM": "NXstxm",
    "Tomography": "NXtomo",
    "MX": "NXmx",
    "APT": "NXapm",
    "Transport": "NXiv_temp",
    "Other": "Other",
}

INSTRUMENT_TO_NX_DEF = {
    # "tescan amber x": "NXem",
}


def normalize_instrument_name(name: str | None) -> str:
    return (name or "").strip().lower()


def pick_nexus_definition(instrument_name: str, experiment_type: str) -> str:
    inst_norm = normalize_instrument_name(instrument_name)
    for inst_key, nx_def in INSTRUMENT_TO_NX_DEF.items():
        if inst_key in inst_norm:
            return nx_def
    return EXPERIMENT_TO_NX.get(experiment_type, "NXem")


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_definition_xml(definition: str) -> Path | None:
    stem_lower = definition.lower()
    best_path: Path | None = None
    best_score = float("-inf")

    for folder in APP_DIRS:
        if not folder.exists():
            continue
        for p in folder.rglob("*.nxdl.xml"):
            stem = p.stem.lower()
            if not stem.startswith(stem_lower):
                continue

            score = 0.0
            if stem == stem_lower:
                score += 100.0
            score -= len(stem) * 0.1

            parent_str = str(p.parent).lower()
            if "applications" in parent_str:
                score += 20.0

            if score > best_score:
                best_score = score
                best_path = p

    return best_path


def _group_label(g: ET.Element) -> str | None:
    t = g.get("type")
    if t and t.startswith("NX"):
        return t
    n = g.get("name")
    return n if n else None


def _collect_paths_from_xml_group(g: ET.Element, parent: str = "") -> set[str]:
    paths: set[str] = set()
    label = _group_label(g)
    curr = f"{parent}/{label}" if (parent and label) else (label or parent)
    for child in g:
        tag = _strip_ns(child.tag).lower()
        if tag == "group":
            paths |= _collect_paths_from_xml_group(child, curr)
        elif tag in ("field", "dataset"):
            name = child.get("name")
            if name and curr:
                paths.add(f"{curr}/{name}")
    return paths


def load_definition_schema_xml(definition: str, debug: bool = False) -> tuple[Path | None, set[str]]:
    xml_path = find_definition_xml(definition)
    if not xml_path:
        if debug:
            print(f"[NXDL] No XML file found for {definition} under {APP_DIRS}")
        return None, set()
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        if debug:
            print(f"[NXDL] Parse failed for {xml_path}: {e}")
        return xml_path, set()

    allowed: set[str] = set()
    root_tag = _strip_ns(root.tag).lower()

    if root_tag == "definition":
        for elem in root:
            tag = _strip_ns(elem.tag).lower()
            if tag == "group":
                allowed |= _collect_paths_from_xml_group(elem, "")
            elif tag in ("field", "dataset"):
                name = elem.get("name")
                if name:
                    allowed.add(name)
    else:
        for elem in root.iter():
            if _strip_ns(elem.tag).lower() == "group":
                allowed |= _collect_paths_from_xml_group(elem, "")

    return xml_path, allowed


# =========================================
# Mapping DB (instrument + definition)
# =========================================
MAPPING_DB_PATH = BASE_DIR / "mapping_db.json"


def _mapping_db_load() -> dict:
    try:
        if MAPPING_DB_PATH.exists():
            with open(MAPPING_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _mapping_db_save(db: dict) -> None:
    try:
        with open(MAPPING_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        st.warning(f"Could not save mapping DB: {e}")


def _db_key(instrument: str, definition: str) -> str:
    inst = (instrument or "unknown").strip()[:120]
    dfn = (definition or "NXem").strip()[:80]
    return f"{inst} | {dfn}"


def _mapping_db_get(instrument: str, definition: str) -> dict | None:
    db = _mapping_db_load()
    return db.get(_db_key(instrument, definition))


def _mapping_db_put(
    instrument: str,
    definition: str,
    mapping_simple: dict,
    mapping_full: dict | None = None,
) -> None:
    db = _mapping_db_load()
    db[_db_key(instrument, definition)] = {
        "mapping": mapping_simple or {},
        "mapping_full": mapping_full,
        "definition": definition,
        "instrument": instrument,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _mapping_db_save(db)


# =========================================
# App UI
# =========================================
load_dotenv()
LOGO_NFFA = Path(__file__).parent / "assets" / "nffa.png"
LOGO_LADE = Path(__file__).parent / "assets" / "area.png"

st.set_page_config(page_title="FAIR NeXus Metadata Mapper", layout="centered")

with st.sidebar:
    st.image(LOGO_NFFA, caption="NFFA-DI")
    st.image(LOGO_LADE, caption="AREA Science Park")

st.title("🔬 FAIR NeXus Metadata Mapper")
st.caption(
    "Upload a microscopy image, extract metadata, select an NXDL definition, "
    "and map metadata keys to NeXus paths."
)

models = get_ollama_models()
if not models:
    st.error(
        "⚠️ No ORFEO models are available. Check your credentials and gateway "
        "connection, then refresh."
    )
    st.stop()

selected_model = st.selectbox("ORFEO LLM model", models)
workflow = st.radio("Choose a workflow", ["Upload image + metadata", "Reference a stack (folder / S3 URL)"])

if "meta" not in st.session_state:
    st.session_state.meta = {"instrument": "", "sample_id": "", "operator": ""}

st.subheader("📄 Basic metadata")
for k in list(st.session_state.meta.keys()):
    st.session_state.meta[k] = st.text_input(k.replace("_", " ").title(), st.session_state.meta[k], key=f"meta_{k}")

meta = st.session_state.meta

# =========================================
# Workflow: Upload image + metadata
# =========================================
if workflow == "Upload image + metadata":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")
    header = st.file_uploader("(Optional) Upload separate metadata (JSON)", type="json", key="json_uploader")

    if file is None:
        st.info("👆 Upload an image to get started.")
        st.stop()

    image = read_image_any_format(file)
    st.session_state.image_array = image
    st.session_state.preview = convert_to_preview(image)
    st.image(st.session_state.preview, caption="Preview", use_container_width=True)

    instrument_metadata = {}
    filename = getattr(file, "name", "")
    source_sha256 = hashlib.sha256(file.getvalue()).hexdigest()
    if st.session_state.get("_source_sha256") != source_sha256:
        for state_key in (
            "step1_results",
            "nexus_definition",
            "nxdl_path",
            "allowed_paths",
            "_stash_field_path",
            "_definition_selector",
            "nexus_export",
        ):
            st.session_state.pop(state_key, None)
        st.session_state._source_sha256 = source_sha256

    if filename.lower().endswith((".tif", ".tiff")):
        try:
            instrument_metadata = extract_metadata_from_tiff(file)
        except Exception as e:
            st.warning(f"Couldn't read TIFF metadata: {e}")

    if not instrument_metadata and header is not None:
        try:
            instrument_metadata = extract_metadata_from_json_header(header)
        except Exception as e:
            st.warning(f"Couldn't read JSON header: {e}")

    if not instrument_metadata:
        st.error("❌ No metadata found. Please upload a TIFF with metadata or a separate JSON file.")
        st.stop()

    st.subheader("Extracted Metadata")
    st.json(instrument_metadata)

    safe_metadata = make_json_safe(instrument_metadata)
    pretty_metadata = json.dumps(safe_metadata, indent=2)
    st.session_state.safe_metadata = safe_metadata
    st.session_state.pretty_metadata = pretty_metadata

    if "step1_results" not in st.session_state:
        st.info("Click the button below to analyze metadata with the selected model.")

    if st.button("🤖 Analyze metadata (instrument & experiment)"):
        with st.spinner("⏳ Querying LLM..."):
            results = {
                "raw_instrument": None,
                "instrument_name": "unknown",
                "raw_experiment": None,
                "experiment_type": "Other",
            }

            try:
                raw_inst = query_ollama(build_instrument_detection_prompt(pretty_metadata), model=selected_model)
                results["raw_instrument"] = raw_inst
                try:
                    parsed_inst = extract_json_from_response(raw_inst)
                except Exception:
                    parsed_inst = robust_json_from_text(raw_inst)
                results["instrument_name"] = parsed_inst.get("instrument", "unknown")
            except Exception as e:
                st.error(f"❌ Failed to extract instrument: {e}")

            try:
                raw_exp = query_ollama(
                    build_experiment_type_prompt(pretty_metadata, results["instrument_name"]),
                    model=selected_model,
                )
                results["raw_experiment"] = raw_exp
                try:
                    parsed_exp = extract_json_from_response(raw_exp)
                except Exception:
                    parsed_exp = robust_json_from_text(raw_exp)
                results["experiment_type"] = parsed_exp.get("experiment", "Other")
            except Exception as e:
                st.error(f"❌ Failed to classify experiment type: {e}")

            definition = pick_nexus_definition(results["instrument_name"], results["experiment_type"])
            xml_path, allowed_paths = load_definition_schema_xml(definition)

            st.session_state.step1_results = results
            st.session_state.nexus_definition = definition
            st.session_state._definition_selector = definition
            st.session_state.nxdl_path = str(xml_path) if xml_path else None
            st.session_state.allowed_paths = sorted(list(allowed_paths)) if allowed_paths else []

            # cache stash field for this definition
            aps_list = st.session_state.get("allowed_paths", [])
            st.session_state["_stash_field_path"] = _find_best_stash_field(aps_list)

    if "step1_results" in st.session_state:
        res = st.session_state.step1_results

        available_definitions = sorted(
            {
                nx_definition
                for nx_definition in EXPERIMENT_TO_NX.values()
                if nx_definition != "Other" and find_definition_xml(nx_definition)
            }
        )
        automatic_definition = pick_nexus_definition(
            res.get("instrument_name", "unknown"),
            res.get("experiment_type", "Other"),
        )
        if automatic_definition not in available_definitions:
            automatic_definition = "NXem" if "NXem" in available_definitions else available_definitions[0]
        if st.session_state.get("_definition_selector") not in available_definitions:
            st.session_state._definition_selector = automatic_definition

        selected_definition = st.selectbox(
            "NeXus application definition",
            available_definitions,
            key="_definition_selector",
            help=(
                "The automatic choice is selected initially. Override it when a TIFF header "
                "does not contain enough information to identify the technique."
            ),
        )
        if selected_definition != st.session_state.get("nexus_definition"):
            xml_path, allowed_paths = load_definition_schema_xml(selected_definition)
            st.session_state.nexus_definition = selected_definition
            st.session_state.nxdl_path = str(xml_path) if xml_path else None
            st.session_state.allowed_paths = sorted(allowed_paths) if allowed_paths else []
            st.session_state["_stash_field_path"] = _find_best_stash_field(
                st.session_state.allowed_paths
            )
            st.session_state.pop("nexus_export", None)

        st.subheader("🔍 LLM Raw Responses")
        st.text_area("Instrument response (raw)", res.get("raw_instrument", ""), height=140)
        st.text_area("Experiment-type response (raw)", res.get("raw_experiment", ""), height=140)

        st.subheader("✅ Parsed Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Instrument", res.get("instrument_name", "unknown"))
        with col2:
            st.metric("Experiment", res.get("experiment_type", "Other"))

        st.subheader("📚 Selected NeXus application")
        colA, colB = st.columns([2, 3])
        with colA:
            st.write("**Definition:**", st.session_state.get("nexus_definition") or "—")
            st.write("**NXDL file:**", st.session_state.get("nxdl_path") or "not found")
        with colB:
            aps = st.session_state.get("allowed_paths", [])
            st.write(f"**Allowed ontology paths:** {len(aps)} found")
            if aps:
                st.code("\n".join(aps[:60]), language="text")

        # Show stash field discovery
        stash_field = st.session_state.get("_stash_field_path", "")
        st.subheader("📦 Unmapped metadata stash (schema-valid)")
        if stash_field:
            st.success(f"Stash field found in NXDL: **{stash_field}**")
            st.caption(
                "Unmapped keys will be stored as a JSON blob in this existing field (no invented paths)."
            )
        else:
            st.warning(
                "No obvious NXnote/NXprocess description-like field was found in allowed_paths. "
                "Unmapped keys will still be listed, but you may need to pick a storage location manually."
            )

        st.subheader("🧭 Map metadata keys to NeXus paths")

        aps_list = st.session_state.get("allowed_paths", [])
        safe_md = st.session_state.get("safe_metadata", {})
        pretty_md = st.session_state.get("pretty_metadata", "")

        if not aps_list:
            st.info("No allowed paths found for the selected definition — cannot map.")
        else:
            definition = st.session_state.get("nexus_definition", "NXem")
            example_indexes = st.session_state.setdefault("_example_indexes", {})
            if definition not in example_indexes:
                pairs = load_all_example_pairs_from_dir(
                    EXAMPLE_MAPPINGS_DIR,
                    definition_prefix=definition,
                )
                example_indexes[definition] = build_example_index(pairs)

            index = example_indexes[definition]

            mapping_mode = st.radio(
                "Mapping mode",
                ["Hybrid (LLM only when uncertain)", "Deterministic (no LLM)", "LLM always"],
                index=0,
            )
            st.caption(f"Deterministic thresholds: CONF_MIN={CONF_MIN}, LOGRATIO_MIN={LOGRATIO_MIN}")

            # (unchanged) load curated/db mapping if present
            accepted = {}
            rejected = {}
            raw_log = {}

            cached = _mapping_db_get(res.get("instrument_name", "unknown"), st.session_state.get("nexus_definition", "NXem"))
            if cached and isinstance(cached.get("mapping"), dict):
                accepted = cached.get("mapping") or {}

            # Key selection
            meta_keys = flatten_metadata_keys(safe_md)
            st.caption("Choose how many keys to include (in order).")
            max_keys = st.slider(
                "Number of keys to map",
                min_value=1,
                max_value=min(300, len(meta_keys)),
                value=min(50, len(meta_keys)),
            )
            keys_to_map = meta_keys[:max_keys]

            st.markdown("**Keys selected:**")
            st.code("\n".join(keys_to_map[:80]), language="text")

            show_rank_debug = st.checkbox("Show candidate ranking (debug)", value=False)
            st.caption(f"Per-key candidate paths (fixed): {PER_KEY_CANDIDATE_LIMIT}")

            button_label = "🤖 Map ALL selected keys"
            if mapping_mode == "Hybrid (LLM only when uncertain)":
                button_label = "🤖 Map ALL selected keys (Hybrid: LLM only when uncertain)"
            elif mapping_mode == "Deterministic (no LLM)":
                button_label = "🤖 Map ALL selected keys (Deterministic: NO LLM)"
            else:
                button_label = "🤖 Map ALL selected keys (LLM always)"

            if st.button(button_label):
                accepted = dict(accepted or {})
                rejected = dict(rejected or {})
                raw_log = dict(raw_log or {})

                # NEW: stash bucket for unmapped keys (A)
                stashed: dict[str, dict] = {}
                stash_field = st.session_state.get("_stash_field_path", "")

                md_snippet = pretty_md if len(pretty_md) < 3000 else pretty_md[:3000] + "\n…(truncated)…"

                progress = st.progress(0)
                status = st.empty()
                total_keys = len(keys_to_map)

                det_used = 0
                llm_used = 0
                stashed_count = 0
                reused_count = 0
                curated_used = 0
                newly_mapped_count = 0

                for idx, key in enumerate(keys_to_map, start=1):
                    if key in accepted:
                        reused_count += 1
                        progress.progress(idx / total_keys)
                        continue

                    status.write(f"Processing {idx}/{total_keys}: **{key}**")
                    value_preview = get_value_preview(safe_md, key)
                    sibling_keys = [k for k in keys_to_map if k != key]

                    # Try exact-key reuse from curated examples if present
                    exact_path = next(
                        (
                            index.get("exact_key_map", {}).get(alias)
                            for alias in _metadata_key_aliases(key)
                            if index.get("exact_key_map", {}).get(alias)
                        ),
                        None,
                    )
                    if exact_path and exact_path in aps_list:
                        curated_used += 1
                        # Minimal safety rule: stash encoding keys if mapping is outside safe zones
                        if _should_stash_encoding_key(key, exact_path):
                            stashed[key] = {"value_preview": value_preview, "reason": "encoding-key safety (exact-key)"}
                            stashed_count += 1
                        else:
                            accepted[key] = exact_path
                            newly_mapped_count += 1
                        progress.progress(idx / total_keys)
                        continue

                    # Deterministic proposal
                    det = deterministic_pick_path(
                        aps_list=aps_list,
                        index=index,
                        definition=st.session_state.get("nexus_definition", "NXem"),
                        key=key,
                        value_preview=value_preview,
                        sibling_keys=sibling_keys,
                    )
                    proposed_path = _canonicalize_example_path(_normalize_path(det["chosen"]))

                    use_llm = False
                    if mapping_mode == "LLM always":
                        use_llm = True
                    elif mapping_mode == "Deterministic (no LLM)":
                        use_llm = False
                    else:
                        if det["confidence"] < CONF_MIN or det["log_ratio"] < LOGRATIO_MIN:
                            use_llm = True

                    if show_rank_debug:
                        st.markdown(
                            f"**{key} — deterministic** → `{proposed_path}` "
                            f"(conf={det['confidence']:.2f}, log_ratio={det['log_ratio']:.2f}, "
                            f"cluster={CLUSTERS[det['cluster_id']]['label']})"
                        )
                        st.code(json.dumps(det["candidates"], indent=2), language="json")

                    # If NOT using LLM, accept/reject/stash now
                    if not use_llm:
                        det_used += 1

                        # Minimal concrete suggestion: encoding-key safety guard
                        if _should_stash_encoding_key(key, proposed_path):
                            stashed[key] = {
                                "value_preview": value_preview,
                                "reason": "encoding-key safety (deterministic)",
                                "proposed_path": proposed_path,
                                "confidence": det["confidence"],
                                "log_ratio": det["log_ratio"],
                            }
                            stashed_count += 1
                            progress.progress(idx / total_keys)
                            continue

                        if proposed_path and proposed_path in aps_list:
                            accepted[key] = proposed_path
                            newly_mapped_count += 1
                            raw_log[key] = {
                                "mode": "deterministic",
                                "chosen": proposed_path,
                                "confidence": det["confidence"],
                                "log_ratio": det["log_ratio"],
                                "cluster": det["cluster_id"],
                            }
                        else:
                            # NEW: instead of just rejected, stash as unmapped (A)
                            stashed[key] = {
                                "value_preview": value_preview,
                                "reason": "no valid path (deterministic)",
                                "proposed_path": proposed_path,
                                "confidence": det["confidence"],
                                "log_ratio": det["log_ratio"],
                            }
                            stashed_count += 1
                        progress.progress(idx / total_keys)
                        continue

                    # LLM fallback
                    llm_used += 1
                    cluster_id = det["cluster_id"]
                    cluster_reason = det["cluster_reason"]

                    scored_examples = pick_similar_examples(index, cluster_id, key, top_n=6)
                    example_pairs = [(k, p) for (k, p, _) in scored_examples]

                    cluster_paths = filter_paths_by_cluster(aps_list, cluster_id)
                    cache_key = f"{st.session_state.get('nexus_definition','NXem')}::{cluster_id}"

                    allowed_subset = prioritize_paths_lexical(
                        allowed_paths=cluster_paths,
                        key=key,
                        value_preview=value_preview,
                        sibling_keys=sibling_keys,
                        cache_key=cache_key,
                        top_k=PER_KEY_CANDIDATE_LIMIT,
                        show_debug=show_rank_debug,
                    )

                    prompt = build_single_key_mapping_prompt(
                        definition=st.session_state.get("nexus_definition", "NXem"),
                        selected_key=key,
                        value_preview=value_preview,
                        allowed_subset=allowed_subset,
                        metadata_context_snippet=md_snippet,
                        example_pairs=example_pairs,
                    )

                    try:
                        resp = query_ollama(prompt, model=selected_model)
                        raw_log[key] = resp
                        if show_rank_debug:
                            raw_log[f"{key}__cluster"] = {"cluster": cluster_id, "reason": cluster_reason}
                            raw_log[f"{key}__examples"] = [
                                {"key": ek, "path": ep, "score": s} for (ek, ep, s) in scored_examples
                            ]

                        try:
                            parsed = extract_json_from_response(resp)
                        except Exception:
                            parsed = robust_json_from_text(resp)

                        llm_path = ""
                        if isinstance(parsed, dict):
                            mapping_obj = parsed.get("mapping", {})
                            if isinstance(mapping_obj, dict):
                                if key in mapping_obj and isinstance(mapping_obj[key], str):
                                    llm_path = mapping_obj[key]
                                else:
                                    for _, v in mapping_obj.items():
                                        if isinstance(v, str):
                                            llm_path = v
                                            break

                        llm_path = _canonicalize_example_path(_normalize_path(llm_path))

                        # Minimal concrete suggestion: encoding-key safety guard
                        if _should_stash_encoding_key(key, llm_path):
                            stashed[key] = {
                                "value_preview": value_preview,
                                "reason": "encoding-key safety (LLM)",
                                "proposed_path": llm_path,
                            }
                            stashed_count += 1
                        elif llm_path and llm_path in aps_list:
                            accepted[key] = llm_path
                            newly_mapped_count += 1
                        else:
                            # NEW: stash instead of plain reject (A)
                            stashed[key] = {
                                "value_preview": value_preview,
                                "reason": "no valid path (LLM)",
                                "proposed_path": llm_path,
                            }
                            stashed_count += 1

                    except Exception as e:
                        stashed[key] = {"value_preview": value_preview, "reason": f"LLM error: {e}"}
                        stashed_count += 1

                    progress.progress(idx / total_keys)

                stash_payload = {
                    key: {
                        **details,
                        "value": make_json_safe(get_metadata_value(safe_md, key)),
                    }
                    for key, details in stashed.items()
                }

                status.empty()
                st.info(
                    f"Selected: {total_keys} | Reused: {reused_count} | "
                    f"Newly mapped: {newly_mapped_count} | Stashed: {stashed_count}"
                )
                st.caption(
                    f"Methods — curated: {curated_used}, deterministic: {det_used}, "
                    f"LLM attempted: {llm_used}"
                )

                st.subheader("✅ Accepted (valid mappings)")
                st.json(accepted or {})

                # Keep a small rejected bucket for transparency, but prefer stash
                st.subheader("📦 Stashed (unmapped or unsafe-to-map) — schema-valid preservation")
                if not stashed:
                    st.success("No stashed keys — everything mapped cleanly.")
                else:
                    st.write(
                        "These keys were not mapped to a specific ontology path (or were prevented by the safety rule). "
                        "They will be preserved as a JSON blob in an existing NXDL field."
                    )
                    st.json(stashed)

                    # Show the JSON blob the user should write later
                    st.markdown("**Recommended stash field (must exist in NXDL):**")
                    st.code(stash_field or "(not found automatically)", language="text")

                    # Build blob from original values when possible (not just previews)
                    # Note: safe_md can be nested; if key is dotted, we show preview only here.
                    # If your keys are not dotted, you can use safe_md.get(key).
                    blob = {
                        "stashed_at": datetime.now(timezone.utc).isoformat(),
                        "definition": st.session_state.get("nexus_definition", "NXem"),
                        "instrument": res.get("instrument_name", "unknown"),
                        "keys": {k: stash_payload[k]["value"] for k in stashed},
                    }
                    blob_text = json.dumps(blob, indent=2, ensure_ascii=False)
                    st.markdown("**JSON blob to store at the stash field:**")
                    st.code(blob_text[:12000] + ("\n…(truncated)…" if len(blob_text) > 12000 else ""), language="json")

                    # Optional: store a special mapping entry so downstream knows where to write the blob
                    if stash_field:
                        accepted_with_stash = dict(accepted)
                        accepted_with_stash["__UNMAPPED_METADATA_JSON__"] = stash_field

                        with st.expander("Optional: include stash field in mapping output"):
                            st.write(
                                "If you want the mapping output itself to indicate where to store unmapped metadata, "
                                "use this version:"
                            )
                            st.json(accepted_with_stash)

                with st.expander("LLM raw responses per key (debug)"):
                    st.code(json.dumps(raw_log, indent=2), language="json")

                # Persist accepted mappings and full values for keys that need a stash.
                instrument_name = res.get("instrument_name", "unknown")
                definition = st.session_state.get("nexus_definition", "NXem")
                if accepted or stashed:
                    mapping_full = {
                        "stash_field": stash_field,
                        "stashed": stash_payload,
                    }
                    _mapping_db_put(instrument_name, definition, accepted, mapping_full=mapping_full)
                    st.success(
                        f"Saved mappings and stashed metadata for "
                        f"**{instrument_name} | {definition}** → mapping_db.json"
                    )

                    try:
                        nexus_bytes, export_report = build_nexus_file(
                            image_array=st.session_state.image_array,
                            source_filename=filename,
                            metadata=safe_md,
                            mapping=accepted,
                            definition=definition,
                            instrument=instrument_name,
                            stashed_metadata=stash_payload,
                        )
                        output_stem = Path(filename).stem or "mapped_metadata"
                        st.session_state.nexus_export = {
                            "bytes": nexus_bytes,
                            "filename": f"{output_stem}_{definition}.nxs",
                            "report": export_report,
                            "source_sha256": source_sha256,
                        }
                    except Exception as e:
                        st.session_state.pop("nexus_export", None)
                        st.error(f"Could not create the NeXus file: {e}")

            nexus_export = st.session_state.get("nexus_export")
            if nexus_export and nexus_export.get("source_sha256") == source_sha256:
                export_report = nexus_export.get("report", {})
                st.subheader("💾 Export mapped NeXus file")
                st.download_button(
                    "Download NeXus (.nxs)",
                    data=nexus_export["bytes"],
                    file_name=nexus_export["filename"],
                    mime="application/x-hdf5",
                    key="download_nexus_file",
                )
                st.caption(
                    f"Written mapped fields: {export_report.get('written_count', 0)} | "
                    f"Unresolved accepted mappings: {export_report.get('unresolved_count', 0)} | "
                    f"Previously stashed keys: {export_report.get('stashed_input_count', 0)}"
                )
                if export_report.get("unresolved_count", 0):
                    st.warning(
                        "Some accepted mappings targeted an NX group or collided with another dataset. "
                        "They were preserved under /entry/unmapped_metadata instead of being overwritten."
                    )
                with st.expander("NeXus export report"):
                    st.json(export_report)

else:
    st.info("This workflow will be implemented in a later step. For now, use ‘Upload image + metadata’.")

st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

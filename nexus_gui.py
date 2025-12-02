import streamlit as st
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import Counter

from upload_utils import (
    get_ollama_models,
    query_ollama,
    read_image_any_format,
    convert_to_preview,
    extract_metadata_from_tiff,
    extract_metadata_from_json_header,
    make_json_safe,
    extract_json_from_response,
)

# =========================================
# Constants
# =========================================
PER_KEY_CANDIDATE_LIMIT = 50  # fixed; not shown in the UI
_WORD_RE = re.compile(r"[a-z0-9_]+")

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
env_root = os.getenv("NEXUS_DEFINITIONS_ROOT", "").strip()
DEF_ROOT = Path(env_root) if env_root else (BASE_DIR / "nexus_definitions")
APP_DIRS = [DEF_ROOT / "applications", DEF_ROOT / "contributed_definitions"]

# Folder where you put curated example mappings (per-instrument)
EXAMPLE_MAPPINGS_DIR = BASE_DIR / "example_mappings"

# =========================================
# JSON extraction helper
# =========================================
def robust_json_from_text(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    candidate = m.group(0)
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', candidate)
    candidate = re.sub(r',\s*(?=[}\]])', '', candidate)
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
{
  "instrument": "TESCAN Vega3"
}

or if missing:
{
  "instrument": "unknown"
}
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
{
  "experiment": "SEM"
}
{
  "experiment": "Other"
}

❌ Do NOT include explanations, markdown, or extra text.
✅ Return only one valid JSON object.
"""
    )

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
        example_lines = "\n".join(
            f'- "{k}" → "{p}"' for (k, p) in example_pairs
        )
        examples_block = f"""
Example mappings from other {definition} instruments
(metadata key → NeXus path):
{example_lines}

Use these as guidance: if the current key is similar to an example key,
prefer a NeXus path that plays an analogous role.
"""
    else:
        examples_block = "\n(No example mappings available for this key.)\n"

    return f"""
You are a FAIRmat metadata assistant.

Goal:
Given ONE metadata key and its value, pick the **best matching NeXus path** from the allowed list.

Important:
- You should **prefer choosing the best candidate** over returning an empty string.
- Return "" only if **none** of the allowed paths are meaningfully related to this metadata key.
- It is acceptable to make a reasonable choice when multiple paths are plausible.

Selected NeXus application:
{definition}

Metadata context (truncated):
{metadata_context_snippet}

Metadata key to map:
- key: "{selected_key}"
- value preview: {value_preview}

{examples_block}

Allowed NeXus paths (choose at most one; do NOT invent):
{allowed_text}

Strict output rules:
- Output ONLY JSON (no prose), with this exact shape:
  {{
    "mapping": {{"{selected_key}": "<one_allowed_path_or_empty>"}}
  }}
- The chosen path MUST be one of the allowed paths above (or "").
- You may reuse a path from the examples if it is also present in the allowed list.
- Prefer the most specific field if multiple are reasonable.
"""

# ==================================================
# Tokenisation helper
# ==================================================
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tok(text: str) -> list[str]:
    return _WORD_RE.findall(_norm(text))

# ==================================================
# Lexical ranking of NeXus paths
# ==================================================
_GENERIC_LEAVES = {"name", "title", "value", "data", "type", "model"}  # mild penalty


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
        if not p or len(p) >= 200:
            continue
        parts = _split_path_parts(p)
        leaf = parts[-1].lower() if parts else ""
        tokens = _build_path_doc_tokens(p)
        entry = {
            "path": p,
            "leaf": leaf,
            "tokens": tokens,
            "bigrams": _bigrams(tokens),
            "depth": len(parts),
        }
        index.append(entry)
    return index


def _lexical_score(query_tokens: list[str], query_bigrams: set[tuple[str, str]], entry: dict) -> float:
    doc_tokens = entry["tokens"]
    doc_bigrams = entry["bigrams"]
    leaf = entry["leaf"]
    depth = entry["depth"]
    path_l = entry["path"].lower()

    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)

    # unigram overlap
    uni = 0
    for t, qc in q_counts.items():
        if t in d_counts:
            uni += min(qc, d_counts[t])

    # bigram overlap
    bi = len(query_bigrams & doc_bigrams)

    # substring hits (e.g., width in /image_width/)
    sub_hits = 0
    for t in set(query_tokens):
        if len(t) >= 3:
            if t in leaf:
                sub_hits += 1
            elif t in path_l:
                sub_hits += 0.5

    score = 3.0 * uni + 2.0 * bi + 1.0 * sub_hits + 0.25 * min(depth, 12)

    # penalize very generic leaf names if they don't appear in the query
    if (leaf in _GENERIC_LEAVES) and (leaf not in query_tokens):
        score -= 1.0
    return score


def make_query_tokens(key: str, value_preview: str, sibling_keys: list[str]) -> tuple[list[str], set[tuple[str, str]]]:
    sib_snip = " ".join(sibling_keys[:6]) if sibling_keys else ""
    q_text = f"{key} {value_preview} {sib_snip}"
    q_tokens = _tok(q_text)
    q_bigrams = _bigrams(q_tokens)
    return q_tokens, q_bigrams


def _get_path_index_for_definition(allowed_paths: list[str], definition: str) -> list[dict]:
    """
    Cache a path index per definition in the session state.
    """
    key_idx = "_path_index"
    key_def = "_path_index_definition"
    if (key_idx not in st.session_state) or (st.session_state.get(key_def) != definition):
        st.session_state[key_idx] = build_path_index(allowed_paths)
        st.session_state[key_def] = definition
    return st.session_state[key_idx]


def prioritize_paths_lexical(
    allowed_paths: list[str],
    key: str,
    value_preview: str,
    sibling_keys: list[str],
    definition: str,
    top_k: int = PER_KEY_CANDIDATE_LIMIT,
    show_debug: bool = False,
) -> list[str]:
    path_index = _get_path_index_for_definition(allowed_paths, definition)
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
            st.write(f"{i:02d}. {p} — score={sc:.1f}, depth={depth}")

    return [p for p, _, _ in top]

# ==============================
# Helpers for mapping
# ==============================
def flatten_metadata_keys(data, parent=""):
    keys = []
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{parent}.{k}" if parent else k
            keys.extend(flatten_metadata_keys(v, path))
    elif isinstance(data, list):
        keys.append(parent)
    else:
        keys.append(parent)
    return list(dict.fromkeys(keys))


def get_value_preview(data: dict, dotted_key: str, max_len: int = 240) -> str:
    try:
        node = data
        for part in dotted_key.split("."):
            if isinstance(node, dict):
                node = node.get(part, None)
            else:
                node = None
                break
        s = repr(node)
    except Exception:
        s = "None"
    if s is None:
        s = "None"
    s = str(s)
    return (s[:max_len] + "…") if len(s) > max_len else s

# ==============================
# NXDL filesystem + XML parsing
# ==============================
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

# Optional explicit instrument→definition overrides
INSTRUMENT_TO_NX_DEF = {
    # "tescan amber x": "NXem",
    # "nova nanosem 450": "NXem",
}

def normalize_instrument_name(name: str | None) -> str:
    return (name or "").strip().lower()


def pick_nexus_definition(instrument_name: str, experiment_type: str) -> str:
    """
    First use explicit instrument→definition overrides if present.
    Otherwise fall back to experiment_type→EXPERIMENT_TO_NX.
    """
    inst_norm = normalize_instrument_name(instrument_name)
    for inst_key, nx_def in INSTRUMENT_TO_NX_DEF.items():
        if inst_key in inst_norm:
            return nx_def
    return EXPERIMENT_TO_NX.get(experiment_type, "NXem")


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_definition_xml(definition: str) -> Path | None:
    """
    Find the best NXDL file for a given application definition.

    Heuristics:
    - Only consider stems that start with the requested definition, e.g. "nxem".
    - Prefer an exact stem match ("nxem") over suffixed variants ("nxem_eels").
    - Prefer files in 'applications/' over 'contributed_definitions/'.
    - Among the rest, prefer shorter stems (more general).
    """
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
            elif "contributed_definitions" in parent_str:
                score += 0.0

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
    if curr:
        paths.add(curr)
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

# ==============================
# Curated example mappings (per-instrument) + global example pairs
# ==============================
_EXAMPLE_MAPPING_CACHE: dict[str, dict[str, str]] = {}  # filepath -> {meta_key -> nx_path}
_ALL_EXAMPLE_PAIRS_CACHE: dict[str, list[tuple[str, str]]] = {}  # definition -> [(meta_key, nx_path)]


def normalize_instr_for_filename(name: str) -> str:
    """Normalize instrument name to a safe filename fragment."""
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _canonicalize_example_path(p: str) -> str:
    """
    Convert example paths like '/ENTRY/...' or 'ENTRY/...' into NXDL-style paths:
      'NXentry/...'
    and remove any leading slash.
    """
    if not isinstance(p, str):
        return p

    p = p.strip()

    # Replace common variants of ENTRY with NXentry
    if p.startswith("/ENTRY"):
        p = "NXentry" + p[len("/ENTRY") :]
    elif p.startswith("ENTRY"):
        p = "NXentry" + p[len("ENTRY") :]
    elif p.startswith("/NXentry"):
        p = p[1:]  # drop leading slash

    if p.startswith("/"):
        p = p[1:]

    return p


def mapping_path_for(definition: str, instrument_name: str) -> Path:
    """
    Build the expected JSON path for a given (definition, instrument).
    Example: NXem + 'Nova NanoSEM 450'
    -> example_mappings/NXem__nova_nanosem_450.json
    """
    instr_norm = normalize_instr_for_filename(instrument_name)
    fname = f"{definition}__{instr_norm}.json"
    return EXAMPLE_MAPPINGS_DIR / fname


def _load_single_example_mapping(path: Path) -> dict[str, str] | None:
    """
    Load one example mapping file and convert to {meta_key -> nx_path}.
    """
    global _EXAMPLE_MAPPING_CACHE

    key = str(path)
    if key in _EXAMPLE_MAPPING_CACHE:
        return _EXAMPLE_MAPPING_CACHE[key]

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    mapping: dict[str, str] = {}
    for meta_key, info in raw.items():
        if isinstance(info, dict) and "nx_path" in info:
            nx_path = _canonicalize_example_path(str(info["nx_path"]))
            mapping[str(meta_key)] = nx_path

    if not mapping:
        return None

    _EXAMPLE_MAPPING_CACHE[key] = mapping
    return mapping


def get_example_mapping_for(instrument_name: str, definition: str) -> dict[str, str] | None:
    """
    Instrument-specific curated mapping: (definition, instrument) → {key -> nx_path}
    """
    if not instrument_name:
        return None

    path = mapping_path_for(definition, instrument_name)
    return _load_single_example_mapping(path)


def _load_all_example_pairs(definition: str) -> list[tuple[str, str]]:
    """
    Build a global list of (meta_key, nx_path) pairs for one definition
    by reading ALL example JSON files for that definition (NXem__*.json).
    """
    global _ALL_EXAMPLE_PAIRS_CACHE
    if definition in _ALL_EXAMPLE_PAIRS_CACHE:
        return _ALL_EXAMPLE_PAIRS_CACHE[definition]

    pairs: list[tuple[str, str]] = []

    if EXAMPLE_MAPPINGS_DIR.exists():
        pattern = f"{definition}__*.json"
        for p in EXAMPLE_MAPPINGS_DIR.glob(pattern):
            m = _load_single_example_mapping(p)
            if not m:
                continue
            for k, nx_path in m.items():
                pairs.append((k, nx_path))

    _ALL_EXAMPLE_PAIRS_CACHE[definition] = pairs
    return pairs


def _key_similarity_score(q_key: str, ex_key: str) -> float:
    """
    Simple lexical similarity between two metadata keys.
    """
    q_tokens = set(_tok(q_key))
    e_tokens = set(_tok(ex_key))
    if not q_tokens or not e_tokens:
        return 0.0

    overlap = len(q_tokens & e_tokens)
    substring = 1.0 if (ex_key.lower() in q_key.lower() or q_key.lower() in ex_key.lower()) else 0.0
    return 2.0 * overlap + substring


def get_example_pairs_for_key(definition: str, key: str, top_n: int = 6) -> list[tuple[str, str]]:
    """
    For a given key, return the top-N most similar example key→path pairs
    from all known instruments for this definition.
    """
    all_pairs = _load_all_example_pairs(definition)
    if not all_pairs:
        return []

    scored = []
    for ex_key, nx_path in all_pairs:
        s = _key_similarity_score(key, ex_key)
        if s > 0:
            scored.append((ex_key, nx_path, s))

    if not scored:
        return []

    scored.sort(key=lambda t: t[2], reverse=True)
    top = scored[:top_n]
    return [(k, p) for (k, p, _) in top]

# ==============================
# Mapping DB (instrument + definition)
# ==============================
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
    except Exception as e:
        st.warning(f"Could not save mapping DB: {e}")


def _db_key(instrument: str, definition: str) -> str:
    inst = (instrument or "unknown").strip()[:120]
    dfn = (definition or "NXem").strip()[:80]
    return f"{inst} | {dfn}"


def _mapping_db_get(instrument: str, definition: str) -> dict | None:
    db = _mapping_db_load()
    return db.get(_db_key(instrument, definition))


def _mapping_db_put(instrument: str, definition: str, mapping: dict) -> None:
    db = _mapping_db_load()
    db[_db_key(instrument, definition)] = {
        "mapping": mapping or {},
        "definition": definition,
        "instrument": instrument,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _mapping_db_save(db)

# ==============================
# Small normalization helper
# ==============================
def _normalize_path(p: str) -> str:
    """Drop a leading slash if the model returns '/ENTRY/...' instead of 'ENTRY/...'. """
    if not isinstance(p, str):
        return p
    return p[1:] if p.startswith("/") else p

# ==============================
# App UI
# ==============================
load_dotenv()
LOGO_NFFA = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/nffa.png"
LOGO_LADE = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/area.png"

st.set_page_config(
    page_title="FAIR NeXus File Metadata Assistant — Step 1 + Mapping (beta)",
    layout="centered",
)

with st.sidebar:
    st.image(LOGO_NFFA, caption="NFFA-DI")
    st.image(LOGO_LADE, caption="AREA Science Park")

st.title("🔬 FAIR NeXus Assistant — Step 1 + Mapping (beta)")
st.caption(
    "Upload → preview → extract metadata → instrument/experiment via LLM → select NXDL → "
    "map keys to NeXus paths (curated examples + lexical ranking + LLM)."
)

# LLM model selection
models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on your Ollama host (e.g., `ollama run llama3`), then refresh.")
    st.stop()

selected_model = st.selectbox("LLM model served by Ollama", models)
workflow = st.radio("Choose a workflow", ["Upload image + metadata", "Reference a stack (folder / S3 URL)"])

# Basic metadata
if "meta" not in st.session_state:
    st.session_state.meta = {"instrument": "", "sample_id": "", "operator": ""}

st.subheader("📄 Basic metadata")
for k in list(st.session_state.meta.keys()):
    st.session_state.meta[k] = st.text_input(
        k.replace("_", " ").title(), st.session_state.meta[k], key=f"meta_{k}"
    )

meta = st.session_state.meta

# ==============================
# Workflow: Upload image + metadata
# ==============================
if workflow == "Upload image + metadata":
    file = st.file_uploader(
        "Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader"
    )
    header = st.file_uploader(
        "(Optional) Upload separate metadata (JSON)", type="json", key="json_uploader"
    )

    if file is None:
        st.info("👆 Upload an image to get started.")
        st.stop()

    # Preview image
    image = read_image_any_format(file)
    st.session_state.image_array = image
    st.session_state.preview = convert_to_preview(image)
    st.image(st.session_state.preview, caption="Preview", use_container_width=True)

    # Extract metadata
    instrument_metadata = {}
    filename = getattr(file, "name", "")
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

    # Pretty JSON for prompts + keep in session for mapping step
    safe_metadata = make_json_safe(instrument_metadata)
    pretty_metadata = json.dumps(safe_metadata, indent=2)
    st.session_state.safe_metadata = safe_metadata
    st.session_state.pretty_metadata = pretty_metadata

    # --------------------
    # LLM (instrument + experiment) + NXDL load
    # --------------------
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

            # Instrument
            try:
                raw_inst = query_ollama(
                    build_instrument_detection_prompt(pretty_metadata),
                    model=selected_model,
                )
                results["raw_instrument"] = raw_inst
                try:
                    parsed_inst = extract_json_from_response(raw_inst)
                except Exception:
                    parsed_inst = robust_json_from_text(raw_inst)
                results["instrument_name"] = parsed_inst.get("instrument", "unknown")
            except Exception as e:
                st.error(f"❌ Failed to extract instrument: {e}")

            # Experiment type
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

            # Pick NX app using instrument overrides if any, else experiment type
            definition = pick_nexus_definition(
                instrument_name=results["instrument_name"],
                experiment_type=results["experiment_type"],
            )
            xml_path, allowed_paths = load_definition_schema_xml(definition)

            st.session_state.step1_results = results
            st.session_state.nexus_definition = definition
            st.session_state.nxdl_path = str(xml_path) if xml_path else None
            st.session_state.allowed_paths = sorted(list(allowed_paths)) if allowed_paths else []

    # Show results (if any)
    if "step1_results" in st.session_state:
        res = st.session_state.step1_results
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

        # ==============================
        # Map keys → NeXus paths
        # ==============================
        st.subheader("🧭 Map metadata keys to NeXus paths")

        aps = st.session_state.get("allowed_paths", [])
        safe_md = st.session_state.get("safe_metadata", {})
        pretty_md = st.session_state.get("pretty_metadata", "")
        instrument_name = st.session_state.step1_results.get("instrument_name", "unknown")
        definition = st.session_state.get("nexus_definition", "NXem")

        if not aps:
            st.info("No allowed paths found for the selected definition — cannot map.")
        else:
            # ----------------------------------
            # 1) Try curated example mapping first (instrument-specific)
            # ----------------------------------
            curated = get_example_mapping_for(instrument_name, definition)

            if curated:
                st.success(
                    f"Loaded curated example mapping for **{instrument_name} | {definition}** "
                    f"from JSON file `{mapping_path_for(definition, instrument_name).name}`"
                )
                st.subheader("📘 Curated mapping (from example JSON)")
                st.json(curated)

                st.session_state.bulk_mapping_accepted = curated.copy()
                st.session_state.bulk_mapping_rejected = {}
                st.session_state.bulk_mapping_rawlog = {}

                if not st.button("♻️ Extend mapping with LLM (only for missing keys)"):
                    st.stop()
                else:
                    st.info("Extending curated mapping with LLM for keys not covered in the example...")

            else:
                # ----------------------------------
                # 2) If no curated mapping, try DB
                # ----------------------------------
                cached = _mapping_db_get(instrument_name, definition)
                if cached and isinstance(cached.get("mapping"), dict):
                    cached_map = cached["mapping"]
                    valid_cached = cached_map

                    st.success(
                        f"Loaded mapping from local DB (mapping_db.json) for **{instrument_name} | {definition}**"
                    )
                    st.caption(f"Updated at: {cached.get('updated_at', '—')}")
                    st.subheader("✅ Mapping (from DB)")
                    st.json(valid_cached or {})

                    st.session_state.bulk_mapping_accepted = valid_cached
                    st.session_state.bulk_mapping_rejected = {}
                    st.session_state.bulk_mapping_rawlog = {}

                    if not st.button("♻️ Re-map with LLM (ignore cache)"):
                        st.stop()
                    else:
                        st.info("Ignoring cache and re-mapping with LLM from scratch...")

            # ----------------------------------
            # 3) LLM-based mapping for remaining keys (with global examples)
            # ----------------------------------
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

            if st.button("🤖 Map ALL selected keys (LLM for missing ones)"):
                accepted = dict(st.session_state.get("bulk_mapping_accepted", {}))
                rejected = dict(st.session_state.get("bulk_mapping_rejected", {}))
                raw_log = dict(st.session_state.get("bulk_mapping_rawlog", {}))

                md_snippet = (
                    pretty_md if len(pretty_md) < 3000 else pretty_md[:3000] + "\n…(truncated)…"
                )

                progress = st.progress(0)
                status = st.empty()
                total_keys = len(keys_to_map)

                for idx, key in enumerate(keys_to_map, start=1):
                    if key in accepted:
                        progress.progress(idx / total_keys)
                        continue

                    status.write(f"Processing {idx}/{total_keys}: **{key}**")
                    value_preview = get_value_preview(safe_md, key)
                    sibling_keys = [k for k in keys_to_map if k != key]

                    # --- global example pairs (from all instruments for this definition)
                    example_pairs = get_example_pairs_for_key(definition, key, top_n=6)
                    
                    # NEW: show which example mappings are being used for this key
                    if show_rank_debug and example_pairs:
                        st.markdown(f"**{key} — example mappings used**")
                        lines = [f'"{ex_key}" → "{ex_path}"' for ex_key, ex_path in example_pairs]
                        st.code("\n".join(lines), language="text")

                    # 1) exact-key reuse: if an example key matches exactly, and its path is allowed, reuse it
                    exact_match = None
                    key_l = key.lower()
                    for ex_key, ex_path in example_pairs:
                        if ex_key.lower() == key_l:
                            exact_match = ex_path
                            break

                    if exact_match is not None:
                        if exact_match in aps:
                            accepted[key] = exact_match
                            progress.progress(idx / total_keys)
                            continue
                        # if not in aps, we still pass it as example but don't auto-accept

                    # 2) build candidate list via lexical ranking
                    allowed_subset = prioritize_paths_lexical(
                        allowed_paths=aps,
                        key=key,
                        value_preview=value_preview,
                        sibling_keys=sibling_keys,
                        definition=definition,
                        top_k=PER_KEY_CANDIDATE_LIMIT,
                        show_debug=show_rank_debug,
                    )

                    prompt = build_single_key_mapping_prompt(
                        definition=definition,
                        selected_key=key,
                        value_preview=value_preview,
                        allowed_subset=allowed_subset,
                        metadata_context_snippet=md_snippet,
                        example_pairs=example_pairs,
                    )

                    try:
                        resp = query_ollama(prompt, model=selected_model)
                        raw_log[key] = resp

                        try:
                            parsed = extract_json_from_response(resp)
                        except Exception:
                            parsed = robust_json_from_text(resp)

                        proposed_path = ""
                        if isinstance(parsed, dict):
                            mapping_obj = parsed.get("mapping", {})
                            if isinstance(mapping_obj, dict):
                                if key in mapping_obj and isinstance(mapping_obj[key], str):
                                    proposed_path = mapping_obj[key]
                                else:
                                    for _, v in mapping_obj.items():
                                        if isinstance(v, str):
                                            proposed_path = v
                                            break

                        proposed_path = _normalize_path(_canonicalize_example_path(proposed_path))

                        if proposed_path and proposed_path in aps:
                            accepted[key] = proposed_path
                        else:
                            rejected[key] = proposed_path

                    except Exception as e:
                        raw_log[key] = f"ERROR: {e}"
                        rejected[key] = ""

                    progress.progress(idx / total_keys)

                status.empty()

                st.subheader("✅ Accepted (valid / curated + examples + LLM)")
                st.json(accepted or {})

                st.subheader("⚠️ Rejected or empty (review needed)")
                st.json(rejected or {})

                with st.expander("LLM raw responses per key (debug)"):
                    st.code(json.dumps(raw_log, indent=2), language="json")

                st.session_state.bulk_mapping_accepted = accepted
                st.session_state.bulk_mapping_rejected = rejected
                st.session_state.bulk_mapping_rawlog = raw_log

                if accepted:
                    _mapping_db_put(instrument_name, definition, accepted)
                    st.success(
                        f"Saved mapping to DB for **{instrument_name} | {definition}** → mapping_db.json"
                    )

# ==============================
# Other workflow placeholder
# ==============================
else:
    st.info("This workflow will be implemented in a later step. For now, use ‘Upload image + metadata’.")

st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

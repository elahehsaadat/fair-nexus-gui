import streamlit as st
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from datetime import datetime

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
PER_KEY_CANDIDATE_LIMIT = 300  # hidden from UI as requested

# ==============================
# Robust JSON extraction helper
# ==============================

def robust_json_from_text(text: str) -> dict:
    m = re.search(r'\{.*?\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    candidate = m.group(0)
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', candidate)
    candidate = re.sub(r',\s*(?=[}\]])', '', candidate)
    decoder = json.JSONDecoder(strict=False)
    return decoder.decode(candidate)

# ==============================
# Prompt builders (instrument & experiment)
# ==============================

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

def prioritize_paths(allowed_paths: list[str], hint_terms: list[str], max_paths: int = 300) -> list[str]:
    if not allowed_paths:
        return []
    lower = [(p, p.lower()) for p in allowed_paths]
    def score(lp: str) -> int:
        s = 0
        for t in hint_terms:
            if t in lp:
                s += 2
        if "nxinstrument" in lp:
            s += 1
        return s
    ranked = sorted(lower, key=lambda item: score(item[1]), reverse=True)
    return [p for p, _ in ranked[:max_paths]]

def build_single_key_mapping_prompt(
    definition: str,
    selected_key: str,
    value_preview: str,
    allowed_subset: list[str],
    metadata_context_snippet: str
) -> str:
    allowed_text = "\n".join("- " + p for p in allowed_subset) if allowed_subset else "- (no paths)"
    key_l = selected_key.lower()
    is_time_like = any(t in key_l for t in ["time", "date", "timestamp", "datetime", "start", "end"])
    return f"""
You are a FAIRmat metadata assistant.

Goal:
Given ONE metadata key and its value, pick the **best matching NeXus path** from the allowed list.
If there is **no clear match**, you MUST return an empty string "" (do not guess).

Selected NeXus application:
{definition}

Metadata context (truncated):
{metadata_context_snippet}

Metadata key to map:
- key: "{selected_key}"
- value preview: {value_preview}

Allowed NeXus paths (choose at most one; do NOT invent):
{allowed_text}

Strict rules:
- Output ONLY JSON (no prose), with this exact shape:
  {{
    "mapping": {{"{selected_key}": "<one_allowed_path_or_empty>"}}
  }}
- If no path clearly matches, return "".
- Do **NOT** choose a time/date related path (e.g., NXentry/start_time, end_time) **unless** the key clearly expresses time/date semantics.
- Prefer the most specific field if multiple are reasonable.

Good examples:
{{
  "mapping": {{"DateTime": "NXentry/start_time"}}
}}
{{
  "mapping": {{"ImageWidth": ""}}
}}
"""

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
    "Other": "NXem",
}

BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
env_root = os.getenv("NEXUS_DEFINITIONS_ROOT", "").strip()
DEF_ROOT = Path(env_root) if env_root else (BASE_DIR / "nexus_definitions")
APP_DIRS = [DEF_ROOT / "applications", DEF_ROOT / "contributed_definitions"]

def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag

def find_definition_xml(definition: str) -> Path | None:
    stem_lower = definition.lower()
    for folder in APP_DIRS:
        if not folder.exists():
            continue
        for p in folder.rglob("*.nxdl.xml"):
            s = p.stem.lower()
            if s == stem_lower or s.startswith(stem_lower):
                return p
    return None

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

    if debug and not allowed:
        print(f"[NXDL] Zero paths collected for {definition} at {xml_path}")
        try:
            for i, el in enumerate(root.iter()):
                if i >= 25: break
                print("  TAG:", _strip_ns(el.tag), "ATTR:", el.attrib)
        except Exception as e:
            print("[NXDL] Debug iteration error:", e)

    return xml_path, allowed

# ==============================
# Mapping DB (instrument + definition)
# ==============================
MAPPING_DB_PATH = (BASE_DIR / "mapping_db.json")

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
    dfn  = (definition or "NXem").strip()[:80]
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
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _mapping_db_save(db)

# --- Guards for bad mappings ---
def _looks_time_path(p: str) -> bool:
    if not isinstance(p, str):
        return False
    pl = p.lower()
    return ("start_time" in pl) or ("end_time" in pl) or pl.endswith("/time") or "/time/" in pl or "/date" in pl

def _suspicious_collapse(accepted_map: dict) -> tuple[bool, str]:
    """Return (is_suspicious, reason)."""
    if not accepted_map:
        return (False, "")
    values = list(accepted_map.values())
    most = max((values.count(v), v) for v in set(values))
    if most[0] / max(1, len(values)) >= 0.5:
        return (True, f"More than half of keys map to the same path: {most[1]}")
    non_time_keys = [k for k in accepted_map if not any(t in k.lower() for t in ["time", "date", "timestamp", "datetime", "start", "end"])]
    non_time_to_time = [k for k in non_time_keys if _looks_time_path(accepted_map[k])]
    if len(non_time_keys) > 0 and (len(non_time_to_time) / len(non_time_keys)) >= 0.3:
        return (True, "Too many non-time keys mapped to time-related paths")
    return (False, "")

# ==============================
# App UI
# ==============================

load_dotenv()
LOGO_NFFA = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/nffa.png"
LOGO_LADE = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/area.png"

st.set_page_config(page_title="FAIR NeXus File Metadata Assistant — Step 1 + Mapping (beta)", layout="centered")

with st.sidebar:
    st.image(LOGO_NFFA, caption="NFFA-DI")
    st.image(LOGO_LADE, caption="AREA Science Park")

st.title("🔬 FAIR NeXus Assistant — Step 1 + Mapping (beta)")
st.caption("Upload → preview → extract metadata → instrument/experiment via LLM → select NXDL → map keys to NX paths with caching.")

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
    st.session_state.meta[k] = st.text_input(k.replace("_", " ").title(), st.session_state.meta[k], key=f"meta_{k}")

meta = st.session_state.meta

# ==============================
# Workflow: Upload image + metadata
# ==============================

if workflow == "Upload image + metadata":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")
    header = st.file_uploader("(Optional) Upload separate metadata (JSON)", type="json", key="json_uploader")

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
                raw_inst = query_ollama(build_instrument_detection_prompt(pretty_metadata), model=selected_model)
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

            # Pick NX app and load NXDL allowed paths
            definition = EXPERIMENT_TO_NX.get(results["experiment_type"], "NXem")
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
                st.code("\n".join(aps[:60]), language="text")  # sample preview only

        # ==============================
        # 🧭 Map keys → NeXus paths  (with DB cache)
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
            # Try cache first
            cached = _mapping_db_get(instrument_name, definition)
            if cached and isinstance(cached.get("mapping"), dict):
                cached_map = cached["mapping"]
                # validate against current allowed paths
                valid_cached = {k: v for k, v in cached_map.items() if v in aps}
                invalid_cached = {k: v for k, v in cached_map.items() if v not in aps}

                st.success(f"Loaded mapping from local DB (mapping_db.json) for **{instrument_name} | {definition}**")
                st.caption(f"Updated at: {cached.get('updated_at', '—')}")
                st.subheader("✅ Mapping (from DB)")
                st.json(valid_cached or {})

                if invalid_cached:
                    st.subheader("⚠️ Cached entries not valid for this NXDL (ignored)")
                    st.json(invalid_cached)

                # set session for downstream use (if any later step needs it)
                st.session_state.bulk_mapping_accepted = valid_cached
                st.session_state.bulk_mapping_rejected = {}
                st.session_state.bulk_mapping_rawlog = {}

                # Allow user to force a recompute
                if not st.button("♻️ Re-map with LLM (ignore cache)"):
                    st.stop()
                else:
                    st.info("Ignoring cache and re-mapping with LLM...")

            # No cache or user chose to recompute
            meta_keys = flatten_metadata_keys(safe_md)

            st.caption("Choose how many keys to include (in order).")
            max_keys = st.slider(
                "Number of keys to map",
                min_value=1,
                max_value=min(300, len(meta_keys)),
                value=min(50, len(meta_keys))
            )
            keys_to_map = meta_keys[:max_keys]

            st.markdown("**Keys selected:**")
            st.code("\n".join(keys_to_map[:80]), language="text")

            use_time_synonyms = st.checkbox("Bias time/date synonyms for time-like keys", value=True)
            st.caption(f"Per-key candidate paths (fixed): {PER_KEY_CANDIDATE_LIMIT}")

            if st.button("🤖 Map ALL selected keys"):
                accepted = {}
                rejected = {}
                raw_log = {}

                md_snippet = pretty_md if len(pretty_md) < 3000 else pretty_md[:3000] + "\n…(truncated)…"

                progress = st.progress(0)
                status = st.empty()

                for idx, key in enumerate(keys_to_map, start=1):
                    status.write(f"Processing {idx}/{len(keys_to_map)}: **{key}**")
                    value_preview = get_value_preview(safe_md, key)

                    # Build hints from key tokens
                    hints = set()
                    for token in re.split(r"[^a-z0-9]+", key.lower()):
                        if token and len(token) >= 2:
                            hints.add(token)

                    # Broaden hints so relevant fields appear
                    name_like = {"name", "title", "label", "model", "manufacturer", "vendor", "make", "artist", "user", "operator", "software", "program", "version"}
                    image_like = {"image", "width", "height", "length", "rows", "cols", "pixels", "pixel", "bits", "bit", "sample", "samples", "compression", "photometric", "planar", "orientation", "strip", "bytecount"}
                    detector_like = {"detector", "camera", "data", "axis", "axes", "x", "y", "z"}
                    beam_like = {"voltage", "current", "emission", "source", "beam", "extraction", "filament", "hv"}
                    stage_like = {"stage", "tilt", "rotation", "x", "y", "z"}
                    time_like  = {"time", "date", "timestamp", "datetime", "start", "end", "acquisition", "collection"}

                    hints |= name_like | image_like | detector_like | beam_like | stage_like
                    if use_time_synonyms and any(t in key.lower() for t in time_like):
                        hints |= time_like

                    allowed_subset = prioritize_paths(aps, list(hints), max_paths=PER_KEY_CANDIDATE_LIMIT)

                    prompt = build_single_key_mapping_prompt(
                        definition=definition,
                        selected_key=key,
                        value_preview=value_preview,
                        allowed_subset=allowed_subset,
                        metadata_context_snippet=md_snippet,
                    )

                    try:
                        resp = query_ollama(prompt, model=selected_model)
                        raw_log[key] = resp

                        # Parse JSON
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

                        # Validate
                        if proposed_path and proposed_path in aps:
                            # extra guard: non-time key must not map to time-like path
                            k_is_time = any(t in key.lower() for t in ["time", "date", "timestamp", "datetime", "start", "end"])
                            v_is_time = _looks_time_path(proposed_path)
                            if (not k_is_time) and v_is_time:
                                rejected[key] = proposed_path
                            else:
                                accepted[key] = proposed_path
                        else:
                            rejected[key] = proposed_path  # "" or invalid path

                    except Exception as e:
                        raw_log[key] = f"ERROR: {e}"
                        rejected[key] = ""

                    progress.progress(idx / len(keys_to_map))

                status.empty()

                # Show results
                st.subheader("✅ Accepted (valid paths)")
                st.json(accepted or {})

                st.subheader("⚠️ Rejected or empty (review needed)")
                st.json(rejected or {})

                with st.expander("LLM raw responses per key (debug)"):
                    st.code(json.dumps(raw_log, indent=2), language="json")

                # Save to session
                st.session_state.bulk_mapping_accepted = accepted
                st.session_state.bulk_mapping_rejected = rejected
                st.session_state.bulk_mapping_rawlog = raw_log

                # Save to DB (only if mapping doesn't look suspicious)
                if accepted:
                    is_bad, reason = _suspicious_collapse(accepted)
                    if is_bad:
                        st.warning(f"Mapping looks suspicious; not saving to DB. Reason: {reason}")
                    else:
                        _mapping_db_put(instrument_name, definition, accepted)
                        st.success(f"Saved mapping to DB for **{instrument_name} | {definition}** → mapping_db.json")

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

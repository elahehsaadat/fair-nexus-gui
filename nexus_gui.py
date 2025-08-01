import streamlit as st
import json
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
from nexusformat.nexus import nxload

from upload_utils import (
    get_ollama_models,
    query_ollama,
    read_image_any_format,
    convert_to_preview,
    extract_metadata_from_tiff,
    extract_metadata_from_json_header,
    build_standard_metadata,
    make_json_safe,
    extract_json_from_response,
)

from nexus_generator import generate_nexus_file, validate_nexus_file
from mapping_store import get_mapping, save_mapping

# Prompt helpers
def build_instrument_detection_prompt(pretty_metadata):
    return f"""
You are a FAIRmat metadata assistant.

Given the following metadata:
{pretty_metadata}

Your task is to identify the instrument name or model used to generate this data.

Respond ONLY with a JSON object like:
{{
  "instrument": "FEI Nova NanoSEM 650"
}}

If unknown, respond with:
{{
  "instrument": "unknown"
}}
"""

def build_definition_selection_prompt(pretty_metadata, instrument_name):
    return f"""
You are a FAIRmat metadata assistant.

Given metadata from the instrument "{instrument_name}":
{pretty_metadata}

Choose the most appropriate NeXus application definition from the following list:
- NXxas, NXxps, NXarpes, NXellipsometry, NXraman, NXem, NXtransport, NXafm

Respond only with:
{{
  "definition": "NXem"
}}

If unsure, guess the most likely based on the instrument name and metadata.
"""

def build_mapping_prompt(pretty_metadata, definition):
    return f"""
You are a FAIRmat metadata assistant.

Given the metadata:
{pretty_metadata}

And the selected NeXus application definition: {definition}

Your task:
- Map as many metadata fields as possible (including nested keys) to NeXus ontology paths.
- Try to infer meaning from context, units, or typical key names.

Respond only with valid JSON like:
{{
  "mapping": {{
    "Voltage": "NXinstrument/acceleration_voltage",
    "acquisition.time": "NXinstrument/duration",
    "detector.model": "NXinstrument/NXdetector/model"
  }}
}}

Use dot notation for nested fields if necessary.

If unsure, leave mapping empty:
{{
  "mapping": {{}}
}}

⚠️ No markdown, no comments. Only JSON.
"""

def build_missing_check_prompt(pretty_metadata, definition):
    return f"""
You are a FAIRmat metadata assistant.

Given the following metadata:
{pretty_metadata}

And the required ontology for the application definition '{definition}' (as defined in FAIRmat NeXus specifications):

Return only JSON:
{{ "missing": ["field1", "field2"] }}

If nothing is missing:
{{ "missing": [] }}
"""

# Load env
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LOGO_NFFA = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/nffa.png"
LOGO_LADE = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/area.png"

st.set_page_config(page_title="FAIR NeXus File Metadata Assistant", layout="centered")

with st.sidebar:
    st.image(LOGO_NFFA, caption="NFFA-DI")
    st.image(LOGO_LADE, caption="AREA Science Park")

st.title("🔬 FAIR NeXus Assistant")

models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on ORFEO using `ollama run <model>`. ")
    st.stop()

selected_model = st.selectbox("LLM model served by Ollama", models)
workflow = st.radio("Choose a workflow", ["Upload image + metadata", "Reference a stack (folder / S3 URL)"])

st.subheader("📄 Basic metadata")
if "meta" not in st.session_state:
    st.session_state.meta = {
        "instrument": "",
        "sample_id": "",
        "operator": "",
    }

for key in st.session_state.meta:
    st.session_state.meta[key] = st.text_input(key.replace("_", " ").title(), st.session_state.meta[key], key=key)

meta = st.session_state.meta

if workflow == "Upload image + metadata":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")
    header = st.file_uploader("(Optional) Upload separate metadata (JSON)", type="json", key="json_uploader")

    if file:
        image = read_image_any_format(file)
        st.session_state.image_array = image
        st.session_state.preview = convert_to_preview(image)
        st.image(st.session_state.preview, caption="Preview", use_container_width=True)

        instrument_metadata = {}
        filename = getattr(file, "name", "")
        if filename.lower().endswith((".tif", ".tiff")):
            instrument_metadata = extract_metadata_from_tiff(file)

        if not instrument_metadata and header:
            instrument_metadata = extract_metadata_from_json_header(header)

        if not instrument_metadata:
            st.error("❌ No metadata found. Please upload a TIFF with metadata or a separate JSON file.")
            st.stop()

        st.subheader("Extracted Metadata")
        st.json(instrument_metadata)

        safe_metadata = make_json_safe(instrument_metadata)
        pretty_metadata = json.dumps(safe_metadata, indent=2)

        if "llm_mapping_response" not in st.session_state:
            if st.button("🤖 Analyze metadata with LLM"):
                with st.spinner("⏳ Querying LLM for instrument and mapping..."):
                    try:
                        raw_inst = query_ollama(build_instrument_detection_prompt(pretty_metadata), model=selected_model)
                        instrument_name = extract_json_from_response(raw_inst).get("instrument", "unknown")

                        raw_def = query_ollama(build_definition_selection_prompt(pretty_metadata, instrument_name), model=selected_model)
                        definition = extract_json_from_response(raw_def).get("definition", "NXmicroscopy")

                        raw_map = query_ollama(build_mapping_prompt(pretty_metadata, definition), model=selected_model)
                        mapping = extract_json_from_response(raw_map).get("mapping", {})

                        st.session_state.llm_mapping_response = {"definition": definition, "mapping": mapping}
                        st.session_state.instrument_name = instrument_name
                        st.success("✅ Mapping and definition retrieved.")
                    except Exception as e:
                        st.error(f"❌ LLM query failed: {e}")
                        st.stop()
            else:
                st.stop()

        mapping_response = st.session_state.get("llm_mapping_response", {})
        definition = mapping_response.get("definition", "NXmicroscopy")
        mapping = mapping_response.get("mapping", {})
        st.subheader("🤖 LLM Mapping")
        st.json(mapping_response)

        standard_metadata = build_standard_metadata(mapping, instrument_metadata)

        if f"checked_missing_{workflow}" not in st.session_state:
            check_missing_prompt = build_missing_check_prompt(pretty_metadata, definition)
            with st.spinner("🔍 Checking for required metadata..."):
                try:
                    raw_missing = query_ollama(check_missing_prompt, model=selected_model)
                    cleaned_missing = extract_json_from_response(raw_missing)
                    missing_response = cleaned_missing if isinstance(cleaned_missing, dict) else json.loads(cleaned_missing)
                    st.session_state.missing_fields = missing_response.get("missing", [])
                except Exception as e:
                    st.warning(f"⚠️ Failed to check for missing fields: {e}")
                    st.session_state.missing_fields = []
            st.session_state[f"checked_missing_{workflow}"] = True

        if st.session_state.get("missing_fields"):
            st.warning("⚠️ Some required metadata is missing:")
            for field in st.session_state["missing_fields"]:
                value = st.text_input(f"➕ Provide value for '{field}':", key=f"missing_{field}")
                if value:
                    standard_metadata[field] = value

        all_filled = all(standard_metadata.get(field) for field in st.session_state.get("missing_fields", []))

        if all_filled and st.button("📦 Generate NeXus File"):
            try:
                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=image,
                    fields=standard_metadata,
                    definition=definition,
                    output_path=nexus_path
                )
                st.success("✅ NeXus file created!")
                st.download_button("⬇️ Download NeXus File", nexus_path.read_bytes(), file_name="output.nxs")
                validation_msg = validate_nexus_file(nexus_path)
                st.markdown("### ✅ NeXus File Validation")
                st.code(validation_msg)

                try:
                    nx = nxload(str(nexus_path))
                    st.subheader("📁 NeXus File Tree")
                    st.text(nx.tree)
                except Exception as e:
                    st.warning(f"⚠️ Could not parse NeXus file: {e}")
            except Exception as e:
                st.error(f"❌ Failed to generate NeXus file: {e}")

st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
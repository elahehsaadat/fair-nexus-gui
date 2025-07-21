# nexus_gui.py

import streamlit as st
import json
import numpy as np
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
from nexusformat.nexus import nxload

from upload_utils import (
    extract_json_from_text,
    get_ollama_models,
    query_ollama,
    read_image_any_format,
    convert_to_preview,
    extract_metadata_from_tiff,
    extract_metadata_from_json_header,
    build_standard_metadata,
)

from nexus_generator import generate_nexus_file, validate_nexus_file
from mapping_store import get_mapping, save_mapping

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

# === SINGLE IMAGE WORKFLOW ===
if workflow == "Upload image + metadata":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")
    header = st.file_uploader("(Optional) Upload separate metadata (JSON)", type="json", key="json_uploader")

    if file:
        image = read_image_any_format(file)
        st.session_state.image_array = image
        st.session_state.preview = convert_to_preview(image)
        st.image(st.session_state.preview, caption="Preview", use_container_width=True)

        # === Metadata extraction ===
        instrument_metadata = {}
        filename = getattr(file, "name", "")
        if filename.lower().endswith((".tif", ".tiff")):
            instrument_metadata = extract_metadata_from_tiff(file)

        if not instrument_metadata and header:
            instrument_metadata = extract_metadata_from_json_header(header)

        if not instrument_metadata:
            st.error("❌ No metadata found. Please upload a TIFF with metadata or a separate JSON file.")
            st.stop()

        instrument = meta.get("instrument", "").lower().strip()
        existing_mapping = get_mapping(instrument)

        if not existing_mapping:
            if st.button("🤖 Analyze metadata with LLM"):
                prompt = f"""
You are a FAIRmat metadata assistant.

Given the following raw metadata from an instrument:
{json.dumps(instrument_metadata, indent=2)}

And the shape of the image: {image.shape}

1. Search through the FAIRmat GitHub repositories:
   - https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/applications
   - https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/contributed_definitions

2. Choose the most appropriate NeXus application definition that fits the image and metadata.
3. Match the instrument metadata keys to the standard ontology terms defined in the application definition.
4. Return a JSON with the definition and mapping only.

Respond only as:
{{
  "definition": "NXmicroscopy",
  "mapping": {{
    "instrument_key1": "nexus_key1"
  }}
}}
"""
                with st.spinner("Querying LLM for application definition and mapping..."):
                    raw = query_ollama(prompt, model=selected_model)
                    cleaned = extract_json_from_text(raw)
                try:
                    mapping_response = json.loads(cleaned)
                    definition = mapping_response.get("definition", "NXmicroscopy")
                    mapping = mapping_response.get("mapping", {})
                    save_mapping(instrument, {"definition": definition, "mapping": mapping})
                    st.success("✅ Mapping saved to DB.")
                except Exception as e:
                    st.error(f"Failed to parse LLM response: {e}")
                    st.code(raw)
                    st.stop()
            else:
                st.stop()
        else:
            st.info("📁 Loaded existing mapping from database.")
            definition = existing_mapping.get("definition", "NXmicroscopy")
            mapping = existing_mapping.get("mapping") or existing_mapping.get("fields", {})

        standard_metadata = build_standard_metadata(mapping, instrument_metadata)

        # === Missing metadata check (runs only once) ===
        if f"checked_missing_{workflow}" not in st.session_state:
            check_missing_prompt = f"""
You are a FAIRmat metadata assistant.

Given the following metadata extracted from an instrument:
{json.dumps(standard_metadata, indent=2)}

And the required ontology for the application definition '{definition}' (as defined in FAIRmat NeXus specifications):

1. Compare the provided metadata against the required fields in the application definition.
2. Return a JSON array of any **missing required fields**.

Only respond with JSON. For example:

If fields are missing:
{{
  "missing": ["instrument", "operator", "pixel_size"]
}}

If nothing is missing:
{{
  "missing": []
}}
"""
            with st.spinner("Analyzing required metadata via LLM..."):
                raw_missing = query_ollama(check_missing_prompt, model=selected_model)
                cleaned_missing = extract_json_from_text(raw_missing)
            try:
                missing_response = json.loads(cleaned_missing)
                st.session_state.missing_fields = missing_response.get("missing", [])
                st.session_state[f"checked_missing_{workflow}"] = True
            except Exception as e:
                st.warning(f"⚠️ Failed to check for missing fields: {e}")
                st.session_state.missing_fields = []
                st.session_state[f"checked_missing_{workflow}"] = True

        # === Ask user for missing fields ===
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
                    st.warning(f"Could not parse NeXus file: {e}")
            except Exception as e:
                st.error(f"Failed to generate NeXus file: {e}")

# === STACK IMAGE WORKFLOW ===
elif workflow == "Reference a stack (folder / S3 URL)":
    stack_path = st.text_input("📁 Enter local folder path or MinIO S3 bucket URI:")

    if st.button("🔍 Load Image Stack") and stack_path:
        try:
            if stack_path.startswith("s3://"):
                st.warning("⚠️ MinIO support not implemented yet. Please use a local path.")
                st.stop()
            files = sorted(Path(stack_path).glob("*.tif"))
            stack = np.stack([read_image_any_format(f) for f in files])
            image = stack
            st.image(convert_to_preview(stack[0]), caption="Preview - First Frame", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed to load image stack: {e}")
            st.stop()

        instrument_metadata = extract_metadata_from_tiff(files[0]) if files else {}

        if not instrument_metadata:
            st.warning("⚠️ No metadata found in stack images.")
            st.stop()

        instrument = meta.get("instrument", "").lower().strip()
        existing_mapping = get_mapping(instrument)

        if not existing_mapping:
            prompt = f"""
You are a FAIRmat metadata assistant.

Given the following raw metadata from a stack image:
{json.dumps(instrument_metadata, indent=2)}

And stack shape: {stack.shape}

1. Identify the best NeXus application definition from FAIRmat GitHub.
2. Map the metadata keys to the ontology of that application.

Return:
{{
  "definition": "NXmicroscopy",
  "mapping": {{ "original_key": "nexus_key" }}
}}
"""
            with st.spinner("Querying LLM for application definition and mapping..."):
                raw = query_ollama(prompt, model=selected_model)
                cleaned = extract_json_from_text(raw)
            try:
                mapping_response = json.loads(cleaned)
                definition = mapping_response.get("definition", "NXmicroscopy")
                mapping = mapping_response.get("mapping", {})
                save_mapping(instrument, {"definition": definition, "mapping": mapping})
                st.success("✅ Mapping saved to DB.")
            except Exception as e:
                st.error(f"Failed to parse LLM response: {e}")
                st.code(raw)
                st.stop()
        else:
            st.info("📁 Loaded existing mapping from database.")
            definition = existing_mapping.get("definition", "NXmicroscopy")
            mapping = existing_mapping.get("mapping") or existing_mapping.get("fields", {})

        standard_metadata = build_standard_metadata(mapping, instrument_metadata)

        # === Missing metadata check (runs only once) ===
        if "checked_missing" not in st.session_state:
            check_missing_prompt = f"""
You are a FAIRmat metadata assistant.

Given the following metadata extracted from an instrument:
{json.dumps(standard_metadata, indent=2)}

And the required ontology for the application definition '{definition}' (as defined in FAIRmat NeXus specifications):

1. Compare the provided metadata against the required fields in the application definition.
2. Return a JSON array of any **missing required fields**.

Only respond with JSON. For example:

If fields are missing:
{{
  "missing": ["instrument", "operator", "pixel_size"]
}}

If nothing is missing:
{{
  "missing": []
}}
"""
            with st.spinner("Checking for missing metadata via LLM..."):
                raw_missing = query_ollama(check_missing_prompt, model=selected_model)
                cleaned_missing = extract_json_from_text(raw_missing)
            try:
                missing_response = json.loads(cleaned_missing)
                st.session_state.missing_fields = missing_response.get("missing", [])
                st.session_state.checked_missing = True
            except Exception as e:
                st.warning(f"⚠️ Failed to check for missing fields: {e}")
                st.session_state.missing_fields = []
                st.session_state.checked_missing = True

        # === Ask user for missing fields ===
        if st.session_state.get("missing_fields"):
            st.warning("⚠️ Some required metadata is missing:")
            for field in st.session_state["missing_fields"]:
                value = st.text_input(f"➕ Provide value for '{field}':", key=f"missing_{field}")
                if value:
                    standard_metadata[field] = value

        all_filled = all(standard_metadata.get(field) for field in st.session_state.get("missing_fields", []))

        if all_filled and st.button("📦 Generate NeXus File from Stack"):
            try:
                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=stack,
                    fields=standard_metadata,
                    definition=definition,
                    output_path=nexus_path
                )
                st.success("✅ NeXus file for stack created!")
                st.download_button("⬇️ Download NeXus File", nexus_path.read_bytes(), file_name="stack_output.nxs")
                validation_msg = validate_nexus_file(nexus_path)
                st.markdown("### ✅ NeXus File Validation")
                st.code(validation_msg)
                try:
                    nx = nxload(str(nexus_path))
                    st.subheader("📁 NeXus File Tree")
                    st.text(nx.tree)
                except Exception as e:
                    st.warning(f"Could not parse NeXus file: {e}")
            except Exception as e:
                st.error(f"Failed to generate NeXus file: {e}")

# === Reset Button ===
st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


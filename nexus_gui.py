# nexus_gui.py

import streamlit as st
import json
import numpy as np
from PIL import Image
import tifffile
import requests
import tempfile
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import re
import subprocess
from nexusformat.nexus import nxload

from upload_utils import (
    extract_json_from_text,
    get_ollama_models,
    query_ollama,
    read_image_any_format,
    convert_to_preview,
)

from nexus_generator import generate_nexus_file  # Your local generator
from nexus_generator import validate_nexus_file

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


# === UI Logic ===

models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on ORFEO using `ollama run <model>`. ")
    st.stop()

selected_model = st.selectbox("LLM model served by Ollama", models)
workflow = st.radio("Choose a workflow", ["Upload one image", "Reference a stack (folder / S3 URL)"], key="workflow")

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
if workflow == "Upload one image":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")
    if file:
        image = read_image_any_format(file)
        st.session_state.image_array = image
        st.session_state.preview = convert_to_preview(image)
        st.image(st.session_state.preview, caption="Preview", use_container_width=True)

    if st.button("🔍 Analyse image and create NeXus"):
        image_array = st.session_state.get("image_array")
        if image_array is None:
            st.error("Please upload an image first.")
            st.stop()

        instrument = meta.get("instrument", "").lower().strip()
        existing = get_mapping(instrument)

        if existing:
            st.info("📁 Using saved mapping from database.")
            try:
                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=image_array,  # or stack if in stack workflow
                    fields=existing["fields"],
                    definition=existing["definition"],
                    output_path=nexus_path
                )
                st.success("✅ NeXus file created from saved mapping.")
                st.download_button("⬇️ Download NeXus File", nexus_path.read_bytes(), file_name="output.nxs")
                validation_msg = validate_nexus_file(nexus_path)
                st.markdown("### ✅ NeXus File Validation")
                st.code(validation_msg)
            except Exception as e:
                st.error(f"Failed to generate NeXus file: {e}")
            st.stop()

        # Fallback to LLM if no mapping exists
        prompt = f"""
You are an expert on NeXus data standards and microscopy metadata.

Your tasks:
1. Guess the correct NeXus application definition based on:
   - Image shape: {image_array.shape}
   - User-provided metadata: {json.dumps(meta, indent=2)}
2. Use FAIRmat ontologies and NeXus-compliant fields to construct a valid JSON response.

Important rules:
- Only include fields that are non-empty or can be inferred.
- Any required fields that are missing or empty (e.g. "instrument": "") must be listed under "missing".
- Do not include explanation, markdown, or comments.

Return only one of the following formats:

1. If sufficient fields:
{{
  "definition": "NXmicroscopy",
  "fields": {{
    "sample_id": "S123",
    "image_size_x": 1183,
    "image_size_y": 1024,
    "pixel_size": 0.65,
    "units": "photons"
  }}
}}

2. If fields are missing:
{{
  "missing": ["instrument", "pixel_size", "operator"]
}}
"""


        with st.spinner("Querying LLM..."):
            raw = query_ollama(prompt, model=selected_model)
            cleaned = extract_json_from_text(raw)

        try:
            response_json = json.loads(cleaned)
        except json.JSONDecodeError:
            st.error("LLM did not return valid JSON:")
            st.code(raw)
            st.stop()

        if "missing" in response_json:
            st.session_state.missing_fields = response_json["missing"]
        elif "definition" in response_json and "fields" in response_json:
            # Save the new mapping
            save_mapping(instrument, {"definition": response_json["definition"],"fields": response_json["fields"]})

            try:
                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=image_array,
                    fields=response_json["fields"],
                    definition=response_json["definition"],
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
        else:
            st.warning("Unexpected response:")
            st.code(cleaned)

# === STACK WORKFLOW ===
elif workflow == "Reference a stack (folder / S3 URL)":
    path = st.text_input("📂 Folder or S3 path", key="stack_path")
    if st.button("🚀 Analyse stack"):
        try:
            files = sorted(Path(path).glob("*.tif"))
            stack = np.stack([read_image_any_format(f) for f in files])
            preview = convert_to_preview(stack[0])
            st.image(preview, caption="First frame", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load image stack: {e}")
            st.stop()

        instrument = meta.get("instrument", "").lower().strip()
        existing = get_mapping(instrument)

        if existing:
            st.info("📁 Using saved mapping from database.")
            try:
                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=image_array,  # or stack if in stack workflow
                    fields=existing["fields"],
                    definition=existing["definition"],
                    output_path=nexus_path
                )
                st.success("✅ NeXus file created from saved mapping.")
                st.download_button("⬇️ Download NeXus File", nexus_path.read_bytes(), file_name="output.nxs")
                validation_msg = validate_nexus_file(nexus_path)
                st.markdown("### ✅ NeXus File Validation")
                st.code(validation_msg)
            except Exception as e:
                st.error(f"Failed to generate NeXus file: {e}")
            st.stop()

        # Fallback to LLM if no mapping exists
        prompt = f"""
You are an expert in NeXus data formats and FAIRmat ontologies.

A user has provided a microscopy image **stack** stored at `{path}`.
Your tasks:

1. Guess the correct NeXus application definition based on:
   - Image stack shape: {stack.shape}
   - Metadata: {json.dumps(meta, indent=2)}

2. Extract relevant metadata using FAIRmat/NeXus ontologies:
   - Use only non-empty, meaningful fields
   - Represent all metadata with valid NeXus field names and types
   - If you infer a value (like number of frames or voxel size), include it

3. Return structured JSON. Follow one of these formats strictly:

**A. If metadata is complete:**
{{
  "definition": "NXmicroscopy",
  "fields": {{
    "sample_id": "ABC123",
    "instrument": "confocal microscope",
    "num_frames": {stack.shape[0]},
    "image_size_x": {stack.shape[-1]},
    "image_size_y": {stack.shape[-2]},
    "pixel_size": 0.2,
    "units": "intensity"
  }}
}}

**B. If some metadata is missing or empty:**
{{
  "missing": ["instrument", "pixel_size", "acquisition_date"]
}}

Rules:
- Do NOT include empty strings in "fields"
- Do NOT return explanation, markdown, or natural language
- Always return VALID JSON
"""


        with st.spinner("Querying LLM..."):
            raw = query_ollama(prompt, model=selected_model)
            cleaned = extract_json_from_text(raw)

        try:
            response_json = json.loads(cleaned)
            if "missing" in response_json:
                st.session_state.missing_fields = response_json["missing"]
            elif "definition" in response_json and "fields" in response_json:
                # Save the new mapping
                save_mapping(instrument, {"definition": response_json["definition"],"fields": response_json["fields"]})

                nexus_path = Path(tempfile.mkstemp(suffix=".nxs")[1])
                generate_nexus_file(
                    image_array=stack,
                    fields=response_json["fields"],
                    definition=response_json["definition"],
                    output_path=nexus_path
                )
                st.success("✅ NeXus file created!")
                st.download_button("⬇️ Download NeXus File", nexus_path.read_bytes(), file_name="stack.nxs")
                validation_msg = validate_nexus_file(nexus_path)
                st.markdown("### ✅ NeXus File Validation")
                st.code(validation_msg)
            else:
                st.error("Unexpected LLM response:")
                st.code(cleaned)
        except Exception as e:
            st.error(f"LLM did not return valid JSON. Full text:\n{raw}")

# === Missing Fields Handler ===
if "missing_fields" in st.session_state and st.session_state.missing_fields:
    st.warning("🚧 Missing required metadata:")
    ready = True
    for f in st.session_state.missing_fields:
        value = st.text_input(f"➕ Provide {f}", key=f"missing_{f}")
        st.session_state.meta[f] = value
        if not value.strip():
            ready = False
    
    st.success("✅ Re-run the analysis to continue with the new metadata.")

# === Reset Button ===
st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


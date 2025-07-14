import streamlit as st
import json
import numpy as np
from PIL import Image
import tifffile
import requests
import tempfile
import os
import base64
from nexusformat.nexus import nxload
from dotenv import load_dotenv
import re
from pathlib import Path
import subprocess

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LOGO_NFFA = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/nffa.png"
LOGO_LADE = "https://raw.githubusercontent.com/elahehsaadat/fair-nexus-gui/main/assets/area.png"

st.set_page_config(page_title="FAIR NeXus File Metadata Assistant", layout="centered")

with st.sidebar:
    st.image(LOGO_NFFA, caption="NFFA-DI")
    st.image(LOGO_LADE, caption="AREA Science Park")

st.title("🔬 FAIR NeXus File Metadata Assistant")


def extract_json_from_text(text):
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    return match.group(1) if match else text


@st.cache_data(show_spinner=False)
def get_ollama_models(host=OLLAMA_BASE_URL):
    try:
        res = requests.get(f"{host}/api/tags")
        if res.status_code == 200:
            return [m["name"] for m in res.json().get("models", [])]
    except Exception:
        pass
    return []


def query_ollama(prompt, model, host=OLLAMA_BASE_URL):
    try:
        res = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True
        )
        if res.status_code != 200:
            return f'{{"error": "Status {res.status_code}: {res.text}"}}'

        output = ""
        for line in res.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    output += chunk.get("response", "")
                except Exception:
                    pass
        return output.strip()
    except Exception as e:
        return f'{{"error": "Ollama request failed: {str(e)}"}}'


def read_image_any_format(file) -> np.ndarray:
    try:
        return tifffile.imread(file)
    except Exception:
        return np.array(Image.open(file))


def convert_to_preview(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if np.ptp(arr) == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - np.min(arr)) / np.ptp(arr) * 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def safe_b64decode(b64_string):
    b64_string = re.sub(r'[^A-Za-z0-9+/=]', '', b64_string)
    padded = b64_string + "=" * (-len(b64_string) % 4)
    try:
        return base64.b64decode(padded)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {e}")


# === Main UI logic ===
models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on ORFEO using `ollama run <model>`. ")
    st.stop()

selected_model = st.selectbox("LLM model served by Ollama", models)
workflow = st.radio("Choose a workflow", ["Upload one image", "Reference a stack (folder / S3 URL)"], key="workflow")

st.subheader("📄 Basic metadata")
if "meta" not in st.session_state:
    st.session_state.meta = {
        "instrument": "confocal microscope",
        "sample_id": "",
        "operator": "",
    }

for key in st.session_state.meta:
    st.session_state.meta[key] = st.text_input(key.replace("_", " ").title(), st.session_state.meta[key], key=key)

meta = st.session_state.meta

# === Single image workflow ===
if workflow == "Upload one image":
    file = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="file_uploader")

    if file and "image_array" not in st.session_state:
        st.session_state.image_array = read_image_any_format(file)
        st.session_state.preview = convert_to_preview(st.session_state.image_array)

    if "preview" in st.session_state:
        st.image(st.session_state.preview, caption="Preview", use_container_width=True)

    if st.button("🔍 Analyse image in LLM"):
        if "image_array" not in st.session_state:
            st.error("⚠️ Please upload an image before analysing.")
            st.stop()

        prompt = f"""
You are an expert AI that builds FAIR-compliant NeXus (.nxs) files for microscopy data.

Here is the experimental context:
- Number of images: 1
- Image shape: {st.session_state.image_array.shape}
- Metadata:
{json.dumps(meta, indent=2)}

Find the most appropriate NeXus application definition from:
- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/applications
- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/contributed_definitions

Respond in **valid JSON only** with one of the following:
1. If metadata is missing:
{{"missing": ["field1", "field2"]}}
2. If all required fields are present and the image is valid:
{{"nexus_b64": "<base64-encoded-nexus-file>"}}
NO explanation. NO markdown.
"""
        with st.spinner("Querying LLM via Ollama..."):
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
        elif "nexus_b64" in response_json:
            try:
                decoded = safe_b64decode(response_json["nexus_b64"])
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nxs") as tmp:
                    tmp.write(decoded)
                    st.download_button("⬇️ Download NeXus File", tmp.read(), "output.nxs")
                    try:
                        nx = nxload(tmp.name)
                        st.subheader("📁 NeXus File Tree")
                        st.text(nx.tree)
                    except Exception as e:
                        st.warning(f"Could not parse NeXus file: {e}")
            except Exception as e:
                st.error(f"❌ Failed to decode or validate NeXus file: {e}")
        else:
            st.warning("Unexpected LLM response:")
            st.code(cleaned)

# === Stack workflow ===
elif workflow == "Reference a stack (folder / S3 URL)":
    path = st.text_input("📂 Folder or S3 path", key="stack_path")
    if st.button("🚀 Analyse stack in LLM") and path:
        try:
            first_file = subprocess.check_output(["ls", "-1", path], text=True).splitlines()[0]
            img = read_image_any_format(Path(path) / first_file)
            first_frame = convert_to_preview(img)
            st.image(first_frame, caption="First Frame", use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed: {e}")

        prompt = f"""
You are an expert in NeXus data formats.

A **stack** of microscopy images is stored at `{path}`.
Assume the stack shape is (N, Y, X).
User metadata:
{json.dumps(meta, indent=2)}

Same TASK as before (pick definition, list missing, OR return nexus_b64)
but the NeXus file must contain the whole *stack* (not just one frame).
If you cannot access the stack pixel data, create dummy zeros with the correct dimensions.

Answer only the JSON object described.
"""
        with st.spinner("Querying LLM..."):
            raw = query_ollama(prompt, model=selected_model)
            cleaned = extract_json_from_text(raw)

        try:
            out = json.loads(cleaned)
            if "missing" in out:
                st.session_state.missing_fields = out["missing"]
            elif "nexus_b64" in out:
                tmp = Path(tempfile.mkstemp(suffix=".nxs")[1])
                tmp.write_bytes(safe_b64decode(out["nexus_b64"]))
                st.success("✅ NeXus file created!")
                st.download_button("⬇️ Download .nxs", tmp.read_bytes(), "stack.nxs")
            else:
                st.error("Unexpected JSON response:")
                st.code(out)
        except Exception as e:
            st.error(f"❌ LLM did not return JSON. Full text:\n{raw}")

# === Missing fields handler ===
if "missing_fields" in st.session_state and st.session_state.missing_fields:
    st.warning("🚧 Missing required metadata:")
    ready = True
    for f in st.session_state.missing_fields:
        input_key = f"missing_{f}"
        value = st.text_input(f"➕ Provide **{f}**", value=st.session_state.meta.get(f, ""), key=input_key)
        st.session_state.meta[f] = value
        if value.strip() == "":
            ready = False
    if ready:
        st.success("✅ After filling the missing fields, please press 'Analyse image in LLM' again to continue.")

# === Global reset button (always last and always visible) ===
st.markdown("---")
if st.button("🔄 Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

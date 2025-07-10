import streamlit as st
import json
import numpy as np
from PIL import Image, ImageOps
import requests
import tempfile
import os
import base64
from nexusformat.nexus import nxload
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
            return f"❌ Error {res.status_code}: {res.text}"

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
        return f"❌ Failed to connect to Ollama: {e}"

def summarize_image_array(arr):
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr))
    }

def convert_16bit_to_preview(arr_16bit):
    arr = (arr_16bit - np.min(arr_16bit)) / (np.ptp(arr_16bit) + 1e-8) * 255
    return arr.astype(np.uint8)

st.set_page_config(page_title="FAIR NeXus Metadata Assistant", layout="centered")
st.title("🔬 FAIR NeXus File Metadata Assistant")

st.subheader("🤖 Select LLM Model (from Ollama)")
models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on ORFEO using `ollama run <model>`.")
    st.stop()

selected_model = st.selectbox("Choose model", models)

# Accept multiple images
image_files = st.file_uploader("Upload microscope image(s)", type=["tif", "tiff", "png", "jpg"], accept_multiple_files=True)

st.subheader("📄 Metadata")
metadata = {
    "instrument": st.text_input("Instrument", "confocal microscope"),
    "sample_id": st.text_input("Sample ID"),
    "timestamp": st.text_input("Timestamp (ISO)", ""),
    "operator": st.text_input("Operator", ""),
    "units": st.text_input("Units", "intensity"),
}

if st.button("🧠 Send to LLM (Ollama on ORFEO)"):
    if not image_files:
        st.error("Please upload at least one image.")
        st.stop()

    try:
        arrays = []
        for f in image_files:
            img = Image.open(f).convert("L")
            arrays.append(np.array(img))
        stack = np.stack(arrays, axis=0)
        image_stats = summarize_image_array(stack)
        stack_shape = stack.shape

        preview = convert_16bit_to_preview(stack[0]) if stack.dtype == np.uint16 else stack[0]
        st.image(preview, caption="Preview of First Image", use_container_width=True)

        prompt = f"""
You are an AI agent that creates FAIR-compliant NeXus (.nxs) files.

Dataset summary:
- Number of images: {len(image_files)}
- Stack shape: {stack_shape}
- Metadata (JSON):
{json.dumps(metadata, indent=2)}

Use these NeXus definitions to find the most appropriate structure:
- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/applications
- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/contributed_definitions

Your task:
1. Choose one NeXus application definition appropriate for this microscopy dataset.
2. List the required fields.
3. If any required fields are missing, return only:
{{"missing": ["field1", "field2"]}}
4. If everything is provided, return:
{{"nexus_b64": "<base64-encoded-NX-file>"}}

The NeXus file should:
- Follow NeXus hierarchy strictly
- Use NXdata to store the image stack (compressed)
- Save it using gzip and return as base64-encoded binary
"""

        with st.spinner(f"Querying `{selected_model}` via Ollama..."):
            raw_response = query_ollama(prompt, model=selected_model)

        try:
            response_json = json.loads(raw_response)
        except Exception:
            st.error("❌ LLM response is not valid JSON.")
            st.code(raw_response)
            st.stop()

        if "missing" in response_json:
            st.warning("⚠️ Missing metadata fields required by NeXus:")
            for field in response_json["missing"]:
                metadata[field] = st.text_input(f"{field} (required):", "")
            if st.button("🔁 Retry with completed metadata"):
                st.rerun()

        elif "nexus_b64" in response_json:
            decoded = base64.b64decode(response_json["nexus_b64"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nxs") as tmp:
                tmp.write(decoded)
                tmp.flush()
                with open(tmp.name, "rb") as f:
                    st.download_button("⬇️ Download NeXus File", f, file_name="output.nxs")
                try:
                    nx = nxload(tmp.name)
                    st.subheader("📁 NeXus File Tree")
                    st.text(nx.tree)
                except Exception as e:
                    st.warning(f"Could not parse NeXus file: {e}")
        else:
            st.warning("❓ Unexpected response format.")
            st.code(raw_response)

    except Exception as e:
        st.error(f"❌ Failed to process image or LLM request: {e}")

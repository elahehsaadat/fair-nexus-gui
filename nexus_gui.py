import streamlit as st
import json
import numpy as np
from PIL import Image, ImageOps
import requests
import tempfile
import os
import contextlib
from nexusformat.nexus import nxload

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

def convert_16bit_to_preview(image, arr_16bit):
    arr = (arr_16bit - np.min(arr_16bit)) / (np.ptp(arr_16bit) + 1e-8) * 255
    arr_8bit = arr.astype(np.uint8)
    return Image.fromarray(arr_8bit, mode="L")

st.set_page_config(page_title="FAIR NeXus Metadata Assistant", layout="centered")
st.title("🔬 FAIR NeXus File Metadata Assistant")

st.subheader("🤖 Select LLM Model (from Ollama)")
models = get_ollama_models()
if not models:
    st.error("⚠️ No models found. Start one on ORFEO using `ollama run <model>`.")
    st.stop()

selected_model = st.selectbox("Choose model", models)

image_file = st.file_uploader("Upload microscope image (.tif, .png, .jpg)", type=["tif", "tiff", "png", "jpg"])

st.subheader("📄 Metadata")
metadata = {
    "instrument": st.text_input("Instrument", "confocal microscope"),
    "sample_id": st.text_input("Sample ID"),
    "timestamp": st.text_input("Timestamp (ISO)", ""),
    "operator": st.text_input("Operator", ""),
    "units": st.text_input("Units", "intensity"),
}

if st.button("🧠 Send to LLM (Ollama on ORFEO)"):
    if not image_file:
        st.error("Please upload an image.")
    else:
        try:
            original_img = Image.open(image_file)
            image_array = np.array(original_img)
            image_stats = summarize_image_array(image_array)

            try:
                if original_img.mode == "I;16":
                    preview_img = convert_16bit_to_preview(original_img, image_array)
                else:
                    preview_img = ImageOps.autocontrast(original_img.convert("L"))
                st.image(preview_img, caption="Microscope Image Preview", use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Image preview failed: {e}")

            prompt = f"""
You are an AI agent that helps generate FAIR-compliant NeXus files.

A microscopy dataset has been uploaded. Based on the metadata and data summary below, determine the most appropriate NeXus application definition from the FAIRmat NeXus repositories:

- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/applications
- https://github.com/FAIRmat-NFDI/nexus_definitions/tree/fairmat/contributed_definitions

Your task:
1. Identify and state the most suitable NeXus application definition for this data
2. Justify your choice (1–2 sentences)
3. Generate valid Python code using the `nexusformat` library to create the NeXus structure according to the selected definition
4. Save the file to the predefined variable `output_path`

User-provided metadata (JSON):
{json.dumps(metadata, indent=2)}

Image data summary (JSON):
{json.dumps(image_stats, indent=2)}

Important:
- Use correct NeXus group names and hierarchy
- Include key metadata fields (sample ID, operator, instrument)
- Use `NXdata` for image array (shape may be dummy)
- Save using `nxroot.save(output_path)`
"""

            with st.spinner(f"Querying `{selected_model}` via Ollama..."):
                response = query_ollama(prompt, model=selected_model)
                st.success("✅ Response from LLM:")
                st.code(response, language="python")

                if st.checkbox("⚠️ Run the code above to generate a NeXus (.nxs) file?"):
                    try:
                        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".nxs")
                        tmp_output.close()
                        output_path = tmp_output.name
                        local_vars = {"output_path": output_path}

                        with contextlib.redirect_stdout(None):
                            exec(response, {}, local_vars)

                        with open(output_path, "rb") as f:
                            st.download_button("⬇️ Download NeXus File", f, file_name="output.nxs")

                        try:
                            nxfile = nxload(output_path)
                            st.subheader("📁 NeXus File Tree")
                            st.text(nxfile.tree)
                        except Exception as e:
                            st.warning(f"Could not preview NeXus file: {e}")

                    except Exception as e:
                        st.error(f"❌ Code execution failed: {e}")
        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")

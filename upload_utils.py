# upload_utils.py

import requests
import tifffile
import numpy as np
from io import BytesIO
import re
import streamlit as st
from PIL import Image
import os
import json
import paramiko
from urllib.parse import urlparse

# === File transfer ===
def send_file_sftp(local_path, remote_uri):
    parsed = urlparse(remote_uri)
    transport = paramiko.Transport((parsed.hostname, 22))
    transport.connect(username=parsed.username)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(local_path, parsed.path)
    sftp.close()
    transport.close()
    print(f"📤 File uploaded to {remote_uri}")

def send_file_http(local_path, url):
    with open(local_path, 'rb') as f:
        res = requests.post(url, files={'file': f})
    print(f"📤 Uploaded to {url}, status code: {res.status_code}")


# === Utilities ===
def extract_json_from_text(text):
    # Extract the first {...} or [...] block that looks like valid JSON
    match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def read_image_any_format(file) -> np.ndarray:
    # Try TIFF first
    try:
        file.seek(0)
        arr = tifffile.imread(file)
        arr = np.asarray(arr)
    except Exception:
        # Fallback: PNG/JPEG etc.
        file.seek(0)
        image = Image.open(BytesIO(file.read())).convert("RGB")  # Enforce 3 channels
        arr = np.asarray(image)

    # Enforce numeric dtype
    if arr.dtype == object:
        arr = arr.astype(np.uint8)

    return arr


def convert_to_preview(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] != 3:
            raise ValueError(f"Unsupported 3D image shape: {arr.shape}")
        return arr.astype(np.uint8)

    elif arr.ndim == 2:
        arr = arr.astype(np.float32)
        return ((arr - np.min(arr)) / (np.ptp(arr) + 1e-6) * 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {arr.shape}")

def get_ollama_models(host=None):
    try:
        host = host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        res = requests.get(f"{host}/api/tags")
        if res.status_code == 200:
            return [m["name"] for m in res.json().get("models", [])]
    except Exception:
        pass
    return []

def query_ollama(prompt, model, host=None):
    try:
        host = host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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

def extract_metadata_from_tiff(file) -> dict:
    try:
        file.seek(0)  # Reset file pointer to beginning
        with tifffile.TiffFile(file) as tif:
            tags = {
                tag.name: str(tag.value)
                for tag in tif.pages[0].tags.values()
                if tag.value is not None and str(tag.value).strip() != ""
            }
        return tags
    except Exception as e:
        print("⚠️ TIFF metadata extraction failed:", e)
        return {}

def extract_metadata_from_json_header(file) -> dict:
    try:
        content = file.read()
        return json.loads(content)
    except Exception:
        return {}

def build_standard_metadata(mapping: dict, metadata: dict) -> dict:
    result = {}
    for source_key, target_key in mapping.items():
        value = metadata.get(source_key)
        if value not in (None, ""):
            result[target_key] = value
    return result

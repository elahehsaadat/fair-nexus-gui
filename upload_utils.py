# upload_utils.py

import requests
import tifffile
import numpy as np
import re
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
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    return match.group(1) if match else text

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
        return ((arr - np.min(arr)) / np.ptp(arr) * 255).astype(np.uint8)
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

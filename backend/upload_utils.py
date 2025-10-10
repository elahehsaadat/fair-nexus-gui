# upload_utils.py

import requests
from tifffile import TiffFile
import numpy as np
from io import BytesIO
import re
from PIL import Image
import os
import json
import re
import paramiko
import string
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
def fix_common_json_errors(json_str: str) -> str:
    # Add commas between properties if missing
    json_str = re.sub(r'("\s*:\s*[^,{}\[\]\n"]+)(\s*")', r'\1,\2', json_str)
    json_str = re.sub(r'(\})(\s*\{)', r'\1,\2', json_str)
    json_str = re.sub(r'(?<=\{|,)\s*(\w+)\s*:', r'"\1":', json_str)
    return json_str

def extract_json_from_response(text) -> dict:
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError(f"Expected string or dict, got {type(text)}")

    try:
        # 1. Remove markdown code fences if present
        text = re.sub(r"^```json\s*|```$", "", text, flags=re.IGNORECASE).strip()

        # 2. Clean non-printable characters (likely binary garbage)
        text = ''.join(ch for ch in text if ch in string.printable)

        # 3. Use brace matching to extract the first valid JSON block
        brace_count = 0
        json_start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if json_start is None:
                    json_start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and json_start is not None:
                    json_candidate = text[json_start:i+1]
                    break
        else:
            raise ValueError("No complete JSON object found in response.")

        # 4. Attempt to parse
        return json.loads(json_candidate)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error during JSON extraction: {e}")

def make_json_safe(obj):
    """Recursively convert non-serializable values to strings."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (int, float, str, type(None), bool)):
        return obj
    else:
        return str(obj)

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

def extract_metadata_from_tiff(uploaded_file) -> dict:

    uploaded_file.seek(0)                 # REWIND — critical!
    meta = {}

    with TiffFile(BytesIO(uploaded_file.read())) as tif:
        tags = tif.pages[0].tags
        for tag in tags.values():
            try:
                meta[tag.name] = tag.value
            except Exception:
                continue
    return meta

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

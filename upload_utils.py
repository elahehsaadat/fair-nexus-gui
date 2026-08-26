# upload_utils.py

import requests
from tifffile import TiffFile
import tifffile  # <-- needed for tifffile.imread used below
import numpy as np
from io import BytesIO
import re
from PIL import Image
import os
import json
import string

# =========================
# Helpers: parse vendor blobs (bytes/text) into key=value dict
# =========================
_PRINTABLE_KEEP = set(string.printable) | {"\n", "\r", "\t"}

def _decode_bytes_best_effort(b: bytes) -> str:
    # Try UTF-8 first, then latin1 as a fallback (never throws)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def _clean_text_for_kv(s: str) -> str:
    # Keep printable chars + whitespace; strip NULs
    s = s.replace("\x00", "\n")
    s = "".join(ch for ch in s if (ch in _PRINTABLE_KEEP))
    # Normalize line endings a bit
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

_KV_RE = re.compile(r"(?m)^\s*([A-Za-z][A-Za-z0-9_\-]*)\s*=\s*(.*)\s*$")

def _coerce_value(v: str):
    v = v.strip()
    if v == "":
        return ""
    # bool-ish
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    # int
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
    except Exception:
        pass
    # float (including scientific)
    try:
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", v):
            return float(v)
    except Exception:
        pass
    return v

def _merge_kv(dest: dict, src: dict) -> None:
    """
    Merge src into dest. If a key repeats with different values, store list of unique values.
    """
    for k, v in src.items():
        if k not in dest:
            dest[k] = v
        else:
            if dest[k] == v:
                continue
            # convert to list (unique)
            if not isinstance(dest[k], list):
                dest[k] = [dest[k]]
            if v not in dest[k]:
                dest[k].append(v)

def _parse_kv_from_text(text: str) -> dict:
    """
    Parse lines like `Key=Value` into a dict.
    Only parses if it finds enough key=value lines to be meaningful.
    """
    if not isinstance(text, str) or "=" not in text:
        return {}

    cleaned = _clean_text_for_kv(text)
    matches = list(_KV_RE.finditer(cleaned))
    if len(matches) < 3:
        return {}

    out = {}
    for m in matches:
        k = m.group(1).strip()
        v = m.group(2).strip()
        out[k] = _coerce_value(v)
    return out

def _parse_kv_from_bytes(blob: bytes) -> dict:
    if not isinstance(blob, (bytes, bytearray)) or len(blob) == 0:
        return {}
    txt = _decode_bytes_best_effort(bytes(blob))
    return _parse_kv_from_text(txt)

def _summarize_bytes(blob: bytes, max_preview: int = 200) -> str:
    txt = _clean_text_for_kv(_decode_bytes_best_effort(blob))
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return f"<bytes length={len(blob)}>"
    if len(txt) > max_preview:
        txt = txt[:max_preview] + "…"
    return f"<bytes length={len(blob)} preview='{txt}'>"

# === Utilities ===
def extract_json_from_response(text) -> dict:
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError(f"Expected string or dict, got {type(text)}")

    try:
        text = re.sub(r"^```json\s*|```$", "", text, flags=re.IGNORECASE).strip()
        text = "".join(ch for ch in text if ch in string.printable)

        brace_count = 0
        json_start = None
        json_candidate = None
        for i, ch in enumerate(text):
            if ch == "{":
                if json_start is None:
                    json_start = i
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and json_start is not None:
                    json_candidate = text[json_start : i + 1]
                    break

        if not json_candidate:
            raise ValueError("No complete JSON object found in response.")

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

def get_jwt():
    auth_url = os.getenv(
        "ORFEO_AUTH_URL",
        "https://orfeo-auth.areasciencepark.it/application/o/token/",
    )

    data = {
        "grant_type": "password",
        "client_id": os.getenv("ORFEO_CLIENT_ID", ""),
        "username": os.getenv("ORFEO_USERNAME", ""),
        "password": os.getenv("ORFEO_PASSWORD", ""),
        "scope": "profile",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    resp = requests.post(auth_url, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_ollama_models(host=None):
    try:
        host = host or os.getenv(
            "ORFEO_LLM_BASE_URL",
            "https://orfeo-llm.areasciencepark.it/vllm",
        )
        jwt_token = get_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }

        res = requests.get(f"{host}/v1/models", headers=headers, timeout=20)

        if res.status_code == 200:
            data = res.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass

    return []

def query_ollama(prompt, model, host=None):
    try:
        host = host or os.getenv(
            "ORFEO_LLM_BASE_URL",
            "https://orfeo-llm.areasciencepark.it/vllm",
        )
        jwt_token = get_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        res = requests.post(
            f"{host}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )

        if res.status_code != 200:
            return f'{{"error": "Status {res.status_code}: {res.text}"}}'

        data = res.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f'{{"error": "Orfeo LLM request failed: {str(e)}"}}'

def extract_metadata_from_tiff(uploaded_file) -> dict:
    """
    Improved TIFF metadata extraction:
    - Reads standard TIFF tags (ImageWidth, Make, Model, Software, etc.)
    - Detects vendor/private tags that contain binary/text blobs
    - Parses key=value pairs from those blobs into meta["VendorKV"]
      (e.g. HV, SpotSize, WD, StageX, Detector0Gain, ...)
    - Avoids dumping huge byte strings into JSON (stores a short summary instead)
    """
    uploaded_file.seek(0)  # REWIND — critical!
    meta: dict = {}
    vendor_kv: dict = {}
    vendor_sources: list = []

    raw = uploaded_file.read()
    with TiffFile(BytesIO(raw)) as tif:
        # Most metadata is in page 0 tags; some vendors repeat in other pages.
        pages = list(tif.pages[: min(3, len(tif.pages))])  # check first few pages
        for page_i, page in enumerate(pages):
            for tag in page.tags.values():
                try:
                    name = str(tag.name)
                    value = tag.value
                except Exception:
                    continue

                # Keep first occurrence for standard tags; later pages shouldn't overwrite unless empty
                if name not in meta or meta.get(name) in (None, "", "<empty>"):
                    # Handle bytes blobs: summarize + parse KV
                    if isinstance(value, (bytes, bytearray)):
                        blob = bytes(value)
                        kv = _parse_kv_from_bytes(blob)
                        if kv:
                            _merge_kv(vendor_kv, kv)
                            vendor_sources.append(
                                {"tag": name, "page": page_i, "kind": "bytes", "kv_count": len(kv)}
                            )

                        # TESCAN tag 50431 is a binary container. Map its parsed
                        # VendorKV fields, not the raw JPEG 2000/INI payload.
                        is_tescan_container = getattr(tag, "code", None) == 50431
                        if not (is_tescan_container and kv):
                            meta[name] = _summarize_bytes(blob)

                    # Handle text blobs that look like KV blocks (some vendors use ImageDescription etc.)
                    elif isinstance(value, str):
                        meta[name] = value
                        kv = _parse_kv_from_text(value)
                        if kv:
                            _merge_kv(vendor_kv, kv)
                            vendor_sources.append(
                                {"tag": name, "page": page_i, "kind": "text", "kv_count": len(kv)}
                            )

                    else:
                        # Normal scalar/list types
                        meta[name] = value

    if vendor_kv:
        meta["VendorKV"] = vendor_kv
        meta["VendorKV_sources"] = vendor_sources

    return meta

def extract_metadata_from_json_header(file) -> dict:
    try:
        content = file.read()
        return json.loads(content)
    except Exception:
        return {}

import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import h5py
import platform

# ----------------------------
# 1. Patch os.get_terminal_size globally
# ----------------------------
def _patched_terminal_size(fd=None):
    try:
        return shutil.get_terminal_size()
    except Exception:
        return os.terminal_size((80, 24))

os.get_terminal_size = _patched_terminal_size

# ----------------------------
# 2. Constants
# ----------------------------
FAIRMAT_DEFINITIONS = Path(__file__).parent / "nexus_definitions"

# ----------------------------
# 3. Generate NeXus file
# ----------------------------
def generate_nexus_file(image_array: np.ndarray,
                        fields: dict,
                        definition: str,
                        output_path: Path):
    """
    Generate a minimal NeXus file.

    Parameters:
        image_array: np.ndarray
        fields: metadata dict, possibly with dotted keys like "instrument.name"
        definition: NeXus application definition (e.g. "NXmicroscopy")
        output_path: Path to write .nxs file
    """
    with h5py.File(output_path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["definition"] = definition

        # Base instrument group
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"

        # Image data
        dset = detector.create_dataset("data", data=image_array)
        dset.attrs["units"] = str(fields.get("units", "counts"))
        detector.attrs["data_shape"] = str(image_array.shape)

        # Inject metadata
        for key, value in fields.items():
            if value in (None, "", [], {}):
                continue
            try:
                parts = key.split(".")
                if len(parts) == 2:
                    group, attr = parts
                    target = {
                        "instrument": instrument,
                        "detector": detector
                    }.get(group, entry.require_group(group))
                    target.attrs[attr] = str(value)
                else:
                    entry.attrs[key] = str(value)
            except Exception:
                pass  # Silently ignore any HDF5 write errors

# ----------------------------
# 4. Validate NeXus file
# ----------------------------
def validate_nexus_file(file_path: Path) -> str:
    """
    Validate the generated NeXus file using `nxinspect` from `nxvalidate`.
    Works across platforms, avoids WinError 6 via patch injection.
    """
    if not file_path.exists():
        return f"❌ File does not exist: {file_path}"
    if not FAIRMAT_DEFINITIONS.exists():
        return f"❌ FAIRmat definitions folder not found at: {FAIRMAT_DEFINITIONS}"

    if platform.system() == "Windows":
        # Escape backslashes to avoid unicode errors
        nxs_path = str(file_path).replace("\\", "\\\\")
        def_path = str(FAIRMAT_DEFINITIONS).replace("\\", "\\\\")

        cmd = [
            sys.executable, "-c",
            (
                "import os, shutil, runpy, sys;"
                "os.get_terminal_size = lambda fd=None: os.terminal_size((80, 24));"
                f"sys.argv = ['nxinspect', '-f', '{nxs_path}', '-a', '-d', '{def_path}', '-e'];"
                "runpy.run_module('nxvalidate.scripts.nxinspect', run_name='__main__')"
            )
        ]
    else:
        # Use 'script' on Unix to simulate TTY
        cmd = [
            "script", "-q", "-c",
            f"nxinspect -f '{file_path}' -a -d '{FAIRMAT_DEFINITIONS}' -e",
            "/dev/null"
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout or result.stderr
        if result.returncode == 0 and "Traceback" not in output:
            return "✅ Validation successful."
        return f"❌ Validation failed:\n{output.strip()}"
    except FileNotFoundError as e:
        return f"❌ Required command not found: {e}"
    except Exception as e:
        return f"❌ Unexpected error during validation: {e}"

# nexus_generator.py

import numpy as np
import h5py
from pathlib import Path
import subprocess
import os


FAIRMAT_DEFINITIONS = Path(__file__).parent / "nexus_definitions"

def generate_nexus_file(image_array: np.ndarray, fields: dict, definition: str, output_path: Path):
    """
    Create a NeXus-compliant HDF5 file from image or stack data.

    Parameters:
    - image_array: numpy array (2D, 3D, or 4D)
    - fields: metadata dictionary (only valid values)
    - definition: application definition (e.g. NXmicroscopy)
    - output_path: file path to write .nxs file
    """

    with h5py.File(output_path, 'w') as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        # Write valid metadata to entry (skip empty)
        for key, value in fields.items():
            if value is None or str(value).strip() == "":
                continue
            try:
                entry.attrs[key] = str(value)
            except Exception:
                entry.create_dataset(key, data=np.string_(str(value)))

        # Set application definition
        entry.attrs["definition"] = definition

        # Create instrument/detector/data group
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"

        detector.create_dataset("data", data=image_array)

        # Add shape hint
        detector.attrs["data_shape"] = str(image_array.shape)

        # Handle units (if given)
        units = fields.get("units", "counts")
        detector["data"].attrs["units"] = str(units)


def validate_nexus_file(file_path: Path) -> str:
    """
    Validate a NeXus file using nxinspect in a pseudo-terminal to avoid TTY errors.
    This method works reliably inside Streamlit and is portable across systems with `script`.
    """

    if not file_path.exists():
        return f"❌ File does not exist: {file_path}"

    if not FAIRMAT_DEFINITIONS.exists():
        return f"❌ FAIRmat definitions folder not found at: {FAIRMAT_DEFINITIONS}"

    # Use `script` to create a pseudo-TTY environment for nxinspect
    try:
        result = subprocess.run(
            [
                "script", "-q", "-c", 
                f"nxinspect -f {file_path} -a -d {FAIRMAT_DEFINITIONS} -e",
                "/dev/null"
            ],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0 and "Traceback" not in result.stdout:
            return "✅ Validation successful: conforms to the selected NeXus application definition."
        else:
            return f"❌ Validation failed:\n{result.stdout or result.stderr}"

    except FileNotFoundError:
        return "❌ Required command `script` or `nxinspect` not found. Please install `nxvalidate` and ensure it's in your PATH."
    except Exception as e:
        return f"❌ Unexpected error during validation: {e}"

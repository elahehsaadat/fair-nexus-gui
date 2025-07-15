# nexus_generator.py

import numpy as np
import h5py
from pathlib import Path

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


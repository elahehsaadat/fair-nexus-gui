import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from nexus_generator import generate_nexus_file


class GenerateNexusFileTests(unittest.TestCase):
    def test_generates_expected_nexus_structure(self):
        image = np.arange(12, dtype=np.uint16).reshape(3, 4)
        fields = {
            "units": "counts",
            "instrument.name": "Test microscope",
            "sample.name": "Test sample",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_output.nxs"
            generate_nexus_file(image, fields, "NXem", output_path)

            self.assertTrue(output_path.exists())
            with h5py.File(output_path, "r") as nexus_file:
                entry = nexus_file["entry"]
                detector = entry["instrument/detector"]

                self.assertEqual(entry.attrs["NX_class"], "NXentry")
                self.assertEqual(entry.attrs["definition"], "NXem")
                self.assertEqual(
                    entry["instrument"].attrs["name"], "Test microscope"
                )
                self.assertEqual(entry["sample"].attrs["name"], "Test sample")
                self.assertEqual(detector["data"].attrs["units"], "counts")
                np.testing.assert_array_equal(detector["data"][:], image)


if __name__ == "__main__":
    unittest.main()

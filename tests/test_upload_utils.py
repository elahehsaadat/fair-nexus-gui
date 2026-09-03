import unittest
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

from upload_utils import (
    build_nexus_file,
    extract_json_from_response,
    extract_metadata_from_tiff,
)


class ExtractJsonFromResponseTests(unittest.TestCase):
    def test_extracts_json_from_markdown_response(self):
        response = '```json\n{"instrument_name": "TESCAN AMBER X"}\n```'

        result = extract_json_from_response(response)

        self.assertEqual(result, {"instrument_name": "TESCAN AMBER X"})

    def test_returns_existing_dictionary(self):
        response = {"experiment_type": "electron microscopy"}

        self.assertIs(extract_json_from_response(response), response)


class ExtractTescanMetadataTests(unittest.TestCase):
    def test_parses_tag_50431_without_exposing_raw_container(self):
        sample_path = Path("benchmark/nxem/tescan_amber_x/input.tif")

        with sample_path.open("rb") as sample_file:
            metadata = extract_metadata_from_tiff(sample_file)

        self.assertNotIn("50431", metadata)
        self.assertIn("VendorKV", metadata)
        self.assertIn("StageX", metadata["VendorKV"])
        self.assertIn("ChamberPressure", metadata["VendorKV"])
        self.assertTrue(
            any(
                source.get("tag") == "50431"
                for source in metadata.get("VendorKV_sources", [])
            )
        )


class BuildNexusFileTests(unittest.TestCase):
    def test_writes_image_definition_and_mapped_metadata(self):
        file_bytes, report = build_nexus_file(
            image_array=np.arange(4, dtype=np.uint16).reshape(2, 2),
            source_filename="instrument.tif",
            metadata={"Make": "TESCAN"},
            mapping={
                "Make": "NXentry/NXinstrument/NXfabrication/vendor",
            },
            definition="NXem",
            instrument="TESCAN AMBER X",
        )

        self.assertEqual(report["written_count"], 1)
        self.assertEqual(report["unresolved_count"], 0)
        with h5py.File(BytesIO(file_bytes), "r") as handle:
            self.assertEqual(handle["entry"].attrs["NX_class"], "NXentry")
            self.assertEqual(handle["entry/definition"].asstr()[()], "NXem")
            np.testing.assert_array_equal(
                handle["entry/source_image/image"][...],
                np.arange(4, dtype=np.uint16).reshape(2, 2),
            )
            vendor = handle["entry/instrument/fabrication/vendor"]
            self.assertEqual(vendor.asstr()[()], "TESCAN")
            self.assertEqual(vendor.attrs["source_metadata_key"], "Make")

    def test_preserves_group_targets_collisions_and_existing_stash(self):
        file_bytes, report = build_nexus_file(
            image_array=np.zeros((2, 2), dtype=np.uint8),
            source_filename="instrument.tif",
            metadata={
                "Make": "TESCAN",
                "Model": "AMBER X",
                "Width": 1024,
                "Compression": 1,
            },
            mapping={
                "Make": "NXentry/NXinstrument/NXfabrication/vendor",
                "Model": "NXentry/NXinstrument/NXfabrication/vendor",
                "Width": "NXentry/NXinstrument/NXdetector",
            },
            definition="NXem",
            instrument="TESCAN AMBER X",
            stashed_metadata={
                "Compression": {"value": 1, "reason": "encoding key"},
            },
        )

        self.assertEqual(report["written_count"], 1)
        self.assertEqual(report["unresolved_count"], 2)
        self.assertEqual(report["stashed_input_count"], 1)
        with h5py.File(BytesIO(file_bytes), "r") as handle:
            unresolved = json.loads(
                handle["entry/unmapped_metadata/data"].asstr()[()]
            )
        self.assertIn("Compression", unresolved)
        self.assertIn("Model", unresolved)
        self.assertIn("Width", unresolved)
        self.assertIn("already exists", unresolved["Model"]["reason"])
        self.assertIn("ends at an NX group", unresolved["Width"]["reason"])


if __name__ == "__main__":
    unittest.main()

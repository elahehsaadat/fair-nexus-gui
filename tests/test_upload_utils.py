import unittest
from pathlib import Path

from upload_utils import extract_json_from_response, extract_metadata_from_tiff


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
        sample_path = Path("sample_data/lamella attached FIB.tif")

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


if __name__ == "__main__":
    unittest.main()

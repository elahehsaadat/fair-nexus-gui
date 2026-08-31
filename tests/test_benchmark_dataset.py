import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = REPO_ROOT / "benchmark" / "benchmark_db.json"
TIFF_DATABASE_PATH = (
    REPO_ROOT / "benchmark" / "tiff_by_technique" / "tiff_database.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BenchmarkDatabaseTests(unittest.TestCase):
    def test_files_and_checksums(self):
        database = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))
        cases = database["cases"]

        self.assertEqual(len(cases), 7)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(
            {case["target_definition"] for case in cases},
            {"NXafm", "NXem", "NXraman"},
        )

        for case in cases:
            with self.subTest(case=case["id"]):
                input_path = REPO_ROOT / case["input_path"]
                self.assertTrue(input_path.is_file())
                self.assertEqual(file_sha256(input_path), case["input_sha256"])

                reference = case["reference_path"]
                if reference is not None:
                    reference_path = REPO_ROOT / reference
                    self.assertTrue(reference_path.is_file())
                    self.assertEqual(
                        file_sha256(reference_path),
                        case["reference_sha256"],
                    )

    def test_tiff_database_matches_parent_database(self):
        database = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))
        tiff_database = json.loads(
            TIFF_DATABASE_PATH.read_text(encoding="utf-8")
        )
        parent_cases = {case["id"]: case for case in database["cases"]}

        self.assertEqual(len(tiff_database["cases"]), 7)
        self.assertEqual(
            {case["target_definition"] for case in tiff_database["cases"]},
            {"NXafm", "NXem", "NXraman"},
        )

        for case in tiff_database["cases"]:
            with self.subTest(case=case["id"]):
                image_path = REPO_ROOT / case["image_path"]
                reference_path = REPO_ROOT / case["reference_path"]
                self.assertEqual(image_path.suffix.lower(), ".tif")
                self.assertEqual(file_sha256(image_path), case["image_sha256"])
                self.assertEqual(
                    file_sha256(reference_path),
                    case["reference_sha256"],
                )
                self.assertEqual(
                    parent_cases[case["id"]]["input_path"],
                    case["image_path"],
                )


if __name__ == "__main__":
    unittest.main()

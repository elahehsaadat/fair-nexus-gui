import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = REPO_ROOT / "benchmark" / "benchmark_db.json"


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
            {"NXem", "NXraman", "NXxps"},
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


if __name__ == "__main__":
    unittest.main()

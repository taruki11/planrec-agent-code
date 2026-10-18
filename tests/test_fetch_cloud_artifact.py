import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_cloud_artifact.py"


class FetchCloudArtifactTests(unittest.TestCase):
    def test_valid_json_is_archived_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.json"
            target = tmp_path / "archive.json"
            source.write_text(json.dumps({"ok": True}), encoding="utf-8")

            run = subprocess.run(
                [sys.executable, str(SCRIPT), source.as_uri(), str(target), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("FETCH_OK", run.stdout)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"ok": True})
            self.assertFalse((tmp_path / "archive.json.part").exists())

    def test_invalid_json_is_not_promoted_to_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.json"
            target = tmp_path / "archive.json"
            source.write_text("5715\n    }", encoding="utf-8")

            run = subprocess.run(
                [sys.executable, str(SCRIPT), source.as_uri(), str(target), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(run.returncode, 0)
            self.assertIn("FETCH_ERROR", run.stderr)
            self.assertFalse(target.exists())
            self.assertTrue((tmp_path / "archive.json.part").exists())


if __name__ == "__main__":
    unittest.main()

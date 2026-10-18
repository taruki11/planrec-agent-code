import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.genrec.provenance import file_sha256, write_run_manifest


class GenRecProvenanceTests(unittest.TestCase):
    def test_manifest_records_input_hashes_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic_ids = root / "semantic_ids.json"
            corpus = root / "train.jsonl"
            semantic_ids.write_text("{}", encoding="utf-8")
            corpus.write_text('{"prompt":"p"}\n', encoding="utf-8")
            output = root / "checkpoint"
            write_run_manifest(
                output,
                "sft",
                "Qwen/Qwen3-0.6B",
                semantic_ids,
                [corpus],
                {"seed": 42},
                "running",
            )
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["inputs"][str(corpus)], file_sha256(corpus))
            self.assertEqual(manifest["pipeline"], "planrec_lm_v1")


if __name__ == "__main__":
    unittest.main()

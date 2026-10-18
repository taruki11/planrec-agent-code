import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MainlineStatusTests(unittest.TestCase):
    def test_status_ledger_and_checker_agree_on_current_gate(self) -> None:
        status = json.loads((ROOT / "MAINLINE_STATUS.json").read_text(encoding="utf-8"))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_mainline_status.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        all_complete = all(
            g.get("status") == "complete" for g in status["gates"].values()
        )
        if all_complete:
            self.assertIn("PROJECT_CLOSED", result.stdout)
            self.assertIn("ALL_GATES=COMPLETE (7/7 E0..E6)", result.stdout)
        else:
            self.assertIn(f"CURRENT_GATE={status['current_gate']}", result.stdout)
            self.assertIn(
                status["gates"][status["current_gate"]]["status"],
                {"in_progress", "blocked"},
            )


if __name__ == "__main__":
    unittest.main()

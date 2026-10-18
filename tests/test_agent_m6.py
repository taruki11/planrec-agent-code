import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from planrec.agent.environment import AgentAction, AgentState
from planrec.agent.recommendation_agent import AgenticRecommendationEngine
from planrec.config import SASRecConfig
from planrec.models.sasrec import SASRec


class M6AgentTests(unittest.TestCase):
    def test_agent_environment_state(self) -> None:
        state = AgentState(max_budget_steps=3)
        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.remaining_budget, 1.0)
        self.assertFalse(state.is_halted)

    def test_agentic_policy_execution(self) -> None:
        config = SASRecConfig(num_items=10, max_seq_len=5, hidden_size=16)
        model = SASRec(config)
        seq = torch.tensor([[1, 2, 3, 0, 0]])

        agent = AgenticRecommendationEngine(max_budget_steps=3, uncertainty_threshold=0.5)
        res = agent.execute_policy(model, seq, torch.device("cpu"))

        self.assertIn("steps_taken", res)
        self.assertTrue(res["budget_compliant"])
        self.assertGreater(res["steps_taken"], 0)
        self.assertIn("actions", res)


if __name__ == "__main__":
    unittest.main()

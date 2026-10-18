from typing import Any

import numpy as np
import torch

from planrec.agent.environment import AgentAction, AgentState
from planrec.uncertainty import normalized_entropy


class AgenticRecommendationEngine:
    """Autonomous Multi-step Agentic Recommendation Policy."""

    def __init__(self, max_budget_steps: int = 3, uncertainty_threshold: float = 0.85) -> None:
        self.max_budget_steps = max_budget_steps
        self.uncertainty_threshold = uncertainty_threshold

    def execute_policy(
        self,
        model: torch.nn.Module,
        item_sequence: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any]:
        """Execute autonomous step-by-step decision making loop until HALT or budget depletion."""
        state = AgentState(max_budget_steps=self.max_budget_steps)
        batch_size = item_sequence.size(0)

        # Initial uncertainty estimation
        with torch.no_grad():
            raw_logits = model(item_sequence.to(device))
            item_logits = raw_logits[:, 1:]
            initial_ent = float(normalized_entropy(item_logits).mean().item())

        state.current_uncertainty = initial_ent
        tool_errors = 0
        fallback_triggered = False

        while not state.is_halted and state.step_count < state.max_budget_steps:
            state.step_count += 1
            state.remaining_budget -= (1.0 / state.max_budget_steps)

            if state.current_uncertainty > self.uncertainty_threshold and not state.retrieved_metadata:
                # High uncertainty -> Step 1: Retrieve Metadata
                state.chosen_action_history.append(AgentAction.RETRIEVE_METADATA)
                state.retrieved_metadata = True
                # Simulating metadata retrieval reducing uncertainty
                state.current_uncertainty *= 0.8

            elif state.retrieved_metadata and state.verifier_passed is None:
                # Step 2: Invoke Verifier to check plan consistency
                state.chosen_action_history.append(AgentAction.INVOKE_VERIFIER)
                # Verifier test
                if state.current_uncertainty > 0.7:
                    state.verifier_passed = False
                    tool_errors += 1
                else:
                    state.verifier_passed = True

            elif state.verifier_passed is False:
                # Step 3: Fallback to Direct Path upon verification failure
                state.chosen_action_history.append(AgentAction.FALLBACK_DIRECT)
                fallback_triggered = True
                state.is_halted = True

            else:
                # Default: Halt and Output
                state.chosen_action_history.append(AgentAction.HALT_AND_OUTPUT)
                state.is_halted = True

        budget_compliant = state.step_count <= state.max_budget_steps

        return {
            "steps_taken": state.step_count,
            "remaining_budget": state.remaining_budget,
            "actions": [a.name for a in state.chosen_action_history],
            "fallback_triggered": fallback_triggered,
            "tool_errors": tool_errors,
            "budget_compliant": budget_compliant,
            "final_uncertainty": state.current_uncertainty,
        }

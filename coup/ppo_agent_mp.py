"""
PPO Agent wrapper for Gen 6 multiplayer model.
Plays via the standard Agent interface with two-step action/target selection.
"""
import torch
from typing import List, Optional
from torch.distributions import Categorical

from .game import Card, ActionType, Action
from .agents import Agent
from .ppo_model_gen6 import CoupLSTMPPOv2
from .coup_env_mp import CoupEnvMP, OBS_DIM, TARGET_SIZE, MAX_OPPONENTS, TARGETED_ACTIONS
from .coup_env import RLAction, ACT_TYPE_TO_RL, RL_TO_ACT_TYPE, BLOCK_TO_RL, RL_TO_BLOCK


class PPOAgentMP(Agent):
    """
    Agent wrapper for the Gen 6 multiplayer PPO model.
    Handles two-step action selection (action type + target).
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CoupLSTMPPOv2(input_dim=OBS_DIM, hidden_dim=512,
                                    num_actions=RLAction.SIZE, num_targets=TARGET_SIZE).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.env_helper = CoupEnvMP()
        self.hidden_state = self.model.reset_hidden(1, self.device)

    def _get_sorted_opponents(self, view: dict) -> list:
        """Get opponents sorted by relative seat order."""
        return self.env_helper._get_relative_opponents(view)

    def _forward(self, obs, mask, use_target_head=False):
        """Run forward pass and sample from the appropriate head."""
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(mask).view(1, 1, -1).to(self.device)

        with torch.no_grad():
            action_logits, target_logits, _, self.hidden_state = self.model(state_tensor, self.hidden_state)

            if use_target_head:
                target_logits[~mask_tensor] = -1e9
                dist = Categorical(logits=target_logits)
            else:
                action_logits[~mask_tensor] = -1e9
                dist = Categorical(logits=action_logits)

            return dist.sample().item()

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        # Step 1: Choose action type
        msg = {'req_type': 'ACT', 'view': view, 'context': legal_actions}
        obs, action_mask, _, _ = self.env_helper._encode_state(msg)
        idx = self._forward(obs, action_mask, use_target_head=False)

        act_type = RL_TO_ACT_TYPE.get(idx, ActionType.INCOME)
        matching = [a for a in legal_actions if a.action_type == act_type]

        if not matching:
            return legal_actions[0]

        if act_type in TARGETED_ACTIONS and len(matching) > 1:
            # Step 2: Choose target
            msg_target = {'req_type': 'TARGET', 'view': view,
                          'context': {'action_type': act_type, 'targets': matching}}
            obs_t, _, target_mask, _ = self.env_helper._encode_state(msg_target)
            target_slot = self._forward(obs_t, target_mask, use_target_head=True)

            # Map slot back to action
            sorted_opps = self._get_sorted_opponents(view)
            if target_slot < len(sorted_opps):
                target_pid = sorted_opps[target_slot]['player_id']
                for a in matching:
                    if a.target_idx == target_pid:
                        return a
            return matching[0]

        return matching[0]

    def choose_challenge(self, view: dict, claimer_idx: int, claimed_card: Card) -> bool:
        msg = {'req_type': 'CHL', 'view': view, 'context': claimed_card}
        obs, action_mask, _, _ = self.env_helper._encode_state(msg)
        idx = self._forward(obs, action_mask, use_target_head=False)
        return idx == RLAction.YES

    def choose_counteraction(self, view: dict, actor_idx: int, action_type: ActionType, blocking_cards: List[Card]) -> Optional[Card]:
        msg = {'req_type': 'CTR', 'view': view, 'context': {'action': action_type, 'cards': blocking_cards}}
        obs, action_mask, _, _ = self.env_helper._encode_state(msg)
        idx = self._forward(obs, action_mask, use_target_head=False)
        if idx == RLAction.NO:
            return None
        return RL_TO_BLOCK.get(idx, None)

    def choose_challenge_counter(self, view: dict, blocker_idx: int, blocking_card: Card) -> bool:
        msg = {'req_type': 'CC', 'view': view, 'context': blocking_card}
        obs, action_mask, _, _ = self.env_helper._encode_state(msg)
        idx = self._forward(obs, action_mask, use_target_head=False)
        return idx == RLAction.YES

    def choose_card_to_lose(self, view: dict) -> int:
        msg = {'req_type': 'LOSE', 'view': view, 'context': None}
        obs, action_mask, _, _ = self.env_helper._encode_state(msg)
        idx = self._forward(obs, action_mask, use_target_head=False)
        return 0 if idx == RLAction.LOSE_CARD_0 else 1

    def choose_exchange_cards(self, view: dict, all_cards: List[Card], num_to_keep: int) -> List[int]:
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        ranked = sorted(range(len(all_cards)), key=lambda i: -val[all_cards[i]])
        return sorted(ranked[:num_to_keep])

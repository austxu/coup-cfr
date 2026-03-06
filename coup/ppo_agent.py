import torch
import numpy as np
from typing import List, Optional

from .game import Card, ActionType, Action
from .agents import Agent
from .ppo_model import CoupLSTMPPO
from .coup_env import CoupEnv

class PPOAgent(Agent):
    """
    An Agent wrapper that allows a trained PyTorch PPO model 
    to play natively via the CoupGame engine's Agent interface.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CoupLSTMPPO(input_dim=23, hidden_dim=64, num_actions=15).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # We use CoupEnv's encoding logic to process views
        self.env_helper = CoupEnv()
        self.hidden_state = self.model.reset_hidden(1, self.device)
        
    def _get_action_index(self, req_type: str, view: dict, legal_actions: List[Action] = None, context=None) -> int:
        msg = {'req_type': req_type, 'view': view, 'context': context}
        if legal_actions:
            msg['context'] = legal_actions
            
        obs, action_mask = self.env_helper._encode_state(msg)
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).view(1, 1, -1).to(self.device)
        
        with torch.no_grad():
            logits, _, self.hidden_state = self.model(state_tensor, self.hidden_state)
            logits[~mask_tensor] = -1e9
            
            # Use probabilistic sampling instead of greedy argmax so the AI varies its plays
            from torch.distributions import Categorical
            dist = Categorical(logits=logits)
            action_idx = dist.sample().item()
            
        return action_idx

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        idx = self._get_action_index('ACT', view, legal_actions=legal_actions)
        from .coup_env import RL_TO_ACT_TYPE
        
        if idx not in RL_TO_ACT_TYPE:
            return legal_actions[0]
            
        act_type = RL_TO_ACT_TYPE[idx]
        
        for a in legal_actions:
            if a.action_type == act_type:
                return a
                
        # Fallback if greedy choice is somehow illegal
        return legal_actions[0]

    def choose_challenge(self, view: dict, claimer_idx: int, claimed_card: Card) -> bool:
        idx = self._get_action_index('CHL', view, context=claimed_card)
        from .coup_env import RLAction
        return idx == RLAction.YES
        
    def choose_counteraction(self, view: dict, actor_idx: int, action_type: ActionType, blocking_cards: List[Card]) -> Optional[Card]:
        idx = self._get_action_index('CTR', view, context={'action': action_type, 'cards': blocking_cards})
        from .coup_env import RLAction, RL_TO_BLOCK
        if idx == RLAction.NO:
            return None
        return RL_TO_BLOCK.get(idx, None)

    def choose_challenge_counter(self, view: dict, blocker_idx: int, blocking_card: Card) -> bool:
        idx = self._get_action_index('CC', view, context=blocking_card)
        from .coup_env import RLAction
        return idx == RLAction.YES

    def choose_card_to_lose(self, view: dict) -> int:
        idx = self._get_action_index('LOSE', view)
        from .coup_env import RLAction
        return 0 if idx == RLAction.LOSE_CARD_0 else 1

    def choose_exchange_cards(self, view: dict, all_cards: List[Card], num_to_keep: int) -> List[int]:
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        ranked = sorted(range(len(all_cards)), key=lambda i: -val[all_cards[i]])
        return sorted(ranked[:num_to_keep])

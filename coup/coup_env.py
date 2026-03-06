"""
Gymnasium environment wrapper for 2-player Coup.

Allows a Deep RL agent to play against a suite of hardcoded
Zoo bots. Uses a background thread for the game engine to allow
the synchronous `CoupGame` to yield control to the asynchronous `env.step()`.
"""

import threading
import queue
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Any, Tuple

from .game import (
    CoupGame, Card, ActionType, Action, ACTION_CHARACTER, COUNTERABLE_BY
)
from .agents import Agent
from .info_set import CARD_ABBREV, bucket_coins


# -----------------------------------------------------------------------------
# Discrete Action Space
# -----------------------------------------------------------------------------
# We map all 15 possible decisions into a discrete action space.
# The environment provides an `action_mask` so PPO only picks legal moves.

class RLAction:
    # 0-6: Primary Actions (Target is implicitly opponent in 2P)
    INCOME = 0
    FOREIGN_AID = 1
    COUP = 2
    TAX = 3
    ASSASSINATE = 4
    STEAL = 5
    EXCHANGE = 6
    
    # 7-8: Binary Yes/No (Challenge, Challenge Counter)
    YES = 7
    NO = 8
    
    # 9-12: Counteractions (Blocks)
    BLOCK_DUKE = 9
    BLOCK_CAPTAIN = 10
    BLOCK_AMBASSADOR = 11
    BLOCK_CONTESSA = 12
    # NO is shared with 8
    
    # 13-14: Lose Card Choices
    LOSE_CARD_0 = 13
    LOSE_CARD_1 = 14
    
    # 15: Exchange (we use a heuristic for exchange to simplify RL)
    # Heuristic: keep most valuable cards.
    
    SIZE = 15

# Map engine ActionTypes to RLAction
ACT_TYPE_TO_RL = {
    ActionType.INCOME: RLAction.INCOME,
    ActionType.FOREIGN_AID: RLAction.FOREIGN_AID,
    ActionType.COUP: RLAction.COUP,
    ActionType.TAX: RLAction.TAX,
    ActionType.ASSASSINATE: RLAction.ASSASSINATE,
    ActionType.STEAL: RLAction.STEAL,
    ActionType.EXCHANGE: RLAction.EXCHANGE,
}
RL_TO_ACT_TYPE = {v: k for k, v in ACT_TYPE_TO_RL.items()}

# Map Card blocks to RLAction
BLOCK_TO_RL = {
    Card.DUKE: RLAction.BLOCK_DUKE,
    Card.CAPTAIN: RLAction.BLOCK_CAPTAIN,
    Card.AMBASSADOR: RLAction.BLOCK_AMBASSADOR,
    Card.CONTESSA: RLAction.BLOCK_CONTESSA,
}
RL_TO_BLOCK = {v: k for k, v in BLOCK_TO_RL.items()}


# -----------------------------------------------------------------------------
# RL Agent Proxy (Runs inside the game engine thread)
# -----------------------------------------------------------------------------

class RLAgentProxy(Agent):
    """
    Acts as a standard Coup Agent inside the engine thread.
    When a decision is needed, it pushes the observation to the queue
    and blocks waiting for an action from the RL thread.
    """
    def __init__(self, obs_queue: queue.Queue, act_queue: queue.Queue):
        self.obs_q = obs_queue
        self.act_q = act_queue

    def _request_action(self, req_type: str, view: dict, context: Any = None) -> int:
        """Send state to RL thread and wait for choice."""
        payload = {
            'req_type': req_type,
            'view': view,
            'context': context
        }
        self.obs_q.put(payload)
        
        # Block until RL thread calls env.step(action)
        action_idx = self.act_q.get()
        return action_idx

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        act_idx = self._request_action('ACT', view, legal_actions)
        act_type = RL_TO_ACT_TYPE[act_idx]
        
        # Reconstruct full Action object (target is opponent in 2P)
        opp = view['opponents'][0]['player_id']
        target = opp if act_type in (ActionType.COUP, ActionType.ASSASSINATE, ActionType.STEAL) else None
        return Action(act_type, view['player_id'], target)

    def choose_challenge(self, view: dict, claimer_idx: int, claimed_card: Card) -> bool:
        act_idx = self._request_action('CHL', view, claimed_card)
        return act_idx == RLAction.YES

    def choose_counteraction(self, view: dict, actor_idx: int, action_type: ActionType, blocking_cards: List[Card]) -> Optional[Card]:
        act_idx = self._request_action('CTR', view, {'action': action_type, 'cards': blocking_cards})
        if act_idx == RLAction.NO:
            return None
        return RL_TO_BLOCK[act_idx]

    def choose_challenge_counter(self, view: dict, blocker_idx: int, blocking_card: Card) -> bool:
        act_idx = self._request_action('CC', view, blocking_card)
        return act_idx == RLAction.YES

    def choose_card_to_lose(self, view: dict) -> int:
        act_idx = self._request_action('LOSE', view)
        return 0 if act_idx == RLAction.LOSE_CARD_0 else 1

    def choose_exchange_cards(self, view: dict, all_cards: List[Card], num_to_keep: int) -> List[int]:
        # Heuristic exchange: keep best cards. RL agent doesn't do combinatorics.
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        ranked = sorted(range(len(all_cards)), key=lambda i: -val[all_cards[i]])
        return sorted(ranked[:num_to_keep])


# -----------------------------------------------------------------------------
# Gymnasium Env (Runs in main RL thread)
# -----------------------------------------------------------------------------

class CoupEnv(gym.Env):
    """
    Gym environment for 2-player Coup.
    Observation is a float32 vector.
    Action is a discrete 15-class integer.
    Info dict contains 'action_mask' (boolean array).
    """

    def __init__(self, opponent_cls: type = None):
        super(CoupEnv, self).__init__()
        
        self.action_space = spaces.Discrete(RLAction.SIZE)
        
        # State vector size:
        # My cards one-hot: 5
        # My coins: 1
        # Opponent coins: 1
        # Opponent alive cards: 1
        # Revealed cards counts: 5 (max 3 each)
        # Decision type one-hot: 5 (ACT, CHL, CTR, CC, LOSE)
        # Pending card claim one-hot (context): 5
        # My claimed cards one-hot: 5 (NEW)
        # Opponent claimed cards one-hot: 5 (NEW)
        # My caught bluff count: 1 (NEW)
        # Opponent caught bluff count: 1 (NEW)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(35,), dtype=np.float32)
        
        self.opponent_cls = opponent_cls  # Set for evaluation, otherwise PBT Zoo handles opponent
        
        self.obs_q = queue.Queue()
        self.act_q = queue.Queue()
        
        self.game_thread = None
        self.game_result = None

    def _game_worker(self, opponent):
        """Runs the Coup game engine in a background thread."""
        try:
            agents = [RLAgentProxy(self.obs_q, self.act_q), opponent]
            # Randomize seat to prevent first-player bias
            if random.random() < 0.5:
                agents.reverse()
                self.rl_player_id = 1
            else:
                self.rl_player_id = 0
                
            game = CoupGame(agents, num_players=2, verbose=False)
            self.game_result = game.play_game()
            
            # Signal game over to env
            self.obs_q.put({'req_type': 'DONE', 'winner': self.game_result})
        except Exception as e:
            self.obs_q.put({'req_type': 'ERROR', 'error': e})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # If thread is running (from early terminate), clear it
        if self.game_thread and self.game_thread.is_alive():
            # Send dummy actions until it finishes
            while self.game_thread.is_alive():
                try:
                    msg = self.obs_q.get(timeout=0.1)
                    if msg['req_type'] != 'DONE':
                        self.act_q.put(0) # dummy action
                except queue.Empty:
                    pass
            self.game_thread.join()

        # Empty queues
        while not self.obs_q.empty(): self.obs_q.get()
        while not self.act_q.empty(): self.act_q.get()

        # Instantiate opponent (if none provided, use Random as fallback for now)
        from .agents import RandomAgent
        opp = self.opponent_cls() if self.opponent_cls else RandomAgent()

        self.game_thread = threading.Thread(target=self._game_worker, args=(opp,))
        self.game_thread.start()

        # Wait for first decision
        obs, reward, done, trunc, info = self._wait_for_obs()
        return obs, info

    def step(self, action: int):
        # Send action to game engine thread
        self.act_q.put(action)
        return self._wait_for_obs()

    def _wait_for_obs(self):
        """Block until the game thread sends an observation or game ends."""
        try:
            msg = self.obs_q.get(timeout=5.0)
        except queue.Empty:
            if not self.game_thread.is_alive():
                raise RuntimeError("Game thread crashed!")
            raise RuntimeError("Game thread hung!")
        
        if msg['req_type'] == 'ERROR':
            raise msg['error']
            
        if msg['req_type'] == 'DONE':
            winner = msg['winner']
            # Reward: +1 for win, -1 for loss, 0 for timeout draw
            if winner == self.rl_player_id:
                reward = 1.0
            elif winner is not None:
                reward = -1.0
            else:
                reward = 0.0
                
            self.game_thread.join()
            return np.zeros(35, dtype=np.float32), reward, True, False, {'action_mask': np.zeros(RLAction.SIZE, dtype=bool)}

        # Otherwise, parse the game state request into a vector observation
        obs, action_mask = self._encode_state(msg)
        return obs, 0.0, False, False, {'action_mask': action_mask}

    def _encode_state(self, msg: dict) -> Tuple[np.ndarray, np.ndarray]:
        req_type = msg['req_type']
        view = msg['view']
        ctx = msg['context']
        
        obs = np.zeros(35, dtype=np.float32)
        mask = np.zeros(RLAction.SIZE, dtype=bool)
        
        # 0-4: My cards (one hot)
        cards_idx = {Card.DUKE: 0, Card.ASSASSIN: 1, Card.CAPTAIN: 2, Card.AMBASSADOR: 3, Card.CONTESSA: 4}
        for c in view['my_cards']:
            obs[0 + cards_idx[c]] += 1.0
            
        # 5: My coins
        obs[5] = float(view['my_coins'])
        
        # 6-7: Opponent data
        opp = view['opponents'][0]
        obs[6] = float(opp['coins'])
        obs[7] = float(opp['influence_count'])
        
        # 8-12: Revealed cards
        for c in opp['revealed']:
            obs[8 + cards_idx[c]] += 1.0
            
        # 13-17: My Claimed Cards
        for c in view.get('my_claimed_cards', []):
            obs[13 + cards_idx[c]] = 1.0
            
        # 18-22: Opponent's Claimed Cards
        for c in opp.get('claimed_cards', []):
            obs[18 + cards_idx[c]] = 1.0
            
        # 23: My Caught Bluff Count
        obs[23] = float(view.get('my_caught_bluff_count', 0))
        
        # 24: Opponent's Caught Bluff Count
        obs[24] = float(opp.get('caught_bluff_count', 0))
            
        # 25-29: Decision Type
        dec_map = {'ACT': 25, 'CHL': 26, 'CTR': 27, 'CC': 28, 'LOSE': 29}
        obs[dec_map[req_type]] = 1.0
        
        # 30-34: Context Card (if applicable)
        if req_type == 'CHL' and isinstance(ctx, Card):
            obs[30 + cards_idx[ctx]] = 1.0
        elif req_type == 'CTR' and isinstance(ctx, dict):
            # No specific card is claimed in a Steal/Foreign Aid before the counter,
            # but we can optionally encode the action type being countered (Steal/FA).
            # For simplicity, we leave context 0.0 for CTR.
            pass
        elif req_type == 'CC' and isinstance(ctx, Card):
            obs[30 + cards_idx[ctx]] = 1.0

        # Action Mask
        if req_type == 'ACT':
            # Context is legal_actions list
            for a in ctx:
                mask[ACT_TYPE_TO_RL[a.action_type]] = True
        elif req_type in ('CHL', 'CC'):
            mask[RLAction.YES] = True
            mask[RLAction.NO] = True
        elif req_type == 'CTR':
            # Context is dict with 'cards' list of block options
            mask[RLAction.NO] = True
            for c in ctx['cards']:
                mask[BLOCK_TO_RL[c]] = True
        elif req_type == 'LOSE':
            mask[RLAction.LOSE_CARD_0] = True
            if len(view['my_cards']) > 1:
                mask[RLAction.LOSE_CARD_1] = True

        return obs, mask


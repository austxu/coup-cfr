"""
Gymnasium environment wrapper for N-player Coup (3-6 players).

Allows a Deep RL agent to play against multiple opponents.
Uses separate action and target steps for targeted actions (Coup/Assassinate/Steal).
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
from .coup_env import RLAction, ACT_TYPE_TO_RL, RL_TO_ACT_TYPE, BLOCK_TO_RL, RL_TO_BLOCK


# Max opponents (for 6-player game, each player sees 5 opponents)
MAX_OPPONENTS = 5

# Observation dimensions
SELF_DIMS = 18         # My cards(5) + coins(1) + revealed(5) + claimed(5) + bluff_count(1) + alive_count(1)
OPP_DIMS = 14          # Per opponent: coins(1) + influence(1) + alive(1) + revealed(5) + claimed(5) + bluff_count(1)
OPP_TOTAL = MAX_OPPONENTS * OPP_DIMS  # 70
DECISION_DIMS = 5      # ACT, CHL, CTR, CC, LOSE
CONTEXT_DIMS = 5       # Card one-hot
HISTORY_ENTRY_DIMS = 10  # Per history action
HISTORY_LENGTH = 5
HISTORY_TOTAL = HISTORY_LENGTH * HISTORY_ENTRY_DIMS  # 50
ACTOR_DIMS = 6         # Who is acting (one-hot over 6 player slots, relative)
RESERVED = 1           # Padding

OBS_DIM = SELF_DIMS + OPP_TOTAL + DECISION_DIMS + CONTEXT_DIMS + HISTORY_TOTAL + ACTOR_DIMS + RESERVED  # 155

# Target action space
TARGET_SIZE = MAX_OPPONENTS  # 5 possible targets (opponent slots)

# Actions that require a target
TARGETED_ACTIONS = {ActionType.COUP, ActionType.ASSASSINATE, ActionType.STEAL}


class RLAgentProxyMP(Agent):
    """
    Multiplayer RL Agent Proxy — runs inside the game engine thread.
    For targeted actions, makes two requests: action choice then target choice.
    """
    def __init__(self, obs_queue: queue.Queue, act_queue: queue.Queue):
        self.obs_q = obs_queue
        self.act_q = act_queue

    def _request_action(self, req_type: str, view: dict, context: Any = None) -> int:
        payload = {'req_type': req_type, 'view': view, 'context': context}
        self.obs_q.put(payload)
        action_idx = self.act_q.get()
        return action_idx

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        # Step 1: Choose action type
        act_idx = self._request_action('ACT', view, legal_actions)
        act_type = RL_TO_ACT_TYPE.get(act_idx, ActionType.INCOME)

        # Find matching action(s)
        matching = [a for a in legal_actions if a.action_type == act_type]

        if not matching:
            return legal_actions[0]  # Fallback

        if act_type in TARGETED_ACTIONS and len(matching) > 1:
            # Step 2: Choose target
            target_idx = self._request_action('TARGET', view, {
                'action_type': act_type,
                'targets': matching
            })
            if 0 <= target_idx < len(matching):
                return matching[target_idx]
            return matching[0]
        
        return matching[0]

    def choose_challenge(self, view: dict, claimer_idx: int, claimed_card: Card) -> bool:
        act_idx = self._request_action('CHL', view, claimed_card)
        return act_idx == RLAction.YES

    def choose_counteraction(self, view: dict, actor_idx: int, action_type: ActionType, blocking_cards: List[Card]) -> Optional[Card]:
        act_idx = self._request_action('CTR', view, {'action': action_type, 'cards': blocking_cards})
        if act_idx == RLAction.NO:
            return None
        return RL_TO_BLOCK.get(act_idx, None)

    def choose_challenge_counter(self, view: dict, blocker_idx: int, blocking_card: Card) -> bool:
        act_idx = self._request_action('CC', view, blocking_card)
        return act_idx == RLAction.YES

    def choose_card_to_lose(self, view: dict) -> int:
        act_idx = self._request_action('LOSE', view)
        return 0 if act_idx == RLAction.LOSE_CARD_0 else 1

    def choose_exchange_cards(self, view: dict, all_cards: List[Card], num_to_keep: int) -> List[int]:
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        ranked = sorted(range(len(all_cards)), key=lambda i: -val[all_cards[i]])
        return sorted(ranked[:num_to_keep])


class CoupEnvMP(gym.Env):
    """
    Gym environment for N-player Coup (3-6 players).
    Uses two-step action selection for targeted actions.
    """

    def __init__(self, opponent_cls_list=None, num_players=4):
        super(CoupEnvMP, self).__init__()

        self.num_players = num_players
        self.action_space = spaces.Discrete(RLAction.SIZE)
        self.target_space = spaces.Discrete(TARGET_SIZE)
        self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)

        self.opponent_cls_list = opponent_cls_list  # List of callables returning Agent instances
        
        self.obs_q = queue.Queue()
        self.act_q = queue.Queue()
        
        self.game_thread = None
        self.game_result = None
        self.rl_player_id = 0

    def _game_worker(self, opponents):
        """Runs the Coup game engine in a background thread."""
        try:
            rl_proxy = RLAgentProxyMP(self.obs_q, self.act_q)
            
            # Build agent list with RL agent in a random seat
            agents = list(opponents)
            self.rl_player_id = random.randint(0, self.num_players - 1)
            agents.insert(self.rl_player_id, rl_proxy)
            
            game = CoupGame(agents, num_players=self.num_players, verbose=False)
            self.game_result = game.play_game()
            
            self.obs_q.put({'req_type': 'DONE', 'winner': self.game_result})
        except Exception as e:
            self.obs_q.put({'req_type': 'ERROR', 'error': e})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clean up old thread
        if self.game_thread and self.game_thread.is_alive():
            while self.game_thread.is_alive():
                try:
                    msg = self.obs_q.get(timeout=0.1)
                    if msg['req_type'] not in ('DONE', 'ERROR'):
                        self.act_q.put(0)
                except queue.Empty:
                    pass
            self.game_thread.join()

        while not self.obs_q.empty(): self.obs_q.get()
        while not self.act_q.empty(): self.act_q.get()

        # Build opponents
        from .agents import RandomAgent
        if self.opponent_cls_list:
            opponents = [cls() for cls in self.opponent_cls_list]
        else:
            opponents = [RandomAgent() for _ in range(self.num_players - 1)]

        self.game_thread = threading.Thread(target=self._game_worker, args=(opponents,))
        self.game_thread.start()

        obs, reward, done, trunc, info = self._wait_for_obs()
        return obs, info

    def step(self, action: int):
        self.act_q.put(action)
        return self._wait_for_obs()

    def _wait_for_obs(self):
        try:
            msg = self.obs_q.get(timeout=10.0)
        except queue.Empty:
            if not self.game_thread.is_alive():
                raise RuntimeError("Game thread crashed!")
            raise RuntimeError("Game thread hung!")

        if msg['req_type'] == 'ERROR':
            raise msg['error']

        if msg['req_type'] == 'DONE':
            winner = msg['winner']
            if winner == self.rl_player_id:
                reward = 1.0
            elif winner is not None:
                reward = -1.0
            else:
                reward = 0.0
            
            self.game_thread.join()
            return (np.zeros(OBS_DIM, dtype=np.float32), reward, True, False,
                    {'action_mask': np.zeros(RLAction.SIZE, dtype=bool),
                     'target_mask': np.zeros(TARGET_SIZE, dtype=bool),
                     'is_target_step': False})

        obs, action_mask, target_mask, is_target_step = self._encode_state(msg)
        return (obs, 0.0, False, False,
                {'action_mask': action_mask,
                 'target_mask': target_mask,
                 'is_target_step': is_target_step})

    def _get_relative_opponents(self, view: dict) -> list:
        """
        Return opponents sorted by seat order relative to the RL player.
        Clockwise from player's left.
        """
        my_id = view['player_id']
        opps = sorted(view['opponents'], key=lambda o: ((o['player_id'] - my_id) % self.num_players))
        return opps

    def _encode_state(self, msg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        req_type = msg['req_type']
        view = msg['view']
        ctx = msg['context']

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action_mask = np.zeros(RLAction.SIZE, dtype=bool)
        target_mask = np.zeros(TARGET_SIZE, dtype=bool)
        is_target_step = (req_type == 'TARGET')

        cards_idx = {Card.DUKE: 0, Card.ASSASSIN: 1, Card.CAPTAIN: 2, Card.AMBASSADOR: 3, Card.CONTESSA: 4}
        my_id = view['player_id']

        # === SELF (0-17) ===
        # 0-4: My cards
        for c in view['my_cards']:
            obs[0 + cards_idx[c]] += 1.0
        # 5: My coins
        obs[5] = float(view['my_coins'])
        # 6-10: My revealed cards
        for c in view.get('my_revealed', []):
            obs[6 + cards_idx[c]] += 1.0
        # 11-15: My claimed cards
        for c in view.get('my_claimed_cards', []):
            obs[11 + cards_idx[c]] = 1.0
        # 16: My caught bluff count
        obs[16] = float(view.get('my_caught_bluff_count', 0))
        # 17: Number of alive players (normalized)
        alive_count = sum(1 for o in view['opponents'] if o['alive']) + (1 if view['my_cards'] else 0)
        obs[17] = alive_count / 6.0

        # === OPPONENTS (18-87) ===
        opps = self._get_relative_opponents(view)
        for slot_idx, opp in enumerate(opps):
            if slot_idx >= MAX_OPPONENTS:
                break
            base = 18 + slot_idx * OPP_DIMS
            obs[base + 0] = float(opp['coins'])
            obs[base + 1] = float(opp['influence_count'])
            obs[base + 2] = 1.0 if opp['alive'] else 0.0
            for c in opp['revealed']:
                obs[base + 3 + cards_idx[c]] += 1.0
            for c in opp.get('claimed_cards', []):
                obs[base + 8 + cards_idx[c]] = 1.0
            obs[base + 13] = float(opp.get('caught_bluff_count', 0))

        # === DECISION TYPE (88-92) ===
        dec_map = {'ACT': 88, 'CHL': 89, 'CTR': 90, 'CC': 91, 'LOSE': 92, 'TARGET': 88}
        obs[dec_map[req_type]] = 1.0

        # === CONTEXT CARD (93-97) ===
        if req_type == 'CHL' and isinstance(ctx, Card):
            obs[93 + cards_idx[ctx]] = 1.0
        elif req_type == 'CC' and isinstance(ctx, Card):
            obs[93 + cards_idx[ctx]] = 1.0

        # === ACTION HISTORY (98-147) ===
        history = view.get('action_history', [])[-HISTORY_LENGTH:]
        base_idx = 98
        for h in history:
            act_id = float(ACT_TYPE_TO_RL.get(h['action'], 0))
            
            # Relative position encoding for actor
            actor_id = h['player']
            actor_rel = ((actor_id - my_id) % self.num_players) / max(self.num_players - 1, 1)
            
            # Relative position encoding for target
            if h['target'] is not None:
                target_rel = ((h['target'] - my_id) % self.num_players) / max(self.num_players - 1, 1)
            else:
                target_rel = -1.0

            blocked = 1.0 if h.get('was_blocked') else 0.0
            challenged = 1.0 if h.get('was_challenged') else 0.0
            chall_won = 1.0 if h.get('challenge_won') else 0.0
            card_lost = 1.0 if h.get('card_lost') else 0.0

            obs[base_idx:base_idx + 10] = [
                act_id, actor_rel, target_rel,
                blocked, challenged, chall_won, card_lost,
                0.0, 0.0, 0.0  # padding
            ]
            base_idx += HISTORY_ENTRY_DIMS

        # === ACTOR OF PENDING ACTION (148-153) ===
        # Not always applicable, zero otherwise
        
        # === ACTION MASK ===
        if req_type == 'TARGET':
            # Target selection: mask alive opponents
            targets = ctx.get('targets', [])
            for i, opp in enumerate(opps):
                if i >= MAX_OPPONENTS:
                    break
                # Check if this opponent is a valid target
                for t in targets:
                    if t.target_idx == opp['player_id']:
                        target_mask[i] = True
                        break
        elif req_type == 'ACT':
            for a in ctx:
                mask_idx = ACT_TYPE_TO_RL.get(a.action_type)
                if mask_idx is not None:
                    action_mask[mask_idx] = True
        elif req_type in ('CHL', 'CC'):
            action_mask[RLAction.YES] = True
            action_mask[RLAction.NO] = True
        elif req_type == 'CTR':
            action_mask[RLAction.NO] = True
            for c in ctx['cards']:
                action_mask[BLOCK_TO_RL[c]] = True
        elif req_type == 'LOSE':
            action_mask[RLAction.LOSE_CARD_0] = True
            if len(view['my_cards']) > 1:
                action_mask[RLAction.LOSE_CARD_1] = True

        return obs, action_mask, target_mask, is_target_step

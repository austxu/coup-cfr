"""
CFR Agent — plays Coup using a trained MCCFR strategy.

Loads a strategy file (JSON) and looks up the average strategy
at each decision point. Falls back to uniform random for unseen info sets.
"""

import random
from typing import List, Optional

from .game import Card, ActionType, Action, ACTION_CHARACTER, COUNTERABLE_BY
from .agents import Agent
from .info_set import make_info_key
from .cfr import ACTION_KEYS, CARD_VALUE


class CFRAgent(Agent):
    """Agent that plays according to a trained CFR strategy."""

    def __init__(self, strategy: dict):
        """
        strategy: dict mapping info_set_key -> {action_key: probability}
        """
        self.strategy = strategy

    @classmethod
    def from_file(cls, path: str):
        """Load a CFRAgent from a saved strategy JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(data['strategy'])

    def _lookup(self, info_key: str, actions: list) -> dict:
        """Look up strategy for info set. Uniform random if unseen."""
        if info_key in self.strategy:
            strat = self.strategy[info_key]
            # Filter to only available actions and renormalize
            filtered = {a: strat.get(a, 0.0) for a in actions}
            total = sum(filtered.values())
            if total > 0:
                return {a: v / total for a, v in filtered.items()}
        # Fallback: uniform
        u = 1.0 / len(actions)
        return {a: u for a in actions}

    def _sample(self, strat: dict) -> str:
        keys = list(strat.keys())
        weights = [strat[k] for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    # --- Agent interface ---

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        # Build info key from view
        # We need a lightweight state object for make_info_key
        state = _ViewState(view)
        ikey = make_info_key(state, view['player_id'], 'ACT')

        # Map legal actions to CFR keys
        action_map = {}
        for a in legal_actions:
            k = ACTION_KEYS[a.action_type]
            if k not in action_map:  # dedup (multiple targets → same key)
                action_map[k] = a

        akeys = list(action_map.keys())
        strat = self._lookup(ikey, akeys)
        chosen = self._sample(strat)
        return action_map[chosen]

    def choose_challenge(self, view: dict, claimer_idx: int,
                         claimed_card: Card) -> bool:
        state = _ViewState(view)
        ikey = make_info_key(state, view['player_id'], 'CHL', claimed_card.value)
        strat = self._lookup(ikey, ['yes', 'no'])
        return self._sample(strat) == 'yes'

    def choose_counteraction(self, view: dict, actor_idx: int,
                             action_type: ActionType,
                             blocking_cards: List[Card]) -> Optional[Card]:
        state = _ViewState(view)
        options = [f'c_{c.value}' for c in blocking_cards] + ['no']
        ikey = make_info_key(state, view['player_id'], 'CTR', action_type.value)
        strat = self._lookup(ikey, options)
        chosen = self._sample(strat)
        if chosen == 'no':
            return None
        name = chosen[2:]  # strip 'c_'
        return next(c for c in blocking_cards if c.value == name)

    def choose_card_to_lose(self, view: dict) -> int:
        # Heuristic: lose least valuable
        my_cards = view['my_cards']
        if len(my_cards) <= 1:
            return 0
        return min(range(len(my_cards)), key=lambda i: CARD_VALUE[my_cards[i]])

    def choose_exchange_cards(self, view: dict, all_cards: List[Card],
                              num_to_keep: int) -> List[int]:
        # Heuristic: keep most valuable
        ranked = sorted(range(len(all_cards)),
                        key=lambda i: -CARD_VALUE[all_cards[i]])
        return sorted(ranked[:num_to_keep])


class _ViewState:
    """
    Lightweight adapter so make_info_key can work with
    the view dict that agents receive (instead of full GameState).
    """

    def __init__(self, view: dict):
        self.players = []
        # Build minimal player objects
        me = _MiniPlayer(
            view['player_id'], view['my_cards'], view['my_coins'], [])
        self.players.append(me)

        for opp in view['opponents']:
            p = _MiniPlayer(
                opp['player_id'], [None] * opp['influence_count'],
                opp['coins'], opp['revealed'])
            self.players.append(p)

        # Sort by player_id so indexing works
        self.players.sort(key=lambda p: p.player_id)


class _MiniPlayer:
    """Minimal player object for info set key generation."""

    def __init__(self, player_id, cards, coins, revealed):
        self.player_id = player_id
        self.cards = cards
        self.coins = coins
        self.revealed = revealed

    @property
    def influence_count(self):
        return len(self.cards)

    @property
    def alive(self):
        return len(self.cards) > 0

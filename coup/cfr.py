"""
Outcome Sampling Monte Carlo CFR (OS-MCCFR) for 2-player Coup.

Each iteration samples a SINGLE path through the game tree.
At each decision point, one action is sampled according to the current strategy
(epsilon-greedy for exploration). Regrets are updated with importance weights.

This is O(turns_per_game) per iteration, making it tractable for fast training.

References:
  Lanctot et al., "Monte Carlo Sampling for Regret Minimization in
  Extensive Games", NeurIPS 2009.
"""

import json
import math
import random
from collections import defaultdict
from typing import Optional, Tuple

from .game import (
    Card, ActionType, Action, GameState,
    ACTION_CHARACTER, COUNTERABLE_BY, ANYONE_CAN_BLOCK,
)
from .info_set import make_info_key


# Card value for heuristic sub-decisions (exchange, card-to-lose)
CARD_VALUE = {
    Card.DUKE: 5,
    Card.ASSASSIN: 4,
    Card.CAPTAIN: 3,
    Card.AMBASSADOR: 2,
    Card.CONTESSA: 1,
}

# String keys for actions
ACTION_KEYS = {
    ActionType.INCOME: 'income',
    ActionType.FOREIGN_AID: 'foreign_aid',
    ActionType.COUP: 'coup',
    ActionType.TAX: 'tax',
    ActionType.ASSASSINATE: 'assassinate',
    ActionType.STEAL: 'steal',
    ActionType.EXCHANGE: 'exchange',
}
KEY_TO_ACTION_TYPE = {v: k for k, v in ACTION_KEYS.items()}

EPSILON = 0.05   # Exploration: probability of sampling uniformly instead of on-strategy
MAX_TURNS = 100  # Safety cutoff per traversal


class OSMCCFRTrainer:
    """Outcome Sampling MCCFR trainer for 2-player Coup."""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    # -----------------------------------------------------------------
    # Strategy
    # -----------------------------------------------------------------

    def get_strategy(self, info_key: str, actions: list) -> dict:
        """Regret matching."""
        regrets = self.regret_sum[info_key]
        pos_sum = sum(max(0.0, regrets[a]) for a in actions)
        if pos_sum > 0:
            return {a: max(0.0, regrets[a]) / pos_sum for a in actions}
        u = 1.0 / len(actions)
        return {a: u for a in actions}

    def get_average_strategy(self, info_key: str, actions: list) -> dict:
        """Average strategy — what we deploy after training."""
        ss = self.strategy_sum[info_key]
        total = sum(ss.get(a, 0.0) for a in actions)
        if total > 0:
            return {a: ss.get(a, 0.0) / total for a in actions}
        u = 1.0 / len(actions)
        return {a: u for a in actions}

    def _epsilon_sample(self, info_key: str, actions: list) -> Tuple[str, float]:
        """
        Epsilon-greedy sample. Returns (chosen_action, probability_of_sampling).
        With prob EPSILON: uniform over all actions.
        With prob 1-EPSILON: on-strategy sample.
        """
        strat = self.get_strategy(info_key, actions)
        n = len(actions)
        # Blend: epsilon * uniform + (1-epsilon) * strategy
        blend = {a: EPSILON / n + (1 - EPSILON) * strat[a] for a in actions}
        total = sum(blend.values())
        blend = {a: blend[a] / total for a in actions}

        chosen = random.choices(list(blend.keys()),
                                weights=list(blend.values()), k=1)[0]
        return chosen, blend[chosen]

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def train(self, num_iterations: int, print_every: int = 10000):
        for _ in range(num_iterations):
            state = GameState(num_players=2)
            for traverser in range(2):
                self._traverse(state.clone(), traverser, 1.0, 1.0)
            self.iterations += 1
            if self.iterations % print_every == 0:
                n = len(self.regret_sum)
                print(f"  Iter {self.iterations:>7}: {n} info sets")

    # -----------------------------------------------------------------
    # OS-MCCFR traversal
    # -----------------------------------------------------------------

    def _traverse(self, state: GameState, traverser: int,
                  my_reach: float, opp_reach: float) -> float:
        """
        Traverse one sampled path.
        my_reach:  product of traverser's action probs on this path so far
        opp_reach: product of opponent's action probs on this path so far
        Returns the sampled utility for the traverser.
        """
        if state.game_over:
            return self._utility(state, traverser)
        if state.turn_number >= MAX_TURNS:
            return 0.0

        active = state.current_player_idx
        if not state.players[active].alive:
            state.advance_to_next_alive()
            return self._traverse(state, traverser, my_reach, opp_reach)

        state.turn_number += 1
        return self._decide_action(state, traverser, my_reach, opp_reach)

    def _utility(self, state: GameState, traverser: int) -> float:
        w = state.winner
        if w == traverser:
            return 1.0
        elif w is not None:
            return -1.0
        return 0.0

    def _advance(self, state: GameState, traverser: int,
                 my_reach: float, opp_reach: float) -> float:
        if state.game_over:
            return self._utility(state, traverser)
        state.advance_to_next_alive()
        return self._traverse(state, traverser, my_reach, opp_reach)

    # -----------------------------------------------------------------
    # Decision helpers (sample + update)
    # -----------------------------------------------------------------

    def _make_decision(self, state, traverser, player_idx, decision_type,
                       actions, context, my_reach, opp_reach):
        """
        Generic decision: sample one action, update regrets/strategy.
        Returns (chosen_action, new_my_reach, new_opp_reach).
        """
        ikey = make_info_key(state, player_idx, decision_type, context)
        strat = self.get_strategy(ikey, actions)

        chosen, sample_prob = self._epsilon_sample(ikey, actions)

        if player_idx == traverser:
            # Accumulate strategy sum weighted by opponent reach
            for a in actions:
                self.strategy_sum[ikey][a] += opp_reach * strat[a]
            new_my = my_reach * strat[chosen]
            new_opp = opp_reach
        else:
            for a in actions:
                self.strategy_sum[ikey][a] += my_reach * strat[a]
            new_my = my_reach
            new_opp = opp_reach * strat[chosen]

        return ikey, strat, chosen, sample_prob, new_my, new_opp

    def _update_regret(self, ikey, actions, strat, chosen,
                       utility, sample_prob, player_idx, traverser,
                       my_reach, opp_reach):
        """Update regrets after sampling."""
        if player_idx != traverser:
            return

        # Importance weight: 1/sample_prob of chosen action
        w = 1.0 / sample_prob if sample_prob > 1e-9 else 0.0

        # Counterfactual regret for each action
        for a in actions:
            if a == chosen:
                self.regret_sum[ikey][a] += opp_reach * w * utility * (1 - strat[a])
            else:
                self.regret_sum[ikey][a] -= opp_reach * w * utility * strat[a]

    # -----------------------------------------------------------------
    # Game decision points
    # -----------------------------------------------------------------

    def _decide_action(self, state, traverser, my_reach, opp_reach):
        """Active player chooses an action."""
        active = state.current_player_idx
        opp = 1 - active
        akeys = self._legal_action_keys(state, active)

        ikey, strat, chosen, sprob, new_my, new_opp = self._make_decision(
            state, traverser, active, 'ACT', akeys, '', my_reach, opp_reach)

        action = self._make_action(chosen, active, opp)
        utility = self._after_action(state, traverser, action, new_my, new_opp)

        self._update_regret(ikey, akeys, strat, chosen, utility,
                            sprob, active, traverser, my_reach, opp_reach)
        return utility

    def _after_action(self, state, traverser, action, my_reach, opp_reach):
        """Apply action cost and route to next phase."""
        player = state.players[action.player_idx]

        if action.action_type == ActionType.COUP:
            player.coins -= 7
        elif action.action_type == ActionType.ASSASSINATE:
            player.coins -= 3

        if action.action_type == ActionType.INCOME:
            player.coins += 1
            return self._advance(state, traverser, my_reach, opp_reach)

        if action.action_type == ActionType.COUP:
            self._auto_lose(state, action.target_idx)
            return self._advance(state, traverser, my_reach, opp_reach)

        if action.action_type in ACTION_CHARACTER:
            return self._decide_challenge(state, traverser, action,
                                          my_reach, opp_reach)

        return self._decide_counter(state, traverser, action, my_reach, opp_reach)

    def _decide_challenge(self, state, traverser, action, my_reach, opp_reach):
        """Opponent decides whether to challenge."""
        actor = action.player_idx
        opp = 1 - actor
        claimed = ACTION_CHARACTER[action.action_type]

        if not state.players[opp].alive:
            return self._decide_counter(state, traverser, action, my_reach, opp_reach)

        ctx = claimed.value
        ikey, strat, chosen, sprob, new_my, new_opp = self._make_decision(
            state, traverser, opp, 'CHL', ['yes', 'no'], ctx, my_reach, opp_reach)

        if chosen == 'yes':
            utility = self._resolve_challenge(
                state, traverser, action, actor, opp, claimed,
                is_counter=False, my_reach=new_my, opp_reach=new_opp)
        else:
            utility = self._decide_counter(
                state, traverser, action, new_my, new_opp)

        self._update_regret(ikey, ['yes', 'no'], strat, chosen, utility,
                            sprob, opp, traverser, my_reach, opp_reach)
        return utility

    def _resolve_challenge(self, state, traverser, action,
                           claimer_idx, challenger_idx, claimed_card,
                           is_counter, my_reach, opp_reach):
        """Resolve the challenge — no decision, pure game logic."""
        claimer = state.players[claimer_idx]

        if claimer.has_card(claimed_card):
            # Challenge FAILED
            claimer.cards.remove(claimed_card)
            state.court_deck.append(claimed_card)
            random.shuffle(state.court_deck)
            if state.court_deck:
                claimer.cards.append(state.court_deck.pop())
            self._auto_lose(state, challenger_idx)
            if state.game_over:
                return self._utility(state, traverser)
            if is_counter:
                return self._advance(state, traverser, my_reach, opp_reach)
            return self._decide_counter(state, traverser, action, my_reach, opp_reach)
        else:
            # Challenge SUCCEEDED
            self._auto_lose(state, claimer_idx)
            if state.game_over:
                return self._utility(state, traverser)
            if is_counter:
                return self._resolve_action(state, traverser, action, my_reach, opp_reach)
            return self._advance(state, traverser, my_reach, opp_reach)

    def _decide_counter(self, state, traverser, action, my_reach, opp_reach):
        """Eligible player decides whether to block."""
        if action.action_type not in COUNTERABLE_BY:
            return self._resolve_action(state, traverser, action, my_reach, opp_reach)

        blocking_cards = COUNTERABLE_BY[action.action_type]
        if action.action_type in ANYONE_CAN_BLOCK:
            blocker = 1 - action.player_idx
        else:
            blocker = action.target_idx

        if blocker is None or not state.players[blocker].alive:
            return self._resolve_action(state, traverser, action, my_reach, opp_reach)

        options = [f'c_{c.value}' for c in blocking_cards] + ['no']
        ctx = action.action_type.value

        ikey, strat, chosen, sprob, new_my, new_opp = self._make_decision(
            state, traverser, blocker, 'CTR', options, ctx, my_reach, opp_reach)

        if chosen == 'no':
            utility = self._resolve_action(state, traverser, action, new_my, new_opp)
        else:
            bc = self._block_card(chosen, blocking_cards)
            utility = self._decide_challenge_counter(
                state, traverser, action, blocker, bc, new_my, new_opp)

        self._update_regret(ikey, options, strat, chosen, utility,
                            sprob, blocker, traverser, my_reach, opp_reach)
        return utility

    def _decide_challenge_counter(self, state, traverser, action,
                                  blocker_idx, block_card, my_reach, opp_reach):
        """Actor decides whether to challenge the counter."""
        actor = action.player_idx
        if not state.players[actor].alive:
            return self._advance(state, traverser, my_reach, opp_reach)

        options = ['yes', 'no']
        ctx = block_card.value

        ikey, strat, chosen, sprob, new_my, new_opp = self._make_decision(
            state, traverser, actor, 'CC', options, ctx, my_reach, opp_reach)

        if chosen == 'yes':
            utility = self._resolve_challenge(
                state, traverser, action, blocker_idx, actor,
                block_card, is_counter=True, my_reach=new_my, opp_reach=new_opp)
        else:
            utility = self._advance(state, traverser, new_my, new_opp)

        self._update_regret(ikey, options, strat, chosen, utility,
                            sprob, actor, traverser, my_reach, opp_reach)
        return utility

    def _resolve_action(self, state, traverser, action, my_reach, opp_reach):
        """Apply action effects."""
        player = state.players[action.player_idx]

        if action.action_type == ActionType.FOREIGN_AID:
            player.coins += 2
        elif action.action_type == ActionType.TAX:
            player.coins += 3
        elif action.action_type == ActionType.ASSASSINATE:
            if (action.target_idx is not None
                    and state.players[action.target_idx].alive):
                self._auto_lose(state, action.target_idx)
        elif action.action_type == ActionType.STEAL:
            target = state.players[action.target_idx]
            stolen = min(2, target.coins)
            target.coins -= stolen
            player.coins += stolen
        elif action.action_type == ActionType.EXCHANGE:
            self._resolve_exchange(state, action.player_idx)

        return self._advance(state, traverser, my_reach, opp_reach)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _auto_lose(self, state, player_idx):
        """Heuristic: lose least valuable card."""
        p = state.players[player_idx]
        if not p.alive or not p.cards:
            return
        idx = min(range(len(p.cards)), key=lambda i: CARD_VALUE[p.cards[i]])
        p.lose_influence(idx)

    def _legal_action_keys(self, state, pidx):
        p = state.players[pidx]
        if p.coins >= 10:
            return ['coup']
        keys = ['income', 'foreign_aid', 'tax', 'exchange']
        if p.coins >= 7:
            keys.append('coup')
        if p.coins >= 3:
            keys.append('assassinate')
        opp = state.players[1 - pidx]
        if opp.alive:
            keys.append('steal')
        return keys

    def _make_action(self, key, pidx, opp):
        at = KEY_TO_ACTION_TYPE[key]
        target = opp if at in (ActionType.COUP, ActionType.ASSASSINATE,
                               ActionType.STEAL) else None
        return Action(at, pidx, target)

    def _block_card(self, opt, blocking_cards):
        name = opt[2:]
        return next(c for c in blocking_cards if c.value == name)

    def _resolve_exchange(self, state, player_idx):
        p = state.players[player_idx]
        drawn = []
        for _ in range(min(2, len(state.court_deck))):
            drawn.append(state.court_deck.pop())
        if not drawn:
            return
        all_cards = list(p.cards) + drawn
        n_keep = p.influence_count
        ranked = sorted(range(len(all_cards)),
                        key=lambda i: -CARD_VALUE[all_cards[i]])
        keep_idx = sorted(ranked[:n_keep])
        p.cards = [all_cards[i] for i in keep_idx]
        returned = [all_cards[i] for i in range(len(all_cards))
                    if i not in keep_idx]
        state.court_deck.extend(returned)
        random.shuffle(state.court_deck)

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, path: str):
        data = {}
        for ikey in self.strategy_sum:
            actions = list(self.strategy_sum[ikey].keys())
            if not actions:
                continue
            avg = self.get_average_strategy(ikey, actions)
            data[ikey] = avg
        with open(path, 'w') as f:
            json.dump({
                'iterations': self.iterations,
                'num_info_sets': len(data),
                'strategy': data,
            }, f, indent=2)
        print(f"  Saved {len(data)} info sets to {path}")

    @staticmethod
    def load_strategy(path: str) -> dict:
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded strategy: {data['iterations']} iterations, "
              f"{data['num_info_sets']} info sets")
        return data['strategy']

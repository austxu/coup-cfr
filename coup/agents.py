"""
Agent interface and implementations for Coup.

All agents receive a 'view' dict (from GameState.get_player_view) that contains
only the information observable by that player — no peeking at opponents' cards.
"""

import random
from typing import List, Optional

from .game import Card, ActionType, Action


class Agent:
    """
    Base class for Coup agents. Subclass and implement all methods.
    """

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        """Choose an action from the list of legal actions."""
        raise NotImplementedError

    def choose_challenge(self, view: dict, claimer_idx: int,
                         claimed_card: Card) -> bool:
        """Decide whether to challenge a player's character claim."""
        raise NotImplementedError

    def choose_counteraction(self, view: dict, actor_idx: int,
                             action_type: ActionType,
                             blocking_cards: List[Card]) -> Optional[Card]:
        """
        Decide whether to block an action.
        Return a Card to block with, or None to allow the action.
        """
        raise NotImplementedError

    def choose_card_to_lose(self, view: dict) -> int:
        """Choose which card index to reveal when losing influence."""
        raise NotImplementedError

    def choose_exchange_cards(self, view: dict, all_cards: List[Card],
                              num_to_keep: int) -> List[int]:
        """
        During an Exchange, choose which card indices to keep.
        all_cards = your current cards + drawn cards.
        Return exactly num_to_keep indices.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    Makes all decisions uniformly at random from legal options.
    Challenge and block probabilities are tuned so games play out naturally.
    """

    def __init__(self, challenge_rate: float = 0.15,
                 block_rate: float = 0.25):
        self.challenge_rate = challenge_rate
        self.block_rate = block_rate

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        return random.choice(legal_actions)

    def choose_challenge(self, view: dict, claimer_idx: int,
                         claimed_card: Card) -> bool:
        return random.random() < self.challenge_rate

    def choose_counteraction(self, view: dict, actor_idx: int,
                             action_type: ActionType,
                             blocking_cards: List[Card]) -> Optional[Card]:
        if random.random() < self.block_rate:
            return random.choice(blocking_cards)
        return None

    def choose_card_to_lose(self, view: dict) -> int:
        return random.randint(0, len(view['my_cards']) - 1)

    def choose_exchange_cards(self, view: dict, all_cards: List[Card],
                              num_to_keep: int) -> List[int]:
        indices = list(range(len(all_cards)))
        random.shuffle(indices)
        return sorted(indices[:num_to_keep])


# Card value ranking for the heuristic agent (higher = more valuable to keep)
CARD_VALUE = {
    Card.DUKE: 5,       # Tax is the best economy action
    Card.ASSASSIN: 4,   # Cheap kills
    Card.CAPTAIN: 3,    # Steal is solid + blocks steals
    Card.AMBASSADOR: 2, # Exchange is situational, blocks steals
    Card.CONTESSA: 1,   # Only useful for blocking assassination
}


class HeuristicAgent(Agent):
    """
    Rule-based agent that plays a solid, conservative strategy:
    - Uses character actions only when holding the correct card (no bluffing)
    - Always blocks when holding the right card
    - Challenges when holding 2+ copies of the claimed card
    - Keeps the most valuable cards
    """

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        my_cards = view['my_cards']
        my_coins = view['my_coins']
        opponents = [o for o in view['opponents'] if o['alive']]

        # Sort opponents: prefer targeting those with fewer influence (finish them)
        # then those with more coins (bigger threat)
        opponents.sort(key=lambda o: (o['influence_count'], -o['coins']))

        # Helper: find a specific action type (optionally with target)
        def find_action(action_type, target_id=None):
            for a in legal_actions:
                if a.action_type == action_type:
                    if target_id is None or a.target_idx == target_id:
                        return a
            return None

        # --- Priority-based action selection ---

        # 1. Forced coup at 10+ coins (already enforced by legal_actions)
        if my_coins >= 10:
            best_target = opponents[0]['player_id']
            return find_action(ActionType.COUP, best_target) or legal_actions[0]

        # 2. Assassinate if we have Assassin and enough coins
        if Card.ASSASSIN in my_cards and my_coins >= 3:
            best_target = opponents[0]['player_id']
            action = find_action(ActionType.ASSASSINATE, best_target)
            if action:
                return action

        # 3. Coup if we have 7+ coins and no Assassin
        if my_coins >= 7:
            best_target = opponents[0]['player_id']
            action = find_action(ActionType.COUP, best_target)
            if action:
                return action

        # 4. Tax if we have Duke (safe, strong economy)
        if Card.DUKE in my_cards:
            action = find_action(ActionType.TAX)
            if action:
                return action

        # 5. Steal if we have Captain and a good target
        if Card.CAPTAIN in my_cards:
            # Steal from the richest opponent
            rich_opponents = sorted(opponents, key=lambda o: -o['coins'])
            for opp in rich_opponents:
                if opp['coins'] >= 1:
                    action = find_action(ActionType.STEAL, opp['player_id'])
                    if action:
                        return action

        # 6. Exchange if we have Ambassador (try to improve hand)
        if Card.AMBASSADOR in my_cards and len(my_cards) >= 2:
            # Only exchange if our hand could improve (e.g. holding Contessa)
            if Card.CONTESSA in my_cards:
                action = find_action(ActionType.EXCHANGE)
                if action:
                    return action

        # 7. Foreign Aid as fallback (2 coins, but blockable)
        action = find_action(ActionType.FOREIGN_AID)
        if action:
            return action

        # 8. Income as safe fallback
        action = find_action(ActionType.INCOME)
        if action:
            return action

        return random.choice(legal_actions)

    def choose_challenge(self, view: dict, claimer_idx: int,
                         claimed_card: Card) -> bool:
        my_cards = view['my_cards']

        # Count how many of the claimed card WE hold
        my_count = my_cards.count(claimed_card)

        # Count how many of the claimed card are revealed (dead)
        revealed_count = 0
        for opp in view['opponents']:
            revealed_count += opp['revealed'].count(claimed_card)

        # Total accounted for (in our hand + revealed)
        accounted = my_count + revealed_count

        # There are 3 of each card. If we can see all 3, they're definitely lying
        if accounted >= 3:
            return True

        # If we can see 2, there's only 1 in the deck — risky claim, challenge
        if accounted >= 2:
            return True

        # Otherwise, don't challenge (conservative)
        return False

    def choose_counteraction(self, view: dict, actor_idx: int,
                             action_type: ActionType,
                             blocking_cards: List[Card]) -> Optional[Card]:
        my_cards = view['my_cards']

        # Block if we actually have a blocking card (honest blocking)
        for card in blocking_cards:
            if card in my_cards:
                return card

        # Don't bluff-block
        return None

    def choose_card_to_lose(self, view: dict) -> int:
        my_cards = view['my_cards']
        if len(my_cards) <= 1:
            return 0

        # Lose the least valuable card
        min_val = float('inf')
        min_idx = 0
        for i, card in enumerate(my_cards):
            if CARD_VALUE[card] < min_val:
                min_val = CARD_VALUE[card]
                min_idx = i
        return min_idx

    def choose_exchange_cards(self, view: dict, all_cards: List[Card],
                              num_to_keep: int) -> List[int]:
        # Rank all cards by value and keep the best ones
        indexed = [(i, card) for i, card in enumerate(all_cards)]
        indexed.sort(key=lambda x: -CARD_VALUE[x[1]])
        return sorted([idx for idx, _ in indexed[:num_to_keep]])

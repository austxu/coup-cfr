import random
from typing import List, Optional

from .game import (
    Card, ActionType, Action, ACTION_CHARACTER, COUNTERABLE_BY, ANYONE_CAN_BLOCK
)
from .agents import Agent


class ZooAgent(Agent):
    """
    A parameterized bot for Population-Based Training.
    By randomizing its parameters, we can generate an infinite variety
    of opponent playstyles (Aggressive, Turtle, Honest, Maniac, etc.)
    """

    def __init__(self, 
                 bluff_rate: float = 0.5,
                 challenge_rate: float = 0.5,
                 block_rate: float = 0.5,
                 income_preference: float = 0.5,
                 favored_claims: List[Card] = None):
        
        self.bluff_rate = bluff_rate
        self.challenge_rate = challenge_rate
        self.block_rate = block_rate
        self.income_preference = income_preference
        
        if favored_claims is None:
            self.favored_claims = [Card.DUKE, Card.CAPTAIN, Card.ASSASSIN]
        else:
            self.favored_claims = favored_claims

    @classmethod
    def random_profile(cls) -> 'ZooAgent':
        """Generate a completely random player profile."""
        favs = list(Card)
        random.shuffle(favs)
        return cls(
            bluff_rate=random.random(),
            challenge_rate=random.random(),
            block_rate=random.random(),
            income_preference=random.random(),
            favored_claims=favs[:random.randint(1, 3)]
        )

    def _has_card(self, view: dict, card: Card) -> bool:
        return card in view['my_cards']

    def choose_action(self, view: dict, legal_actions: List[Action]) -> Action:
        coins = view['my_coins']
        actions = {a.action_type: a for a in legal_actions}
        
        if coins >= 10 and ActionType.COUP in actions:
            return actions[ActionType.COUP]

        # Sometimes they just want safe income based on preference
        safe_actions = []
        if ActionType.INCOME in actions: safe_actions.append(actions[ActionType.INCOME])
        if ActionType.FOREIGN_AID in actions: safe_actions.append(actions[ActionType.FOREIGN_AID])
        
        if safe_actions and random.random() < self.income_preference:
            return random.choice(safe_actions)

        # Build a list of favored aggressive actions
        candidates = []
        if ActionType.ASSASSINATE in actions:
            if self._has_card(view, Card.ASSASSIN) or random.random() < self.bluff_rate:
                candidates.append(actions[ActionType.ASSASSINATE])
                
        if ActionType.TAX in actions:
            if self._has_card(view, Card.DUKE) or random.random() < self.bluff_rate:
                candidates.append(actions[ActionType.TAX])
                
        if ActionType.STEAL in actions:
            if self._has_card(view, Card.CAPTAIN) or random.random() < self.bluff_rate:
                candidates.append(actions[ActionType.STEAL])
                
        if ActionType.EXCHANGE in actions:
            if self._has_card(view, Card.AMBASSADOR) or random.random() < self.bluff_rate:
                candidates.append(actions[ActionType.EXCHANGE])

        if ActionType.COUP in actions:
            candidates.append(actions[ActionType.COUP])

        if candidates:
            # Bias selection towards their favored claims if bluffing/using character actions
            for a in list(candidates):
                if a.action_type in ACTION_CHARACTER:
                    c = ACTION_CHARACTER[a.action_type]
                    if c in self.favored_claims:
                        # Artificially increase weight of favored actions
                        candidates.append(a) 
                        candidates.append(a)
                        
            return random.choice(candidates)

        # Fallback to a safe action
        return random.choice(safe_actions) if safe_actions else random.choice(legal_actions)

    def choose_challenge(self, view: dict, claimer_idx: int, claimed_card: Card) -> bool:
        # Don't challenge if we hold both copies of the card! ( guaranteed win )
        count = view['my_cards'].count(claimed_card)
        if count == 2:
            return True
            
        # Otherwise, challenge based on slider
        return random.random() < self.challenge_rate

    def choose_counteraction(self, view: dict, actor_idx: int, action_type: ActionType, blocking_cards: List[Card]) -> Optional[Card]:
        # Do we actually have a block card?
        for c in blocking_cards:
            if self._has_card(view, c):
                return c
                
        # If not, do we bluff a block?
        if random.random() < self.block_rate:
            # Pick a preferred block card if possible
            for c in blocking_cards:
                if c in self.favored_claims:
                    return c
            return random.choice(blocking_cards)
            
        return None

    def choose_challenge_counter(self, view: dict, blocker_idx: int, blocking_card: Card) -> bool:
        return random.random() < self.challenge_rate

    def choose_card_to_lose(self, view: dict) -> int:
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        cards = view['my_cards']
        if len(cards) == 1:
            return 0
        return 0 if val[cards[0]] < val[cards[1]] else 1

    def choose_exchange_cards(self, view: dict, all_cards: List[Card], num_to_keep: int) -> List[int]:
        val = {Card.DUKE: 5, Card.ASSASSIN: 4, Card.CAPTAIN: 3, Card.AMBASSADOR: 2, Card.CONTESSA: 1}
        ranked = sorted(range(len(all_cards)), key=lambda i: -val[all_cards[i]])
        return sorted(ranked[:num_to_keep])

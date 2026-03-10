"""
Core game engine for Coup.

Implements all game rules: actions, challenges, counteractions, and the full turn flow.
"""

import copy
import random
from enum import Enum
from typing import List, Optional, Tuple


# =============================================================================
# Enums & Constants
# =============================================================================

class Card(Enum):
    DUKE = "Duke"
    ASSASSIN = "Assassin"
    CAPTAIN = "Captain"
    AMBASSADOR = "Ambassador"
    CONTESSA = "Contessa"


class ActionType(Enum):
    INCOME = "Income"
    FOREIGN_AID = "Foreign Aid"
    COUP = "Coup"
    TAX = "Tax"
    ASSASSINATE = "Assassinate"
    STEAL = "Steal"
    EXCHANGE = "Exchange"


# Which character is required for each character action
ACTION_CHARACTER = {
    ActionType.TAX: Card.DUKE,
    ActionType.ASSASSINATE: Card.ASSASSIN,
    ActionType.STEAL: Card.CAPTAIN,
    ActionType.EXCHANGE: Card.AMBASSADOR,
}

# Which characters can block each action
COUNTERABLE_BY = {
    ActionType.FOREIGN_AID: [Card.DUKE],
    ActionType.ASSASSINATE: [Card.CONTESSA],
    ActionType.STEAL: [Card.CAPTAIN, Card.AMBASSADOR],
}

# Actions where ANY player can block (vs only the target)
ANYONE_CAN_BLOCK = {ActionType.FOREIGN_AID}


# =============================================================================
# Data Classes
# =============================================================================

class Action:
    """Represents a player's chosen action."""

    def __init__(self, action_type: ActionType, player_idx: int,
                 target_idx: Optional[int] = None):
        self.action_type = action_type
        self.player_idx = player_idx
        self.target_idx = target_idx

    def __repr__(self):
        target = f" -> {self.state.names[self.target_idx]}" if self.target_idx is not None else ""
        return f"Action({self.action_type.value}{target})"


class Player:
    """A player in the game with coins and influence cards."""

    def __init__(self, player_id: int, cards: List[Card]):
        self.player_id = player_id
        self.coins = 2
        self.cards = list(cards)       # hidden cards (alive influence)
        self.revealed = []             # face-up cards (dead influence)
        self.claimed_cards = set()     # cards claimed during this game
        self.caught_bluff_count = 0    # number of times this player lost a challenge

    @property
    def alive(self) -> bool:
        return len(self.cards) > 0

    @property
    def influence_count(self) -> int:
        return len(self.cards)

    def has_card(self, card: Card) -> bool:
        return card in self.cards

    def lose_influence(self, card_idx: int) -> Card:
        """Remove a card from hand and reveal it."""
        card = self.cards.pop(card_idx)
        self.revealed.append(card)
        return card

    def __repr__(self):
        return (f"Player({self.player_id}, coins={self.coins}, "
                f"cards={[c.value for c in self.cards]}, "
                f"revealed={[c.value for c in self.revealed]})")

    def _clone(self) -> 'Player':
        """Fast shallow clone (cards/revealed are lists of enums)."""
        p = Player.__new__(Player)
        p.player_id = self.player_id
        p.coins = self.coins
        p.cards = list(self.cards)
        p.revealed = list(self.revealed)
        p.claimed_cards = set(self.claimed_cards)
        p.caught_bluff_count = self.caught_bluff_count
        return p


# =============================================================================
# Game State
# =============================================================================

class GameState:
    """
    Full game state. Agents receive a filtered PlayerView — they should
    never access this object directly.
    """

    def __init__(self, num_players: int, names: List[str] = None):
        assert 2 <= num_players <= 6, "Coup supports 2-6 players"
        self.names = names or [f"{self.state.names[i]}" for i in range(num_players)]

        # Create and shuffle the 15-card court deck
        self.court_deck: List[Card] = []
        for card in Card:
            self.court_deck.extend([card] * 3)
        random.shuffle(self.court_deck)

        # Deal 2 cards to each player
        self.players: List[Player] = []
        for i in range(num_players):
            cards = [self.court_deck.pop(), self.court_deck.pop()]
            self.players.append(Player(i, cards))

        self.current_player_idx = 0
        self.turn_number = 0
        self.action_history: List[dict] = []

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def num_alive(self) -> int:
        return sum(1 for p in self.players if p.alive)

    @property
    def game_over(self) -> bool:
        return self.num_alive <= 1

    @property
    def winner(self) -> Optional[int]:
        if not self.game_over:
            return None
        for p in self.players:
            if p.alive:
                return p.player_id
        return None

    def alive_player_ids(self) -> List[int]:
        return [p.player_id for p in self.players if p.alive]

    def advance_to_next_alive(self):
        """Move current_player_idx to the next alive player."""
        n = len(self.players)
        idx = (self.current_player_idx + 1) % n
        while not self.players[idx].alive:
            idx = (idx + 1) % n
        self.current_player_idx = idx

    def total_cards_check(self):
        """Assert that all 15 cards are accounted for."""
        total = len(self.court_deck)
        for p in self.players:
            total += len(p.cards) + len(p.revealed)
        assert total == 15, f"Card integrity violation: {total} cards (expected 15)"

    def clone(self) -> 'GameState':
        """Fast clone for CFR tree traversal (no deepcopy)."""
        gs = GameState.__new__(GameState)
        gs.names = self.names
        gs.court_deck = list(self.court_deck)
        gs.players = [p._clone() for p in self.players]
        gs.current_player_idx = self.current_player_idx
        gs.turn_number = self.turn_number
        gs.action_history = []  # don't copy history for performance
        return gs

    def get_player_view(self, player_idx: int) -> dict:
        """
        Create the observable game state for a specific player.
        This is what agents receive — no hidden information from other players.
        """
        player = self.players[player_idx]
        opponents = []
        for p in self.players:
            if p.player_id != player_idx:
                opponents.append({
                    'player_id': p.player_id,
                    'name': self.names[p.player_id],
                    'coins': p.coins,
                    'influence_count': p.influence_count,
                    'revealed': list(p.revealed),
                    'claimed_cards': list(p.claimed_cards),
                    'caught_bluff_count': p.caught_bluff_count,
                    'alive': p.alive,
                })
        return {
            'player_id': player_idx,
            'name': self.names[player_idx],
            'my_cards': list(player.cards),
            'my_revealed': list(player.revealed),
            'my_claimed_cards': list(player.claimed_cards),
            'my_caught_bluff_count': player.caught_bluff_count,
            'my_coins': player.coins,
            'opponents': opponents,
            'action_history': self.action_history,  # reference, not copy
            'turn_number': self.turn_number,
            'court_deck_size': len(self.court_deck),
        }


# =============================================================================
# Game Engine
# =============================================================================

class CoupGame:
    """
    Runs a complete game of Coup with the provided agents.
    Handles the full turn flow: action → challenge → counteraction → resolve.
    """

    def __init__(self, agents, num_players: int = 2, verbose: bool = False):
        assert len(agents) == num_players
        self.agents = agents
        names = [getattr(a, 'name', f"P{i}") for i, a in enumerate(agents)]
        self.state = GameState(num_players, names)

        self.verbose = verbose
        self.max_turns = 500  # safety limit

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # Legal actions
    # -------------------------------------------------------------------------

    def get_legal_actions(self, player_idx: int) -> List[Action]:
        """Return all legal actions for the given player."""
        player = self.state.players[player_idx]
        alive_opponents = [
            p.player_id for p in self.state.players
            if p.alive and p.player_id != player_idx
        ]
        actions: List[Action] = []

        # 10-coin rule: must coup
        if player.coins >= 10:
            for t in alive_opponents:
                actions.append(Action(ActionType.COUP, player_idx, t))
            return actions

        # General actions (always available)
        actions.append(Action(ActionType.INCOME, player_idx))
        actions.append(Action(ActionType.FOREIGN_AID, player_idx))

        if player.coins >= 7:
            for t in alive_opponents:
                actions.append(Action(ActionType.COUP, player_idx, t))

        # Character actions — can always *claim* them (bluffing is legal)
        actions.append(Action(ActionType.TAX, player_idx))
        actions.append(Action(ActionType.EXCHANGE, player_idx))

        if player.coins >= 3:
            for t in alive_opponents:
                actions.append(Action(ActionType.ASSASSINATE, player_idx, t))

        for t in alive_opponents:
            actions.append(Action(ActionType.STEAL, player_idx, t))

        return actions

    # -------------------------------------------------------------------------
    # Challenge resolution
    # -------------------------------------------------------------------------

    def resolve_challenge(self, claimer_idx: int, challenger_idx: int,
                          claimed_card: Card) -> bool:
        """
        Resolve a challenge.
        Returns True if the challenge SUCCEEDED (claimer was bluffing).
        """
        claimer = self.state.players[claimer_idx]
        claimer.claimed_cards.add(claimed_card)

        if claimer.has_card(claimed_card):
            # Challenge FAILED — claimer actually had the card
            self.log(f"    Challenge FAILED: {self.state.names[claimer_idx]} had {claimed_card.value}")

            # Claimer shuffles the revealed card back and draws a new one
            claimer.cards.remove(claimed_card)
            self.state.court_deck.append(claimed_card)
            random.shuffle(self.state.court_deck)
            new_card = self.state.court_deck.pop()
            claimer.cards.append(new_card)

            # Challenger loses influence
            self._player_loses_influence(challenger_idx)
            return False
        else:
            # Challenge SUCCEEDED — claimer was bluffing
            self.log(f"    Challenge SUCCEEDED: {self.state.names[claimer_idx]} was bluffing!")
            claimer.caught_bluff_count += 1
            self._player_loses_influence(claimer_idx)
            return True

    def _player_loses_influence(self, player_idx: int):
        """Force a player to choose and lose one influence card."""
        player = self.state.players[player_idx]
        if not player.alive:
            return

        if player.influence_count == 1:
            # No choice — must lose their only card
            card_idx = 0
        else:
            view = self.state.get_player_view(player_idx)
            card_idx = self.agents[player_idx].choose_card_to_lose(view)
            # Clamp to valid range
            card_idx = max(0, min(card_idx, player.influence_count - 1))

        lost_card = player.lose_influence(card_idx)
        self.log(f"    {self.state.names[player_idx]} loses influence: {lost_card.value}"
                 + (" (ELIMINATED)" if not player.alive else ""))

    # -------------------------------------------------------------------------
    # Challenge & counteraction queries
    # -------------------------------------------------------------------------

    def _ask_for_challenges(self, claimer_idx: int, claimed_card: Card) -> Optional[int]:
        """
        Ask each alive player (except the claimer) if they want to challenge.
        Goes clockwise from the claimer. Returns challenger idx or None.
        """
        n = self.state.num_players
        for offset in range(1, n):
            p_idx = (claimer_idx + offset) % n
            p = self.state.players[p_idx]
            if not p.alive:
                continue

            view = self.state.get_player_view(p_idx)
            if self.agents[p_idx].choose_challenge(view, claimer_idx, claimed_card):
                self.log(f"  {self.state.names[p_idx]} challenges {self.state.names[claimer_idx]}'s "
                         f"{claimed_card.value} claim!")
                return p_idx

        return None

    def _ask_for_counteraction(self, action: Action) -> Optional[Tuple[int, Card]]:
        """
        Ask eligible players if they want to counteract.
        Returns (blocker_idx, blocking_card) or None.
        """
        if action.action_type not in COUNTERABLE_BY:
            return None

        blocking_cards = COUNTERABLE_BY[action.action_type]

        # Determine who is eligible to block
        if action.action_type in ANYONE_CAN_BLOCK:
            eligible_ids = [
                p.player_id for p in self.state.players
                if p.alive and p.player_id != action.player_idx
            ]
        else:
            # Only the target can block
            if action.target_idx is None:
                return None
            target = self.state.players[action.target_idx]
            eligible_ids = [target.player_id] if target.alive else []

        for p_idx in eligible_ids:
            view = self.state.get_player_view(p_idx)
            result = self.agents[p_idx].choose_counteraction(
                view, action.player_idx, action.action_type, blocking_cards
            )
            if result is not None:
                self.log(f"  {self.state.names[p_idx]} blocks with {result.value}!")
                return (p_idx, result)

        return None

    # -------------------------------------------------------------------------
    # Action resolution
    # -------------------------------------------------------------------------

    def _resolve_action(self, action: Action):
        """Resolve the effect of an action (after it passes challenges/counters)."""
        player = self.state.players[action.player_idx]

        if action.action_type == ActionType.INCOME:
            player.coins += 1
            self.log(f"  -> {self.state.names[action.player_idx]} takes Income ({player.coins} coins)")

        elif action.action_type == ActionType.FOREIGN_AID:
            player.coins += 2
            self.log(f"  -> {self.state.names[action.player_idx]} takes Foreign Aid ({player.coins} coins)")

        elif action.action_type == ActionType.COUP:
            self.log(f"  -> {self.state.names[action.player_idx]} coups {self.state.names[action.target_idx]}")
            self._player_loses_influence(action.target_idx)

        elif action.action_type == ActionType.TAX:
            player.coins += 3
            self.log(f"  -> {self.state.names[action.player_idx]} takes Tax ({player.coins} coins)")

        elif action.action_type == ActionType.ASSASSINATE:
            target = self.state.players[action.target_idx]
            if target.alive:
                self.log(f"  -> {self.state.names[action.player_idx]} assassinates {self.state.names[action.target_idx]}")
                self._player_loses_influence(action.target_idx)

        elif action.action_type == ActionType.STEAL:
            target = self.state.players[action.target_idx]
            stolen = min(2, target.coins)
            target.coins -= stolen
            player.coins += stolen
            self.log(f"  -> {self.state.names[action.player_idx]} steals {stolen} from "
                     f"{self.state.names[action.target_idx]}")

        elif action.action_type == ActionType.EXCHANGE:
            self._resolve_exchange(action.player_idx)

    def _resolve_exchange(self, player_idx: int):
        """Handle the Ambassador exchange action."""
        player = self.state.players[player_idx]

        # Draw up to 2 cards from the court deck
        drawn = []
        for _ in range(min(2, len(self.state.court_deck))):
            drawn.append(self.state.court_deck.pop())

        if not drawn:
            self.log(f"  -> {self.state.names[player_idx]} exchanges but deck is empty")
            return

        # Player sees all available cards and chooses which to keep
        all_cards = list(player.cards) + drawn
        num_to_keep = player.influence_count

        view = self.state.get_player_view(player_idx)
        kept_indices = self.agents[player_idx].choose_exchange_cards(
            view, all_cards, num_to_keep
        )

        # Validate the selection
        if (kept_indices is None
                or len(kept_indices) != num_to_keep
                or any(i < 0 or i >= len(all_cards) for i in kept_indices)
                or len(set(kept_indices)) != num_to_keep):
            # Invalid choice — keep original cards as fallback
            kept_indices = list(range(num_to_keep))

        new_cards = [all_cards[i] for i in kept_indices]
        returned = [all_cards[i] for i in range(len(all_cards))
                    if i not in kept_indices]

        player.cards = new_cards
        self.state.court_deck.extend(returned)
        random.shuffle(self.state.court_deck)

        self.log(f"  -> {self.state.names[player_idx]} exchanges cards")

    # -------------------------------------------------------------------------
    # Turn flow
    # -------------------------------------------------------------------------

    def play_turn(self):
        """Execute one complete turn."""
        player_idx = self.state.current_player_idx
        player = self.state.players[player_idx]

        if not player.alive:
            self.state.advance_to_next_alive()
            return

        self.state.turn_number += 1
        self.log(f"\n--- Turn {self.state.turn_number}: {self.state.names[player_idx]} "
                 f"({player.coins} coins) ---")

        # 1. Agent chooses an action
        legal_actions = self.get_legal_actions(player_idx)
        view = self.state.get_player_view(player_idx)
        action = self.agents[player_idx].choose_action(view, legal_actions)

        self.log(f"  Action: {action.action_type.value}"
                 + (f" -> {self.state.names[action.target_idx]}"
                    if action.target_idx is not None else ""))

        # Record in history
        history_entry = {
            'turn': self.state.turn_number,
            'player': player_idx,
            'action': action.action_type,
            'target': action.target_idx,
            'was_blocked': False,
            'was_challenged': False,
            'challenge_won': False,
            'card_lost': False,
        }
        self.state.action_history.append(history_entry)

        # 2. Pay costs upfront (coins are lost even if action is blocked)
        if action.action_type == ActionType.COUP:
            player.coins -= 7
        elif action.action_type == ActionType.ASSASSINATE:
            player.coins -= 3

        # 3. Coup and Income are unchallengeable and unblockable
        if action.action_type in (ActionType.COUP, ActionType.INCOME):
            if action.action_type == ActionType.COUP:
                history_entry['card_lost'] = True
            self._resolve_action(action)
            self.state.total_cards_check()
            if not self.state.game_over:
                self.state.advance_to_next_alive()
            return

        # 4. Challenge phase (character actions only)
        action_blocked = False
        if action.action_type in ACTION_CHARACTER:
            claimed_card = ACTION_CHARACTER[action.action_type]
            challenger_idx = self._ask_for_challenges(player_idx, claimed_card)

            if challenger_idx is not None:
                self.state.action_history[-1]['was_challenged'] = True
                bluffing = self.resolve_challenge(
                    player_idx, challenger_idx, claimed_card
                )
                self.state.action_history[-1]['challenge_won'] = bluffing
                if bluffing:
                    # Claimer was bluffing → action fails entirely
                    action_blocked = True

                if self.state.game_over:
                    return

        if action_blocked:
            self.state.total_cards_check()
            if not self.state.game_over:
                self.state.advance_to_next_alive()
            return

        # If the target died during the challenge, skip the rest
        if (action.target_idx is not None
                and not self.state.players[action.target_idx].alive):
            self.state.total_cards_check()
            if not self.state.game_over:
                self.state.advance_to_next_alive()
            return

        # 5. Counteraction phase
        counter = self._ask_for_counteraction(action)
        if counter is not None:
            blocker_idx, blocking_card = counter

            # The counteraction can be challenged by anyone
            counter_challenger = self._ask_for_challenges(
                blocker_idx, blocking_card
            )

            if counter_challenger is not None:
                counter_was_bluff = self.resolve_challenge(
                    blocker_idx, counter_challenger, blocking_card
                )

                if self.state.game_over:
                    return

                if counter_was_bluff:
                    # Counter was a bluff → action goes through
                    if (action.target_idx is None
                            or self.state.players[action.target_idx].alive):
                        if action.action_type == ActionType.ASSASSINATE:
                            self.state.action_history[-1]['card_lost'] = True
                        self._resolve_action(action)
                else:
                    # Counter was legit → action is blocked
                    self.log(f"  Action blocked by {blocking_card.value}")
                    self.state.action_history[-1]['was_blocked'] = True
            else:
                # Counter not challenged → action is blocked
                self.log(f"  Action blocked (unchallenged)")
                self.state.action_history[-1]['was_blocked'] = True
        else:
            # No counteraction → resolve the action
            if action.action_type == ActionType.ASSASSINATE:
                self.state.action_history[-1]['card_lost'] = True
            self._resolve_action(action)

        self.state.total_cards_check()
        if not self.state.game_over:
            self.state.advance_to_next_alive()

    # -------------------------------------------------------------------------
    # Full game
    # -------------------------------------------------------------------------

    def play_game(self) -> Optional[int]:
        """
        Play a full game of Coup.
        Returns the winner's player_id, or None if max turns reached.
        """
        while not self.state.game_over and self.state.turn_number < self.max_turns:
            self.play_turn()

        winner = self.state.winner
        if winner is not None:
            self.log(f"\n{'='*40}")
            self.log(f"  {self.state.names[winner]} wins in {self.state.turn_number} turns!")
            self.log(f"{'='*40}")
        else:
            self.log(f"\n  Draw (max turns reached)")

        return winner

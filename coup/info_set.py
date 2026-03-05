"""
Information set representation for CFR.

An information set groups all game states that look identical to a player.
The key encodes what the player can observe, used to look up CFR strategies.
"""

from .game import Card


# Single-char abbreviations for compact info set keys
CARD_ABBREV = {
    Card.DUKE: 'D',
    Card.ASSASSIN: 'A',
    Card.CAPTAIN: 'C',
    Card.AMBASSADOR: 'B',
    Card.CONTESSA: 'S',
}


def bucket_coins(coins: int) -> str:
    """Bucket coins into strategically meaningful ranges."""
    if coins <= 2:
        return 'L'   # Low: limited options
    elif coins <= 6:
        return 'M'   # Mid: can assassinate (3+)
    elif coins <= 9:
        return 'H'   # High: can coup (7+)
    else:
        return 'X'   # Max: must coup (10+)


def cards_key(cards) -> str:
    """Sorted abbreviation string for a list of cards."""
    return ''.join(sorted(CARD_ABBREV[c] for c in cards))


def make_info_key(state, player_idx: int, decision_type: str,
                  context: str = '') -> str:
    """
    Build a hashable info set key from observable game state.

    Format: "TYPE|my_cards|my_coins|opp_coins|opp_infl|revealed[|context]"

    Components:
    - decision_type: ACT (action), CHL (challenge), CTR (counter), CC (challenge counter)
    - my_cards: sorted card abbreviations (e.g. 'AD' = Assassin+Duke)
    - my_coins: bucketed L/M/H/X
    - opp_coins: bucketed L/M/H/X
    - opp_infl: opponent's alive influence count (1 or 2)
    - revealed: all face-up cards sorted
    - context: what action/card is being responded to
    """
    player = state.players[player_idx]
    opp_idx = 1 - player_idx  # 2-player only
    opp = state.players[opp_idx]

    my_cards = cards_key(player.cards)
    my_coins = bucket_coins(player.coins)
    opp_coins = bucket_coins(opp.coins)
    opp_infl = str(opp.influence_count)

    all_revealed = []
    for p in state.players:
        all_revealed.extend(p.revealed)
    revealed = cards_key(all_revealed)

    parts = [decision_type, my_cards, my_coins, opp_coins, opp_infl, revealed]
    if context:
        parts.append(context)
    return '|'.join(parts)

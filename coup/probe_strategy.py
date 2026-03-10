"""
Strategy Probe: Analyze what the Gen 5 PPO bot "thinks" in various game situations.
Feeds synthetic game states into the model and reads out action probabilities.
"""
import torch
import numpy as np
from torch.distributions import Categorical
from .ppo_model import CoupLSTMPPO
from .game import Card, ActionType, Action
from .coup_env import (
    CoupEnv, RLAction, ACT_TYPE_TO_RL, RL_TO_ACT_TYPE,
    BLOCK_TO_RL, RL_TO_BLOCK
)

ACTION_NAMES = {
    0: "Income", 1: "Foreign Aid", 2: "Coup", 3: "Tax",
    4: "Assassinate", 5: "Steal", 6: "Exchange",
    7: "YES", 8: "NO",
    9: "Block(Duke)", 10: "Block(Captain)", 11: "Block(Ambassador)", 12: "Block(Contessa)",
    13: "Lose Card 0", 14: "Lose Card 1",
}

def make_view(my_cards, my_coins, opp_coins, opp_influence, opp_revealed=None, my_revealed=None, history=None):
    """Create a synthetic player view dict."""
    return {
        'player_id': 0,
        'name': 'Bot',
        'my_cards': list(my_cards),
        'my_revealed': my_revealed or [],
        'my_claimed_cards': [],
        'my_caught_bluff_count': 0,
        'my_coins': my_coins,
        'opponents': [{
            'player_id': 1,
            'name': 'Opponent',
            'coins': opp_coins,
            'influence_count': opp_influence,
            'revealed': opp_revealed or [],
            'claimed_cards': [],
            'caught_bluff_count': 0,
            'alive': True,
        }],
        'action_history': history or [],
        'turn_number': 1,
        'court_deck_size': 11,
    }

def probe_action(model, env, view, legal_action_types, device):
    """Probe what action the model prefers given a view and legal actions."""
    legal_actions = []
    for at in legal_action_types:
        target = 1 if at in (ActionType.COUP, ActionType.ASSASSINATE, ActionType.STEAL) else None
        legal_actions.append(Action(at, 0, target))
    
    msg = {'req_type': 'ACT', 'view': view, 'context': legal_actions}
    obs, mask = env._encode_state(msg)
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    hidden = model.reset_hidden(1, device)
    
    with torch.no_grad():
        logits, value, _ = model(state_tensor, hidden)
        mask_tensor = torch.BoolTensor(mask).view(1, 1, -1).to(device)
        logits[~mask_tensor] = -1e9
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    return probs, value.item()

def probe_challenge(model, env, view, claimed_card, device):
    """Probe whether the model would challenge a claim."""
    msg = {'req_type': 'CHL', 'view': view, 'context': claimed_card}
    obs, mask = env._encode_state(msg)
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    hidden = model.reset_hidden(1, device)
    
    with torch.no_grad():
        logits, value, _ = model(state_tensor, hidden)
        mask_tensor = torch.BoolTensor(mask).view(1, 1, -1).to(device)
        logits[~mask_tensor] = -1e9
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    return probs[RLAction.YES], probs[RLAction.NO]

def probe_block(model, env, view, blocking_cards, action_type, device):
    """Probe whether the model would block and with what card."""
    msg = {'req_type': 'CTR', 'view': view, 'context': {'action': action_type, 'cards': blocking_cards}}
    obs, mask = env._encode_state(msg)
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    hidden = model.reset_hidden(1, device)
    
    with torch.no_grad():
        logits, value, _ = model(state_tensor, hidden)
        mask_tensor = torch.BoolTensor(mask).view(1, 1, -1).to(device)
        logits[~mask_tensor] = -1e9
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    results = {"Don't block": probs[RLAction.NO]}
    for c in blocking_cards:
        results[f"Block({c.value})"] = probs[BLOCK_TO_RL[c]]
    return results

def print_action_probs(probs, label=""):
    """Pretty print action probabilities."""
    if label:
        print(f"  {label}")
    ranked = sorted(range(len(probs)), key=lambda i: -probs[i])
    for i in ranked:
        if probs[i] > 0.001:
            bar = "#" * int(probs[i] * 40)
            print(f"    {ACTION_NAMES[i]:20s} {probs[i]*100:5.1f}%  {bar}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_model_gen5.pt")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    model = CoupLSTMPPO(input_dim=70, hidden_dim=256, num_actions=15).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    env = CoupEnv()
    
    standard_actions = [ActionType.INCOME, ActionType.FOREIGN_AID, ActionType.TAX, 
                        ActionType.EXCHANGE, ActionType.STEAL]
    with_assassinate = standard_actions + [ActionType.ASSASSINATE]
    with_coup = standard_actions + [ActionType.COUP]
    all_actions = standard_actions + [ActionType.ASSASSINATE, ActionType.COUP]
    
    # =========================================================================
    print("=" * 70)
    print("  GEN 5 PPO STRATEGY ANALYSIS")
    print("=" * 70)
    
    # --- OPENING MOVES ---
    print("\n" + "-" * 70)
    print("  1. OPENING MOVES (2 coins, both players full health)")
    print("-" * 70)
    
    hands = [
        ([Card.DUKE, Card.ASSASSIN], "Duke + Assassin"),
        ([Card.DUKE, Card.CAPTAIN], "Duke + Captain"),
        ([Card.CAPTAIN, Card.ASSASSIN], "Captain + Assassin"),
        ([Card.AMBASSADOR, Card.CONTESSA], "Ambassador + Contessa"),
        ([Card.CONTESSA, Card.CONTESSA], "Contessa + Contessa (worst hand)"),
        ([Card.DUKE, Card.DUKE], "Duke + Duke"),
    ]
    
    for cards, label in hands:
        view = make_view(cards, 2, 2, 2)
        probs, val = probe_action(model, env, view, standard_actions, device)
        print(f"\n  Hand: {label}  (Value estimate: {val:.3f})")
        print_action_probs(probs)
    
    # --- MID-GAME WITH COINS ---
    print("\n" + "-" * 70)
    print("  2. MID-GAME: SHOULD IT ASSASSINATE OR SAVE FOR COUP?")
    print("-" * 70)
    
    for coins in [3, 5, 7]:
        view = make_view([Card.ASSASSIN, Card.CAPTAIN], coins, 4, 2)
        actions = with_coup if coins >= 7 else with_assassinate
        probs, val = probe_action(model, env, view, actions, device)
        print(f"\n  Hand: Assassin + Captain, {coins} coins vs opp 4 coins  (Value: {val:.3f})")
        print_action_probs(probs)
    
    # --- FORCED COUP ---
    print("\n" + "-" * 70)
    print("  3. AT 10 COINS (forced coup)")
    print("-" * 70)
    
    view = make_view([Card.DUKE, Card.CAPTAIN], 10, 3, 2)
    probs, val = probe_action(model, env, view, [ActionType.COUP], device)
    print(f"\n  Hand: Duke + Captain, 10 coins  (Value: {val:.3f})")
    print_action_probs(probs)

    # --- BLUFFING TENDENCIES ---
    print("\n" + "-" * 70)
    print("  4. BLUFFING: Does it claim cards it doesn't have?")
    print("-" * 70)
    
    # Has Contessa + Ambassador but can it bluff Tax (Duke)?
    view = make_view([Card.CONTESSA, Card.AMBASSADOR], 2, 4, 2)
    probs, val = probe_action(model, env, view, standard_actions, device)
    print(f"\n  Hand: Contessa + Ambassador, 2 coins  (Value: {val:.3f})")
    print(f"  (Tax=bluff, Exchange=honest, Steal=bluff)")
    print_action_probs(probs)
    
    view = make_view([Card.CONTESSA, Card.CONTESSA], 4, 3, 2)
    probs, val = probe_action(model, env, view, with_assassinate, device)
    print(f"\n  Hand: Contessa + Contessa, 4 coins  (Value: {val:.3f})")
    print(f"  (EVERYTHING except Income/FA is a bluff)")
    print_action_probs(probs)

    # --- CHALLENGE DECISIONS ---
    print("\n" + "-" * 70)
    print("  5. CHALLENGE DECISIONS: When does it call bluffs?")
    print("-" * 70)
    
    challenge_scenarios = [
        ([Card.DUKE, Card.CAPTAIN], Card.DUKE, "Opp claims Duke (we HAVE Duke)"),
        ([Card.DUKE, Card.DUKE], Card.DUKE, "Opp claims Duke (we have BOTH Dukes!)"),
        ([Card.CONTESSA, Card.AMBASSADOR], Card.DUKE, "Opp claims Duke (we have neither)"),
        ([Card.CAPTAIN, Card.AMBASSADOR], Card.ASSASSIN, "Opp claims Assassin (we don't have)"),
        ([Card.CONTESSA, Card.CAPTAIN], Card.CAPTAIN, "Opp claims Captain (we HAVE Captain)"),
    ]
    
    for cards, claimed, label in challenge_scenarios:
        view = make_view(cards, 3, 5, 2)
        yes_p, no_p = probe_challenge(model, env, view, claimed, device)
        bar = "#" * int(yes_p * 40)
        print(f"\n  {label}")
        print(f"    Challenge: {yes_p*100:5.1f}%  {bar}")
        print(f"    Allow:     {no_p*100:5.1f}%")
    
    # --- BLOCKING DECISIONS ---
    print("\n" + "-" * 70)
    print("  6. BLOCKING DECISIONS")
    print("-" * 70)
    
    # Block a steal with Captain vs Ambassador
    view = make_view([Card.CAPTAIN, Card.DUKE], 3, 5, 2)
    results = probe_block(model, env, view, [Card.CAPTAIN, Card.AMBASSADOR], ActionType.STEAL, device)
    print(f"\n  Opponent steals from us. Hand: Captain + Duke")
    for label, p in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"    {label:20s} {p*100:5.1f}%  {bar}")
    
    # Block a steal without the right card 
    view = make_view([Card.DUKE, Card.ASSASSIN], 3, 5, 2)
    results = probe_block(model, env, view, [Card.CAPTAIN, Card.AMBASSADOR], ActionType.STEAL, device)
    print(f"\n  Opponent steals from us. Hand: Duke + Assassin (no blocking card!)")
    for label, p in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"    {label:20s} {p*100:5.1f}%  {bar}")
    
    # Block assassination with Contessa
    view = make_view([Card.CONTESSA, Card.DUKE], 2, 6, 2)
    results = probe_block(model, env, view, [Card.CONTESSA], ActionType.ASSASSINATE, device)
    print(f"\n  Opponent assassinates us. Hand: Contessa + Duke (have Contessa!)")
    for label, p in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"    {label:20s} {p*100:5.1f}%  {bar}")
    
    # Block assassination WITHOUT Contessa
    view = make_view([Card.DUKE, Card.CAPTAIN], 2, 6, 2)
    results = probe_block(model, env, view, [Card.CONTESSA], ActionType.ASSASSINATE, device)
    print(f"\n  Opponent assassinates us. Hand: Duke + Captain (NO Contessa!)")
    for label, p in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"    {label:20s} {p*100:5.1f}%  {bar}")
    
    # --- ONE CARD LEFT ---
    print("\n" + "-" * 70)
    print("  7. DESPERATION: One card left, low coins")
    print("-" * 70)
    
    view = make_view([Card.DUKE], 2, 6, 2, my_revealed=[Card.CAPTAIN])
    probs, val = probe_action(model, env, view, standard_actions, device)
    print(f"\n  Hand: Duke only, 2 coins, opp has 6 coins  (Value: {val:.3f})")
    print_action_probs(probs)
    
    view = make_view([Card.CONTESSA], 1, 7, 2, my_revealed=[Card.AMBASSADOR])
    probs, val = probe_action(model, env, view, standard_actions, device)
    print(f"\n  Hand: Contessa only, 1 coin, opp has 7 coins  (Value: {val:.3f})")
    print_action_probs(probs)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

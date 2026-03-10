"""
Evaluation script for Gen 6 multiplayer model.
Tests against bots at various player counts and optionally against Gen 5.
"""
import argparse
import sys
from collections import defaultdict
import random

from .game import CoupGame
from .agents import RandomAgent, HeuristicAgent
from .zoo_agents import ZooAgent


def play_matchup(ai_agent, ai_name, opponent_factories, num_players, num_games=100):
    """Run games with the AI agent against a set of opponents."""
    print(f"\nEvaluating {ai_name} in {num_players}-player games ({num_games} games)...")
    
    wins = 0
    draws = 0

    for i in range(num_games):
        # Reset hidden state
        if hasattr(ai_agent, 'hidden_state'):
            ai_agent.hidden_state = ai_agent.model.reset_hidden(1, ai_agent.device)

        # Build opponent list
        opponents = [f() for f in opponent_factories[:num_players - 1]]
        
        # Random seat
        agents = list(opponents)
        seat = random.randint(0, num_players - 1)
        agents.insert(seat, ai_agent)

        game = CoupGame(agents, num_players=num_players, verbose=False)
        winner_idx = game.play_game()

        if winner_idx is not None and winner_idx == seat:
            wins += 1
        elif winner_idx is None:
            draws += 1

        if (i + 1) % max(1, num_games // 5) == 0:
            print(f"  Played {i+1}/{num_games}...")

    win_pct = wins / num_games * 100
    expected = 100.0 / num_players  # random baseline
    print(f"\n--- {ai_name} in {num_players}P ---")
    print(f"  Wins: {wins}/{num_games} ({win_pct:.1f}%)")
    print(f"  Random baseline: {expected:.1f}%")
    print(f"  Draws: {draws}")
    return win_pct


def play_head_to_head(agent1, name1, agent2, name2, num_games=500):
    """2-player head-to-head between two AI agents."""
    print(f"\nHead-to-head: {name1} vs {name2} ({num_games} games)...")
    
    wins1 = 0
    wins2 = 0
    draws = 0

    for i in range(num_games):
        if hasattr(agent1, 'hidden_state'):
            agent1.hidden_state = agent1.model.reset_hidden(1, agent1.device)
        if hasattr(agent2, 'hidden_state'):
            agent2.hidden_state = agent2.model.reset_hidden(1, agent2.device)

        seat = i % 2
        agents = [agent1, agent2] if seat == 0 else [agent2, agent1]

        game = CoupGame(agents, num_players=2, verbose=False)
        winner_idx = game.play_game()

        if winner_idx is None:
            draws += 1
        elif winner_idx == seat:
            wins1 += 1
        else:
            wins2 += 1

        if (i + 1) % max(1, num_games // 5) == 0:
            print(f"  Played {i+1}/{num_games}...")

    print(f"\n--- {name1} vs {name2} ---")
    print(f"  {name1}: {wins1} wins ({wins1/num_games*100:.1f}%)")
    print(f"  {name2}: {wins2} wins ({wins2/num_games*100:.1f}%)")
    print(f"  Draws: {draws}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gen 6 model")
    parser.add_argument("--model", type=str, default="versions/gen6/ppo_model_gen6.pt")
    parser.add_argument("--opponent", type=str, default=None, help="Path to opponent model (e.g. versions/gen5/ppo_model_gen5.pt)")
    parser.add_argument("--players", type=int, default=4, help="Number of players for multi-player eval")
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading Gen 6 model from {args.model}...")
    try:
        from .ppo_agent_mp import PPOAgentMP
        gen6 = PPOAgentMP(args.model, device="cpu")
    except Exception as e:
        print(f"Failed to load Gen 6 model: {e}")
        sys.exit(1)

    # Build opponent factories
    opp_factories = []
    for _ in range(5):
        r = random.random()
        if r < 0.5:
            opp_factories.append(lambda: HeuristicAgent())
        else:
            opp_factories.append(lambda: ZooAgent.random_profile())

    # Multi-player evaluation
    for n_players in [3, 4, 5, 6]:
        if n_players <= args.players:
            play_matchup(gen6, "Gen6", opp_factories, n_players, num_games=args.games)

    # Head-to-head vs Gen 5 if specified
    if args.opponent:
        print(f"\nLoading opponent model from {args.opponent}...")
        try:
            from .ppo_agent import PPOAgent
            gen5 = PPOAgent(args.opponent, device="cpu")
            play_head_to_head(gen6, "Gen6", gen5, "Gen5", num_games=args.games)
        except Exception as e:
            print(f"Failed to load opponent model: {e}")

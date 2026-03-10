import argparse
import sys
from collections import defaultdict

from .game import CoupGame
from .agents import RandomAgent, HeuristicAgent
from .cfr_agent import CFRAgent
from .ppo_agent import PPOAgent

def play_matchup(agent1, agent2, name1="PPO", name2="Opponent", num_games=1000):
    print(f"\nEvaluating {name1} vs {name2} over {num_games} games...")
    wins = {0: 0, 1: 0, None: 0}
    
    for i in range(num_games):
        # Swap seats every game to prevent first-player advantage
        seat = i % 2
        agents = [agent1, agent2] if seat == 0 else [agent2, agent1]
        
        # Reset PPO Hidden State if it exists
        if hasattr(agent1, 'hidden_state'):
            agent1.hidden_state = agent1.model.reset_hidden(1, agent1.device)
        if hasattr(agent2, 'hidden_state'):
            agent2.hidden_state = agent2.model.reset_hidden(1, agent2.device)
            
        game = CoupGame(agents, num_players=2, verbose=False)
        winner_idx = game.play_game()
        
        if winner_idx is None:
            wins[None] += 1
        elif winner_idx == seat:
            wins[0] += 1
        else:
            wins[1] += 1
            
        if (i+1) % max(1, num_games // 10) == 0:
            print(f"  Played {i+1}/{num_games}...")
            
    print(f"\n--- Results: {name1} vs {name2} ---")
    print(f"Total Games: {num_games}")
    print(f"{name1} Wins:  {wins[0]} ({wins[0]/num_games*100:.1f}%)")
    print(f"{name2} Wins:  {wins[1]} ({wins[1]/num_games*100:.1f}%)")
    print(f"Draws/Timeouts: {wins[None]} ({wins[None]/num_games*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--model", type=str, default="versions/gen1-4/ppo_model_gen1.pt")
    parser.add_argument("--cfr-model", type=str, default="cfr_model.json")
    args = parser.parse_args()

    print(f"Loading PPO Generation 1 model from {args.model}...")
    try:
        ppo = PPOAgent(args.model, device="cpu")
    except Exception as e:
        print(f"Failed to load PPO model: {e}")
        sys.exit(1)

    # 1. Evaluate against RandomAgent
    play_matchup(ppo, RandomAgent(), name1="PPO Gen1", name2="Random", num_games=args.games)

    # 2. Evaluate against HeuristicAgent
    play_matchup(ppo, HeuristicAgent(), name1="PPO Gen1", name2="Heuristic", num_games=args.games)

    # 3. Evaluate against CFRAgent
    try:
        cfr = CFRAgent(args.cfr_model)
        play_matchup(ppo, cfr, name1="PPO Gen1", name2="CFR", num_games=args.games)
    except Exception as e:
        print(f"Failed to load CFR model for evaluation: {e}")

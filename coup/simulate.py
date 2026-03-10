"""
Simulation runner for Coup.

Runs N games between agents and reports statistics.
Supports different agent matchups (e.g. Heuristic vs Random).
"""

import os
import time
import argparse
from collections import Counter

from .game import CoupGame, ActionType
from .agents import RandomAgent, HeuristicAgent

# Agent registry (CFR is special — needs a strategy file)
AGENT_TYPES = {
    'random': RandomAgent,
    'heuristic': HeuristicAgent,
}


def run_simulation(agents_config, num_games: int = 1000,
                   verbose: bool = False):
    """
    Run num_games games and print aggregate statistics.
    agents_config: list of (agent_name, AgentClass) tuples, one per player.
    """
    num_players = len(agents_config)
    wins = Counter()
    total_turns = 0
    draws = 0
    action_counts = Counter()

    start = time.time()

    for game_num in range(num_games):
        agents = [cls() for _, cls in agents_config]
        game = CoupGame(agents, num_players=num_players, verbose=verbose)
        winner = game.play_game()

        if winner is not None:
            wins[winner] += 1
        else:
            draws += 1

        total_turns += game.state.turn_number

        for entry in game.state.action_history:
            action_counts[entry['action']] += 1

        if (game_num + 1) % 100 == 0 and not verbose:
            print(f"  Completed {game_num + 1}/{num_games} games...", end='\r')

    elapsed = time.time() - start

    # Print results
    print(f"\n{'=' * 55}")
    print(f"  COUP SIMULATION RESULTS")
    print(f"{'=' * 55}")
    print(f"  Games played:     {num_games}")
    print(f"  Players:          {num_players}")
    print(f"  Matchup:          {' vs '.join(name for name, _ in agents_config)}")
    print(f"  Time elapsed:     {elapsed:.2f}s "
          f"({num_games / elapsed:.0f} games/sec)")
    print(f"  Avg game length:  {total_turns / num_games:.1f} turns")
    if draws > 0:
        print(f"  Draws:            {draws}")

    print(f"\n  Win Rates:")
    print(f"  {'-' * 45}")
    for i in range(num_players):
        name = agents_config[i][0]
        w = wins.get(i, 0)
        pct = w / num_games * 100
        bar = '#' * int(pct / 2)
        print(f"  P{i} ({name:>10}): {w:>5} wins ({pct:5.1f}%) {bar}")

    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print(f"\n  Action Distribution:")
        print(f"  {'-' * 45}")
        for action_type in ActionType:
            count = action_counts.get(action_type, 0)
            pct = count / total_actions * 100
            print(f"  {action_type.value:<15} {count:>6} ({pct:5.1f}%)")

    print(f"{'=' * 55}")


def main():
    parser = argparse.ArgumentParser(
        description='Simulate Coup games between agents'
    )
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of games to simulate (default: 1000)')
    parser.add_argument('--agents', type=str, nargs='+',
                        default=['random', 'random'],
                        help='Agent types: random, heuristic, cfr '
                             '(default: random random)')
    parser.add_argument('--strategy', type=str, default='strategy.json',
                        help='Strategy file for CFR agent (default: strategy.json)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game logs')
    args = parser.parse_args()

    if not 2 <= len(args.agents) <= 6:
        parser.error("Need 2-6 agents")

    valid_types = list(AGENT_TYPES.keys()) + ['cfr', 'ppo']
    for name in args.agents:
        if name not in valid_types:
            parser.error(f"Unknown agent type: {name}. Choose from {valid_types}")

    # Build agents config
    agents_config = []
    for name in args.agents:
        if name == 'cfr':
            if not os.path.exists(args.strategy):
                parser.error(f"Strategy file not found: {args.strategy}. Train first.")
            from .cfr_agent import CFRAgent
            cfr = CFRAgent.from_file(args.strategy)
            agents_config.append((name, lambda _cfr=cfr: _cfr))
        elif name == 'ppo':
            from .ppo_agent import PPOAgent
            
            # Use args.strategy parameter to pass the model weights (e.g. versions/gen5/ppo_model_gen5.pt)
            if not os.path.exists(args.strategy):
                parser.error(f"PPO Model file not found: {args.strategy}")
                
            # Usually use CPU for fast evaluation
            ppo = PPOAgent(args.strategy, "cpu")
            agents_config.append((name, lambda _ppo=ppo: _ppo))
        else:
            agents_config.append((name, AGENT_TYPES[name]))

    run_simulation(agents_config, args.games, args.verbose)


if __name__ == '__main__':
    main()


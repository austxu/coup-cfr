"""
Training script for MCCFR on 2-player Coup.

Usage:
    python -m coup.train_cfr --iterations 100000 --output strategy.json
"""

import argparse
import sys
import time
import os

from .cfr import OSMCCFRTrainer
from .cfr_agent import CFRAgent
from .agents import RandomAgent, HeuristicAgent
from .game import CoupGame


def evaluate(strategy: dict, opponent_cls, num_games: int = 500) -> float:
    """Evaluate CFR strategy vs an opponent. Returns CFR win rate."""
    wins = 0
    for _ in range(num_games):
        agents = [CFRAgent(strategy), opponent_cls()]
        game = CoupGame(agents, num_players=2)
        winner = game.play_game()
        if winner == 0:
            wins += 1
    return wins / num_games


def main():
    parser = argparse.ArgumentParser(
        description='Train MCCFR strategy for 2-player Coup')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='Number of MCCFR iterations (default: 100000)')
    parser.add_argument('--output', type=str, default='strategy.json',
                        help='Output file for trained strategy')
    parser.add_argument('--eval-every', type=int, default=25000,
                        help='Evaluate every N iterations')
    parser.add_argument('--eval-games', type=int, default=500,
                        help='Number of games per evaluation')
    args = parser.parse_args()

    sys.setrecursionlimit(10000)

    print(f"Training MCCFR for {args.iterations} iterations...")
    print(f"Output: {args.output}")
    print()

    trainer = OSMCCFRTrainer()
    start = time.time()

    iters_done = 0
    while iters_done < args.iterations:
        batch = min(args.eval_every, args.iterations - iters_done)
        trainer.train(batch, print_every=10000)
        iters_done += batch

        # Evaluate
        elapsed = time.time() - start
        print(f"\n  [{elapsed:.1f}s] Evaluating at {iters_done} iterations "
              f"({len(trainer.regret_sum)} info sets)...")

        # Build strategy for evaluation
        strat = {}
        for ikey in trainer.strategy_sum:
            actions = list(trainer.strategy_sum[ikey].keys())
            if actions:
                strat[ikey] = trainer.get_average_strategy(ikey, actions)

        vs_random = evaluate(strat, RandomAgent, args.eval_games)
        vs_heuristic = evaluate(strat, HeuristicAgent, args.eval_games)
        print(f"  vs Random:    {vs_random:.1%}")
        print(f"  vs Heuristic: {vs_heuristic:.1%}")

    # Save final strategy
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    trainer.save(args.output)


if __name__ == '__main__':
    main()

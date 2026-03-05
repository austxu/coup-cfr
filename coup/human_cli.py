import argparse
import sys
from .game import CoupGame
from .human_agent import HumanAgent
from .ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Play Coup against a trained PPO AI!")
    parser.add_argument("--model", type=str, default="ppo_model_gen1.pt", help="Path to the trained PyTorch model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda/mps)")
    args = parser.parse_args()

    # Load the AI
    print(f"Loading AI from {args.model}...")
    try:
        ai_agent = PPOAgent(args.model, device=args.device)
        ai_agent.name = "AI"
    except FileNotFoundError:
        print(f"Error: Could not find model file {args.model}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Human player
    human_name = input("Enter your name: ") or "Human"
    human_agent = HumanAgent(name=human_name)

    # Set up the game
    # Sequence of players: AI is 0, Human is 1. (Will be randomized in game setup)
    agents = [ai_agent, human_agent]
    game = CoupGame(agents, num_players=2, verbose=True)

    print("\nStarting game! You are playing against the PPO AI.")

    # Find the human player's index and show their starting hand
    human_idx = agents.index(human_agent)
    human_state = game.state.players[human_idx]
    print(f"\nYour starting hand: {[c.name for c in human_state.cards]}")
    print("-" * 25)

    winner_idx = game.play_game()

    if winner_idx is not None:
        winner_name = human_name if game.agents[winner_idx] == human_agent else "AI"
        print(f"\nGAME OVER! The winner is: {winner_name}")
    else:
        print("\nGAME OVER! It's a draw.")

    ai_player = game.state.players[0]
    alive_cards = [c.name for c in ai_player.cards]
    revealed_cards = [c.name for c in ai_player.revealed]
    print("\nAI's Final Cards:")
    if alive_cards:
        print(f"  Hidden (Unrevealed): {', '.join(alive_cards)}")
    if revealed_cards:
        print(f"  Revealed (Dead): {', '.join(revealed_cards)}")


if __name__ == "__main__":
    main()

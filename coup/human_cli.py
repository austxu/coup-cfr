import argparse
import sys
from .game import CoupGame
from .human_agent import HumanAgent
from .ppo_agent import PPOAgent
from .cfr_agent import CFRAgent
import random

def main():
    parser = argparse.ArgumentParser(description="Play Coup against trained AI!")
    parser.add_argument("--model", type=str, default="versions/gen5/ppo_model_gen5.pt", help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--players", type=int, default=2, help="Number of players (2-6)")
    args = parser.parse_args()

    if not 2 <= args.players <= 6:
        print("Error: players must be 2-6")
        sys.exit(1)

    # Load the AI
    print(f"Loading AI from {args.model}...")
    try:
        if args.model.endswith('.json'):
            ai_agent = CFRAgent.from_file(args.model)
            ai_agent.name = "CFR-AI"
        elif args.players > 2:
            from .ppo_agent_mp import PPOAgentMP
            ai_agent = PPOAgentMP(args.model, device=args.device)
            ai_agent.name = "PPO-AI-Gen6"
        else:
            ai_agent = PPOAgent(args.model, device=args.device)
            ai_agent.name = "PPO-AI"
    except FileNotFoundError:
        print(f"Error: Could not find model file {args.model}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Build additional AI bots for multiplayer
    extra_bots = []
    if args.players > 2:
        from .agents import HeuristicAgent
        from .zoo_agents import ZooAgent
        for i in range(args.players - 2):
            if i % 2 == 0:
                bot = HeuristicAgent()
                bot.name = f"Heuristic-{i+1}"
            else:
                bot = ZooAgent.random_profile()
                bot.name = f"Zoo-{i+1}"
            extra_bots.append(bot)

    # Human player
    human_name = input("Enter your name: ") or "Human"
    human_agent = HumanAgent(name=human_name)

    human_wins = 0
    ai_wins = 0
    draws = 0
    total_games = 0

    while True:
        # Reset AI hidden state
        if hasattr(ai_agent, 'hidden_state'):
            ai_agent.hidden_state = ai_agent.model.reset_hidden(1, ai_agent.device)

        # Build agent list
        agents = [ai_agent, human_agent] + extra_bots
        random.shuffle(agents)
        game = CoupGame(agents, num_players=args.players, verbose=True)

        print(f"\n--- Game {total_games + 1} ({args.players} players) ---")
        player_names = [game.state.names[i] for i in range(args.players)]
        print(f"Players: {', '.join(player_names)}")

        # Show starting hand
        human_idx = agents.index(human_agent)
        human_state = game.state.players[human_idx]
        print(f"\nYour starting hand: {[c.name for c in human_state.cards]}")
        print("-" * 25)

        winner_idx = game.play_game()
        total_games += 1

        if winner_idx is not None:
            if game.agents[winner_idx] == human_agent:
                print(f"\nGAME OVER! You win!")
                human_wins += 1
            elif game.agents[winner_idx] == ai_agent:
                print(f"\nGAME OVER! The {ai_agent.name} wins!")
                ai_wins += 1
            else:
                winner_name = game.state.names[winner_idx]
                print(f"\nGAME OVER! {winner_name} wins!")
        else:
            print("\nGAME OVER! It's a draw.")
            draws += 1

        # Show AI final cards
        ai_idx = agents.index(ai_agent)
        ai_player = game.state.players[ai_idx]
        alive_cards = [c.name for c in ai_player.cards]
        revealed_cards = [c.name for c in ai_player.revealed]
        print(f"\n{ai_agent.name}'s Final Cards:")
        if alive_cards:
            print(f"  Hidden (Unrevealed): {', '.join(alive_cards)}")
        if revealed_cards:
            print(f"  Revealed (Dead): {', '.join(revealed_cards)}")

        print("\n" + "=" * 40)
        play_again = input("Play another game? (y/n): ").strip().lower()
        if play_again not in ['y', 'yes', '']:
            break

    print(f"\n--- Session Complete! ---")
    print(f"Total Games Played: {total_games}")
    if total_games > 0:
        print(f"Your Wins: {human_wins} ({(human_wins/total_games)*100:.1f}%)")
        print(f"AI Wins:   {ai_wins} ({(ai_wins/total_games)*100:.1f}%)")
        print(f"Draws:     {draws} ({(draws/total_games)*100:.1f}%)")

if __name__ == "__main__":
    main()

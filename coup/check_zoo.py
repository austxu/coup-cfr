from coup.game import CoupGame
from coup.zoo_agents import ZooAgent

print("Starting Zoo vs Zoo check...")
for i in range(100):
    game = CoupGame([ZooAgent.random_profile(), ZooAgent.random_profile()], num_players=2, verbose=False)
    game.play_game()
print("100 Zoo games completed successfully!")

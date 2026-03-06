import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .coup_env import CoupEnv
from .zoo_agents import ZooAgent
from .agents import HeuristicAgent
from .ppo_agent import PPOAgent
import random

class SelfPlayZoo:
    def __init__(self, selfplay_agent):
        self.selfplay_agent = selfplay_agent
        
    def update_model(self, state_dict):
        """Update the self-play opponent with the latest training weights."""
        self.selfplay_agent.model.load_state_dict(state_dict)
        
    def __call__(self):
        r = random.random()
        if r < 0.15:
            # 15% chance to play the generalized Random Zoo to prevent catastrophic forgetting
            return ZooAgent.random_profile()
        elif r < 0.25:
            # 10% chance to practice specifically against the Heuristic
            return HeuristicAgent()
        else:
            # 75% chance to practice against the current snapshot of itself (Self-Play)
            return self.selfplay_agent
            
from .ppo_model import CoupLSTMPPO

# -----------------------------------------------------------------------------
# PPO Hyperparameters
# -----------------------------------------------------------------------------
GAMMA = 0.99            # Discount factor
GAE_LAMBDA = 0.95       # GAE smoothing
CLIP_EPSILON = 0.2      # PPO Clip range
VF_COEF = 0.5           # Value function loss coefficient
ENT_COEF = 0.01         # Entropy bonus (exploration)
LR = 1e-4               # Learning rate
PPO_EPOCHS = 4          # Number of optimization epochs per rollout
BATCH_SIZE = 64         # Batch size for sequence updates
# -----------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    advantages = []
    gae = 0
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
        gae = delta + gamma * gae_lambda * (1 - int(dones[i])) * gae
        advantages.insert(0, gae)
    return advantages

def rollout(env, model, num_episodes, device, zoo):
    model.eval()
    
    states, actions, log_probs, rewards, dones, values, masks = [], [], [], [], [], [], []
    episode_rewards = []
    
    for _ in range(num_episodes):
        opponent = zoo()
        env.opponent_cls = lambda: opponent
        
        obs, info = env.reset()
        hidden = model.reset_hidden(1, device)
        done = False
        ep_reward = 0.0
        
        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_arr = info['action_mask']
            
            with torch.no_grad():
                logits, value, hidden = model(state_tensor, hidden)
                mask_tensor = torch.BoolTensor(mask_arr).view(1, 1, -1).to(device)
                logits[~mask_tensor] = -1e9
                
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            next_obs, reward, done, _, info = env.step(action.item())
            
            states.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            masks.append(mask_arr)
            
            obs = next_obs
            ep_reward += reward
            
        episode_rewards.append(ep_reward)
        
    return states, actions, log_probs, rewards, dones, values, masks, episode_rewards


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, masks, device):
    model.train()
    
    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
    returns_t = torch.FloatTensor(returns).to(device)
    advantages_t = torch.FloatTensor(advantages).to(device)
    masks_t = torch.BoolTensor(np.array(masks)).to(device)
    
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    n_steps = len(states)
    
    for _ in range(PPO_EPOCHS):
        states_seq = states_t.unsqueeze(0)
        hidden = model.reset_hidden(1, device)
        logits, values, _ = model(states_seq, hidden)
        
        logits = logits.squeeze(0)
        values = values.squeeze()
        
        logits[~masks_t] = -1e9
        
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = F.mse_loss(values, returns_t)
        
        loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
    return actor_loss.item(), critic_loss.item(), entropy.item()


def main():
    parser = argparse.ArgumentParser(description="Train Coup LSTM PPO with Self-Play")
    parser.add_argument("--episodes", type=int, default=1000000)
    parser.add_argument("--rollout-size", type=int, default=200, help="Episodes per PPO update")
    parser.add_argument("--base-model", type=str, default="ppo_model_gen4_init.pt")
    parser.add_argument("--output", type=str, default="ppo_model_gen4.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = CoupEnv()
    model = CoupLSTMPPO(input_dim=35, hidden_dim=64, num_actions=15).to(device)
    
    # Needs a baseline model file for the PPOAgent wrapper to load initially
    if not os.path.exists(args.base_model):
        print(f"Error: Base model {args.base_model} not found.")
        return
        
    model.load_state_dict(torch.load(args.base_model, map_location=device))
    print(f"Loaded Gen 2 model {args.base_model} for self-play training!")
    
    # Initialize the self-play zoo wrapper
    selfplay_agent = PPOAgent(args.base_model, str(device))
    zoo = SelfPlayZoo(selfplay_agent)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    start_time = time.time()
    total_episodes = 0
    
    print(f"Starting Gen 4 Self-Play training for {args.episodes} episodes...")
    
    while total_episodes < args.episodes:
        # Collect rollouts
        b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values, b_masks, ep_rewards = rollout(
            env, model, args.rollout_size, device, zoo
        )
        
        total_episodes += args.rollout_size
        
        # Calculate Advantages and Returns
        advantages = compute_gae(b_rewards, b_values, b_dones)
        returns = [adv + val for adv, val in zip(advantages, b_values)]
        
        # Update Model
        a_loss, c_loss, ent = ppo_update(
            model, optimizer, b_states, b_actions, b_log_probs, returns, advantages, b_masks, device
        )
        
        # Logging
        win_rate = sum(1 for r in ep_rewards if r > 0.5) / len(ep_rewards) * 100
        avg_len = len(b_states) / len(ep_rewards)
        elapsed = time.time() - start_time
        
        print(f"Ep {total_episodes:8d} | WinRT: {win_rate:5.1f}% | AvgLen: {avg_len:4.1f} | "
              f"ALoss: {a_loss:6.3f} | CLoss: {c_loss:6.3f} | Ent: {ent:5.3f} | Time: {elapsed:5.1f}s")
              
        # Update self-play opponent every 2000 episodes
        if total_episodes % (args.rollout_size * 10) == 0:
            torch.save(model.state_dict(), args.output)
            zoo.update_model(model.state_dict())
            print(f"  -> Updated self-play snapshot & saved checkpoint to {args.output}")

    # Final Save
    torch.save(model.state_dict(), args.output)
    print(f"\nTraining Complete! Saved final Gen 4 model to {args.output}")


if __name__ == "__main__":
    main()

"""
Gen 6 Training: Multi-player Coup LSTM PPO with Self-Play.
Trains a single model to handle 3-6 player games.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random

from .coup_env_mp import CoupEnvMP, OBS_DIM, TARGET_SIZE
from .coup_env import RLAction
from .zoo_agents import ZooAgent
from .agents import HeuristicAgent
from .ppo_model_gen6 import CoupLSTMPPOv2


class SelfPlayZooMP:
    """Opponent generator for multiplayer self-play training."""
    def __init__(self, selfplay_agent):
        self.selfplay_agent = selfplay_agent

    def update_model(self, state_dict):
        self.selfplay_agent.model.load_state_dict(state_dict)

    def make_opponents(self, num_opponents):
        """Generate a list of opponents for a game."""
        opponents = []
        for _ in range(num_opponents):
            r = random.random()
            if r < 0.15:
                opponents.append(ZooAgent.random_profile())
            elif r < 0.25:
                opponents.append(HeuristicAgent())
            else:
                opponents.append(self.selfplay_agent)
        return opponents


# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VF_COEF = 0.5
ENT_COEF = 0.04
LR = 1e-4
PPO_EPOCHS = 4


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
    is_target_steps = []
    episode_rewards = []

    for _ in range(num_episodes):
        # Randomize player count for this episode
        num_players = random.choice([3, 4, 5, 6])
        env.num_players = num_players
        
        opponents = zoo.make_opponents(num_players - 1)
        env.opponent_cls_list = [lambda o=o: o for o in opponents]

        obs, info = env.reset()
        hidden = model.reset_hidden(1, device)
        done = False
        ep_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            is_target = info.get('is_target_step', False)

            with torch.no_grad():
                action_logits, target_logits, value, hidden = model(state_tensor, hidden)

                if is_target:
                    mask_arr = info['target_mask']
                    mask_tensor = torch.BoolTensor(mask_arr).view(1, 1, -1).to(device)
                    target_logits[~mask_tensor] = -1e9
                    dist = Categorical(logits=target_logits)
                else:
                    mask_arr = info['action_mask']
                    mask_tensor = torch.BoolTensor(mask_arr).view(1, 1, -1).to(device)
                    action_logits[~mask_tensor] = -1e9
                    dist = Categorical(logits=action_logits)

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
            is_target_steps.append(is_target)

            obs = next_obs
            ep_reward += reward

        episode_rewards.append(ep_reward)

    return states, actions, log_probs, rewards, dones, values, masks, is_target_steps, episode_rewards


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, masks, is_target_steps, dones, device):
    model.train()

    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
    returns_t = torch.FloatTensor(returns).to(device)
    advantages_t = torch.FloatTensor(advantages).to(device)
    is_target_t = torch.BoolTensor(is_target_steps)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # Find episode boundaries
    ep_ranges = []
    start = 0
    for i, d in enumerate(dones):
        if d:
            ep_ranges.append((start, i + 1))
            start = i + 1
    if start < len(states):
        ep_ranges.append((start, len(states)))

    for _ in range(PPO_EPOCHS):
        all_action_logits = []
        all_target_logits = []
        all_values = []

        for ep_start, ep_end in ep_ranges:
            ep_states = states_t[ep_start:ep_end].unsqueeze(0)
            hidden = model.reset_hidden(1, device)
            ep_act_logits, ep_tgt_logits, ep_values, _ = model(ep_states, hidden)
            all_action_logits.append(ep_act_logits.squeeze(0))
            all_target_logits.append(ep_tgt_logits.squeeze(0))
            all_values.append(ep_values.squeeze(0).squeeze(-1))

        action_logits = torch.cat(all_action_logits, dim=0)
        target_logits = torch.cat(all_target_logits, dim=0)
        values = torch.cat(all_values, dim=0)

        # Build masks and compute log probs per step
        new_log_probs = torch.zeros(len(states), device=device)
        entropy_sum = torch.tensor(0.0, device=device)

        for i in range(len(states)):
            mask_tensor = torch.BoolTensor(masks[i]).to(device)
            if is_target_t[i]:
                logits = target_logits[i].clone()
                logits[~mask_tensor] = -1e9
            else:
                logits = action_logits[i].clone()
                logits[~mask_tensor] = -1e9

            dist = Categorical(logits=logits)
            new_log_probs[i] = dist.log_prob(actions_t[i])
            entropy_sum = entropy_sum + dist.entropy()

        entropy = entropy_sum / len(states)

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
    parser = argparse.ArgumentParser(description="Train Gen 6 N-Player Coup LSTM PPO")
    parser.add_argument("--episodes", type=int, default=2000000)
    parser.add_argument("--rollout-size", type=int, default=100, help="Episodes per PPO update (smaller for multiplayer)")
    parser.add_argument("--output", type=str, default="versions/gen6/ppo_model_gen6.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    env = CoupEnvMP()
    model = CoupLSTMPPOv2(input_dim=OBS_DIM, hidden_dim=512, num_actions=RLAction.SIZE, num_targets=TARGET_SIZE).to(device)

    # Initialize fresh weights
    init_path = "versions/gen6/ppo_model_gen6_init.pt"
    if not os.path.exists(init_path):
        print(f"Initializing fresh Gen 6 weights...")
        torch.save(model.state_dict(), init_path)
    else:
        model.load_state_dict(torch.load(init_path, map_location=device))
        print(f"Loaded Gen 6 init weights from {init_path}")

    # Self-play wrapper
    from .ppo_agent_mp import PPOAgentMP
    selfplay_agent = PPOAgentMP(init_path, str(device))
    zoo = SelfPlayZooMP(selfplay_agent)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    total_episodes = 0

    log_file = open("versions/gen6/training_log_gen6.txt", "a")
    log_file.write(f"Starting Gen 6 N-Player training for {args.episodes} episodes on {device}...\n")
    log_file.flush()
    print(f"Starting Gen 6 N-Player training for {args.episodes} episodes...")

    while total_episodes < args.episodes:
        b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values, b_masks, b_target_steps, ep_rewards = rollout(
            env, model, args.rollout_size, device, zoo
        )

        total_episodes += args.rollout_size

        advantages = compute_gae(b_rewards, b_values, b_dones)
        returns = [adv + val for adv, val in zip(advantages, b_values)]

        a_loss, c_loss, ent = ppo_update(
            model, optimizer, b_states, b_actions, b_log_probs, returns, advantages, b_masks, b_target_steps, b_dones, device
        )

        win_rate = sum(1 for r in ep_rewards if r > 0.5) / len(ep_rewards) * 100
        avg_len = len(b_states) / len(ep_rewards)
        elapsed = time.time() - start_time

        log_str = (f"Ep {total_episodes:8d} | WinRT: {win_rate:5.1f}% | AvgLen: {avg_len:4.1f} | "
                   f"ALoss: {a_loss:6.3f} | CLoss: {c_loss:6.3f} | Ent: {ent:5.3f} | Time: {elapsed:5.1f}s")
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        if total_episodes % (args.rollout_size * 20) == 0:
            torch.save(model.state_dict(), args.output)
            zoo.update_model(model.state_dict())
            print(f"  -> Updated self-play snapshot & saved checkpoint to {args.output}")
            log_file.write(f"  -> Updated self-play snapshot & saved checkpoint to {args.output}\n")
            log_file.flush()

    torch.save(model.state_dict(), args.output)
    log_file.write(f"\nTraining Complete! Saved final Gen 6 model to {args.output}\n")
    log_file.close()
    print(f"\nTraining Complete! Saved final Gen 6 model to {args.output}")


if __name__ == "__main__":
    main()

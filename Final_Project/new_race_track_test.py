# File: ppo_snn_highway_env.py

import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import surrogate, neuron, functional

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
ENTROPY_BONUS = 0.01
CRITIC_LOSS_WEIGHT = 0.5
BATCH_SIZE = 64
TIMESTEPS = 10
HIDDEN_SIZE = 256
EPOCHS = 10
STEPS_PER_UPDATE = 2048
MAX_TIMESTEPS = 1e6

# Custom Spiking Actor-Critic Network
class SpikingPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SpikingPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc_policy = nn.Linear(HIDDEN_SIZE, action_dim)
        self.fc_value = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        # Flatten over timesteps if needed
        if x.dim() == 3:
            batch_size, timesteps, _ = x.shape
            x = x.reshape(batch_size * timesteps, -1)

        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)

        # Reset states to ensure the model is stateless between rollouts
        functional.reset_net(self)
        
        return self.fc_policy(x), self.fc_value(x)

# PPO with SNN
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = SpikingPolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def get_action(self, state):
        # Convert state to a tensor with the correct shape
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits, _ = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages, returns = [], []
        gae = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + GAMMA * (1 - dones[i]) * next_value - values[i]
            gae = td_error + GAMMA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]
        return torch.tensor(advantages).to(DEVICE), torch.tensor(returns).to(DEVICE)

    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(EPOCHS):
            for idx in range(0, len(states), BATCH_SIZE):
                batch_idx = slice(idx, idx + BATCH_SIZE)
                state = states[batch_idx]
                action = actions[batch_idx]
                old_log_prob = log_probs[batch_idx]
                advantage = advantages[batch_idx]
                ret = returns[batch_idx]

                logits, value = self.policy(state)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # Compute new log probability and entropy
                new_log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()

                # PPO Clipped Loss
                ratio = torch.exp(new_log_prob - old_log_prob)
                clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
                policy_loss = -torch.min(ratio * advantage, clip_adv).mean()

                # Critic Loss
                critic_loss = (ret - value.squeeze(-1)).pow(2).mean()

                # Total Loss
                loss = policy_loss + CRITIC_LOSS_WEIGHT * critic_loss - ENTROPY_BONUS * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Training Loop
def train():
    env = gym.make('highway-v0')
    env.configure({"vehicles_count": 20})
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)

    total_timesteps = 0
    while total_timesteps < MAX_TIMESTEPS:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        state = env.reset()
        for _ in range(STEPS_PER_UPDATE):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Record data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state
            total_timesteps += 1
            if done:
                state = env.reset()

        # Calculate advantages
        _, next_value = agent.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
        advantages, returns = agent.compute_advantages(rewards, values, dones, next_value.detach())

        # Update policy
        agent.update(
            torch.tensor(states, dtype=torch.float32).to(DEVICE),
            torch.tensor(actions).to(DEVICE),
            torch.tensor(log_probs).to(DEVICE),
            returns,
            advantages
        )

if __name__ == "__main__":
    train()

"""
CartPole DQN Trainer
Author: Milena Napiorkowska
Based on: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import pygame
import math
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from itertools import count

# region constants
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
TARGET_UPDATE = 0.005 # Soft update factor
EPISODES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSITION = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# endregion

class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_observations, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

class ReplayMemory:
    """A cyclic buffer of bounded size that holds recent transitions."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(TRANSITION(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """DQN Agent with experience replay and target network."""
    def __init__(self, num_observations, num_actions):
        self.policy_model = DQN(num_observations, num_actions).to(DEVICE)
        self.target_model = DQN(num_observations, num_actions).to(DEVICE)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, env):
        """Select action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)

    def optimize_model(self):
        """Optimize the agent using experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = TRANSITION(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_model.parameters(), self.policy_model.parameters()):
            target_param.data.copy_(policy_param.data * TARGET_UPDATE + target_param.data * (1.0 - TARGET_UPDATE))

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self, path):
        self.policy_model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.policy_model.eval()

    def train(self, env):
        """Train the DQN agent."""
        episode_durations = []
        for episode in range(EPISODES):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            for t in count():
                action = self.select_action(state, env)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)
                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()
                if terminated or truncated:
                    episode_durations.append(t + 1)
                    break
        self.save("dqn_cartpole.pth")
        env.close()
        return episode_durations
    
    def test(self, env):
        """Test the trained agent."""
        self.load("dqn_cartpole.pth")
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward = 0

        while True:
            env.render()
            action = self.select_action(state, env)
            state, reward, done, _, _ = env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            total_reward += reward
            if done:
                break
        print(f"Test Run - Total Reward: {total_reward}")
        env.close()

def plot_durations(episode_durations):
    plt.figure(1)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    num_observations = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(num_observations, num_actions)
    episode_durations = agent.train(env)
    agent.test(env= gym.make("CartPole-v1", render_mode="human"))
    plot_durations(episode_durations)


import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from env import ShortestPathEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, num_states, num_actions):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_states, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self, env) -> None:
        self.env = env
        self.init_hyperparams()

        self.Q = MLP(env.num_states, env.num_actions)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        self.writer = SummaryWriter()

    def init_hyperparams(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.05

    def get_best_action(self, state):
        state_tensor = torch.eye(self.env.num_states)[state]
        with torch.no_grad():
            action_values = self.Q(state_tensor)
        action = torch.argmax(action_values).item()
        return action

    def explore_policy(self, state):
        """ epsilon greedy"""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.env.num_actions)
        else:
            action = self.get_best_action(state)
        return action

    def learn(self, num_episodes=1000):
        for i in tqdm(range(num_episodes)):
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = self.explore_policy(state)

                next_state, reward, done = self.env.step(action)
                episode_reward += reward

                next_q = torch.max(
                    self.Q(torch.eye(self.env.num_states)[next_state]))
                target_q = reward + self.gamma * next_q
                current_q = self.Q(
                    torch.eye(self.env.num_states)[state])[action]
                loss = self.criterion(current_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

                if done:
                    break
            if i % 5 == 0:
                total_reward, _ = self.eval_policy()
                self.writer.add_scalar("total reward", total_reward, i)
        self.writer.close()

    def eval_policy(self):
        state = self.env.reset()
        path = [state]
        total_reward = 0
        while True:
            action = self.get_best_action(state)
            state, reward, done = self.env.step(action)
            total_reward += reward
            path.append(state)
            if done:
                break
        return total_reward, path


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])

    start = 0
    goal = 4
    env = ShortestPathEnv(G, start, goal)

    model = DQN(env)
    model.learn(1000)
    _, path = model.eval_policy()
    print(f"shortest path: {path}")

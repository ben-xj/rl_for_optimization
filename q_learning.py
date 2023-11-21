import numpy as np
import networkx as nx
from env import ShortestPathEnv


class QLearning:

    def __init__(self, env) -> None:
        self.env = env
        self.init_hyperparams()

        self.Q = np.zeros((env.num_states, env.num_actions))

    def init_hyperparams(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.num_episodes = 1000

    def learn(self):
        for i in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                # epsilon greedy
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = np.argmax(self.Q[state])

                next_state, reward, done = self.env.step(action)
                episode_reward += reward

                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward + np.max(self.Q[next_state]) - self.Q[state, action]
                )

                state = next_state

                if done:
                    print(f'episode {i} reward: {episode_reward}')
                    break

import numpy as np
import networkx as nx
from env import ShortestPathEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class QLearning:

    def __init__(self, env) -> None:
        self.env = env
        self.init_hyperparams()

        self.Q = np.zeros((env.num_states, env.num_actions))
        self.writer = SummaryWriter()

    def init_hyperparams(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.05

    def learn(self, num_episodes=1000):
        for i in tqdm(range(num_episodes)):
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
            action = np.argmax(self.Q[state])
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

    model = QLearning(env)
    model.learn(2000)
    _, path = model.eval_policy()
    print(f"shortest path: {path}")

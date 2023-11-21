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

    def eval_policy(self):
        cur_state = start
        path = [cur_state]

        while cur_state != self.env.goal:
            action = np.argmax(self.Q[cur_state])
            action_edge = list(self.env.graph.edges)[action]
            if cur_state == action_edge[0]:
                cur_state = action_edge[1]
                path.append(cur_state)
            else:
                print('wrong policy')
                break
        return path


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])

    start = 0
    goal = 4
    env = ShortestPathEnv(G, start, goal)

    model = QLearning(env)
    model.learn()
    path = model.eval_policy()
    print(f"shortest path: {path}")

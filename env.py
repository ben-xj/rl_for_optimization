class ShortestPathEnv:

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.num_states = len(graph.nodes)
        self.num_actions = len(graph.edges)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        action_edge = list(self.graph.edges)[action]
        if self.state == action_edge[0]:
            self.state = action_edge[1]
            reward = -1 if self.state != self.goal else 100
        else:
            self.state = self.start
            reward = -100
        done = self.state == self.goal or self.state == self.start
        return self.state, reward, done

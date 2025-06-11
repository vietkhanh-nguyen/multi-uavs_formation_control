import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, bounds, step_size=1.0, goal_radius=1.0, max_iter=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.bounds = bounds
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.nodes = [self.start]

    def sample(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def nearest(self, sample):
        dists = [np.linalg.norm(n.pos - sample) for n in self.nodes]
        return self.nodes[np.argmin(dists)]

    def steer(self, from_node, to_pos):
        direction = to_pos - from_node.pos
        distance = np.linalg.norm(direction)
        direction = direction / distance
        new_pos = from_node.pos + min(self.step_size, distance) * direction
        new_node = Node(new_pos)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(new_pos - from_node.pos)
        return new_node

    def near_nodes(self, node, radius=3.0):
        positions = np.array([n.pos for n in self.nodes])
        tree = cKDTree(positions)
        idxs = tree.query_ball_point(node.pos, radius)
        return [self.nodes[i] for i in idxs]

    def rewire(self, new_node, near_nodes):
        for near in near_nodes:
            cost = new_node.cost + np.linalg.norm(new_node.pos - near.pos)
            if cost < near.cost:
                near.parent = new_node
                near.cost = cost

    def build(self):
        for _ in range(self.max_iter):
            sample = self.sample()
            nearest_node = self.nearest(sample)
            new_node = self.steer(nearest_node, sample)
            self.nodes.append(new_node)

            near = self.near_nodes(new_node)
            best_parent = min(near, key=lambda n: n.cost + np.linalg.norm(n.pos - new_node.pos), default=nearest_node)
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + np.linalg.norm(best_parent.pos - new_node.pos)
            self.rewire(new_node, near)

            if np.linalg.norm(new_node.pos - self.goal.pos) < self.goal_radius:
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + np.linalg.norm(new_node.pos - self.goal.pos)
                return self.extract_path()

        return None

    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

# --------------------- Run Example ---------------------

bounds = np.array([[0, 20], [0, 20], [0, 10]])
rrt_star = RRTStar(start=[0, 0, 0], goal=[18, 18, 8], bounds=bounds, max_iter=900)
path = rrt_star.build()

if path:
    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node in rrt_star.nodes:
        if node.parent:
            ax.plot(*zip(node.pos, node.parent.pos), color='gray', linewidth=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=2, label="Path")
    ax.scatter(*rrt_star.start.pos, color='green', s=100, label='Start')
    ax.scatter(*rrt_star.goal.pos, color='blue', s=100, label='Goal')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.legend()
    plt.show()
else:
    print("No path found.")

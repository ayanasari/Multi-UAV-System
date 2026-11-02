import heapq
import numpy as np
from .env import ACTIONS


def astar(grid, start, goal):
    start, goal = tuple(start), tuple(goal)
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    openq = [(h(start), 0, start)]
    came, g = {start: None}, {start: 0}
    while openq:
        _, cost, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if cost > g[cur]:
            continue
        for dr, dc in ACTIONS[1:]:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and not grid[nxt]:
                nc = cost + 1
                if nc < g.get(nxt, 1e9):
                    g[nxt], came[nxt] = nc, cur
                    heapq.heappush(openq, (nc + h(nxt), nc, nxt))
    return None


def path_to_actions(path):
    acts = []
    for a, b in zip(path, path[1:]):
        d = np.array(b) - np.array(a)
        acts.append(int(np.where((ACTIONS == d).all(1))[0][0]))
    return acts

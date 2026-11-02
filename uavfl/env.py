import numpy as np
from collections import deque

ACTIONS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
VIEW = 7
R = VIEW // 2
OBS_DIM = 4 * VIEW * VIEW + 3


def bfs_dist(grid, start):
    h, w = grid.shape
    d = np.full((h, w), -1, dtype=np.int32)
    d[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS[1:]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


def make_map(rng, size, density):
    while True:
        grid = (rng.random((size, size)) < density).astype(np.int8)
        free = np.argwhere(grid == 0)
        if len(free) < size * size * 0.5:
            continue
        d = bfs_dist(grid, tuple(free[0]))
        if (d[grid == 0] >= 0).all():
            return grid


class MultiUAVGrid:
    def __init__(self, size=12, n_uavs=3, density=0.15, max_steps=60, seed=0):
        self.size, self.n, self.density, self.max_steps = size, n_uavs, density, max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self, grid=None):
        self.grid = make_map(self.rng, self.size, self.density) if grid is None else grid
        free = np.argwhere(self.grid == 0)
        while True:
            idx = self.rng.choice(len(free), 2 * self.n, replace=False)
            self.pos = free[idx[: self.n]].copy()
            self.goal = free[idx[self.n :]].copy()
            if all(abs(self.pos[i] - self.goal[i]).sum() >= self.size // 2 for i in range(self.n)):
                break
        self.done = np.zeros(self.n, dtype=bool)
        self.t = 0
        self.path_len = np.zeros(self.n, dtype=np.int32)
        self.collisions = 0
        self.gdist = [bfs_dist(self.grid, tuple(g)) for g in self.goal]
        self.visits = np.zeros((self.n, self.size + 2 * R, self.size + 2 * R), dtype=np.float32)
        for i in range(self.n):
            self.visits[i, self.pos[i][0] + R, self.pos[i][1] + R] += 1
        return self.observe()

    def _dist(self, i):
        return float(self.gdist[i][tuple(self.pos[i])])

    def observe(self):
        pad = np.pad(self.grid, R, constant_values=1)
        others = np.zeros_like(pad)
        for j in range(self.n):
            if not self.done[j]:
                others[self.pos[j][0] + R, self.pos[j][1] + R] = 1
        goalmap = np.zeros_like(pad)
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        for i in range(self.n):
            r, c = self.pos[i] + R
            g = goalmap.copy()
            g[self.goal[i][0] + R, self.goal[i][1] + R] = 1
            o = others[r - R : r + R + 1, c - R : c + R + 1].copy()
            o[R, R] = 0
            dv = (self.goal[i] - self.pos[i]) / self.size
            obs[i] = np.concatenate([
                pad[r - R : r + R + 1, c - R : c + R + 1].ravel(),
                o.ravel(),
                g[r - R : r + R + 1, c - R : c + R + 1].ravel(),
                np.minimum(self.visits[i, r - R : r + R + 1, c - R : c + R + 1], 3).ravel() / 3,
                [dv[0], dv[1], np.linalg.norm(dv)],
            ])
        return obs

    def step(self, actions):
        self.t += 1
        rew = np.zeros(self.n, dtype=np.float32)
        old_d = np.array([self._dist(i) for i in range(self.n)])
        prop = self.pos.copy()
        for i in range(self.n):
            if self.done[i]:
                continue
            p = self.pos[i] + ACTIONS[actions[i]]
            if not (0 <= p[0] < self.size and 0 <= p[1] < self.size) or self.grid[p[0], p[1]]:
                rew[i] -= 0.5
                self.collisions += 1
            else:
                prop[i] = p
        active = [i for i in range(self.n) if not self.done[i]]
        blocked = set()
        for a in active:
            for b in active:
                if a < b:
                    same = (prop[a] == prop[b]).all()
                    swap = (prop[a] == self.pos[b]).all() and (prop[b] == self.pos[a]).all()
                    if same or swap:
                        blocked |= {a, b}
        for i in blocked:
            prop[i] = self.pos[i]
            rew[i] -= 0.5
        self.collisions += len(blocked) // 2
        for i in active:
            if (prop[i] != self.pos[i]).any():
                self.path_len[i] += 1
        self.pos = prop
        for i in active:
            self.visits[i, self.pos[i][0] + R, self.pos[i][1] + R] += 1
            rew[i] += -0.05 + 0.2 * (old_d[i] - self._dist(i))
            if (self.pos[i] == self.goal[i]).all():
                rew[i] += 2.0
                self.done[i] = True
        timeout = self.t >= self.max_steps
        return self.observe(), rew, self.done.copy(), timeout

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .env import MultiUAVGrid, OBS_DIM, ACTIONS
from .planners import astar, path_to_actions

DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
GRID, N_UAVS, MAX_STEPS = 12, 3, 60
torch.set_num_threads(1)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, len(ACTIONS))

    def forward(self, x):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


def client_density(partition_id, non_iid):
    return DENSITIES[partition_id % len(DENSITIES)] if non_iid else None


def epsilon(rnd, total):
    return max(0.05, 1.0 - rnd / (0.6 * total))


def collect(net, episodes, eps, density, seed):
    rng = np.random.default_rng(seed)
    env = MultiUAVGrid(GRID, N_UAVS, density or 0.15, MAX_STEPS, seed)
    data, rets, succ = [], [], []
    for _ in range(episodes):
        if density is None:
            env.density = float(rng.choice(DENSITIES))
        obs = env.reset()
        ep_ret = 0.0
        while True:
            active = ~env.done
            with torch.no_grad():
                q = net(torch.from_numpy(obs))
            act = q.argmax(1).numpy()
            rnd = rng.random(N_UAVS) < eps
            act[rnd] = rng.integers(0, len(ACTIONS), rnd.sum())
            nobs, rew, done, timeout = env.step(act)
            for i in np.where(active)[0]:
                data.append((obs[i], act[i], rew[i], nobs[i], float(done[i])))
                ep_ret += rew[i]
            obs = nobs
            if done.all() or timeout:
                break
        rets.append(ep_ret / N_UAVS)
        succ.append(env.done.mean())
    return data, float(np.mean(rets)), float(np.mean(succ))


def train(net, data, steps, lr, gamma=0.95, batch=64, mu=0.0, seed=0):
    rng = np.random.default_rng(seed)
    target = copy.deepcopy(net).eval()
    anchor = [p.detach().clone() for p in net.parameters()]
    o, a, r, no, d = (torch.tensor(np.array(x)) for x in zip(*data))
    a = a.long()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    total = 0.0
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(o), batch))
        q = net(o[idx]).gather(1, a[idx, None]).squeeze(1)
        with torch.no_grad():
            best = net(no[idx]).argmax(1, keepdim=True)
            tq = r[idx] + gamma * (1 - d[idx]) * target(no[idx]).gather(1, best).squeeze(1)
        loss = F.smooth_l1_loss(q, tq)
        if mu > 0:
            loss = loss + mu / 2 * sum(((p - w) ** 2).sum() for p, w in zip(net.parameters(), anchor))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if (s + 1) % 100 == 0:
            target.load_state_dict(net.state_dict())
        total += loss.item()
    return total / steps


def test_set(n=60, seed=12345):
    env = MultiUAVGrid(GRID, N_UAVS, 0.15, MAX_STEPS, seed)
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        env.density = float(rng.choice(DENSITIES))
        env.reset()
        cases.append((env.grid.copy(), env.pos.copy(), env.goal.copy()))
    return cases


def _run_case(case, policy):
    grid, pos, goal = case
    env = MultiUAVGrid(GRID, N_UAVS, 0.15, MAX_STEPS)
    env.reset(grid.copy())
    env.pos, env.goal = pos.copy(), goal.copy()
    obs = env.observe()
    while True:
        obs, _, done, timeout = env.step(policy(env, obs))
        if done.all() or timeout:
            break
    opt = np.array([len(astar(grid, pos[i], goal[i])) - 1 for i in range(N_UAVS)])
    ok = env.done
    ratio = (env.path_len[ok] / np.maximum(opt[ok], 1)).mean() if ok.any() else np.nan
    return ok.mean(), float(ok.all()), env.collisions, ratio


def evaluate(net, cases, eps=0.05, seed=7):
    net.eval()
    rng = np.random.default_rng(seed)

    def pol(env, obs):
        with torch.no_grad():
            act = net(torch.from_numpy(obs)).argmax(1).numpy()
        rnd = rng.random(len(act)) < eps
        act[rnd] = rng.integers(0, len(ACTIONS), rnd.sum())
        return act

    res = np.array([_run_case(c, pol) for c in cases], dtype=float)
    return {
        "uav_success": float(res[:, 0].mean()),
        "mission_success": float(res[:, 1].mean()),
        "collisions": float(res[:, 2].mean()),
        "path_ratio": float(np.nanmean(res[:, 3])),
    }


def evaluate_astar(cases):
    def run(case):
        grid, pos, goal = case
        plans = [path_to_actions(astar(grid, pos[i], goal[i])) for i in range(N_UAVS)]
        step = [0]

        def pol(env, obs):
            k = step[0]
            step[0] += 1
            return np.array([p[k] if k < len(p) else 0 for p in plans])

        return _run_case(case, pol)

    res = np.array([run(c) for c in cases], dtype=float)
    return {
        "uav_success": float(res[:, 0].mean()),
        "mission_success": float(res[:, 1].mean()),
        "collisions": float(res[:, 2].mean()),
        "path_ratio": float(np.nanmean(res[:, 3])),
    }

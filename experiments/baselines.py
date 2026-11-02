import json
import sys
import time
import numpy as np
import torch
from uavfl.task import QNet, client_density, collect, epsilon, train, evaluate, evaluate_astar, test_set

ROUNDS, N, EPISODES, STEPS, LR = 40, 5, 20, 500, 5e-4
EVAL_EVERY = 5


def local_only(cases):
    torch.manual_seed(0)
    nets = [QNet() for _ in range(N)]
    hist, t0 = [], time.time()
    for r in range(1, ROUNDS + 1):
        for pid, net in enumerate(nets):
            data, _, _ = collect(net, EPISODES, epsilon(r, ROUNDS), client_density(pid, True), 1000 * pid + r)
            train(net, data, STEPS, LR, seed=1000 * pid + r)
        if r % EVAL_EVERY == 0:
            ms = [evaluate(n, cases) for n in nets]
            m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
            m["worst_uav_success"] = float(min(x["uav_success"] for x in ms))
            m["round"], m["time"] = r, time.time() - t0
            hist.append(m)
            print("local", m, flush=True)
    return hist


def centralized(cases):
    torch.manual_seed(0)
    net = QNet()
    hist, t0 = [], time.time()
    for r in range(1, ROUNDS + 1):
        data = []
        for pid in range(N):
            d, _, _ = collect(net, EPISODES, epsilon(r, ROUNDS), client_density(pid, True), 1000 * pid + r)
            data += d
        train(net, data, STEPS, LR, seed=r)
        if r % EVAL_EVERY == 0:
            m = evaluate(net, cases)
            m["round"], m["time"] = r, time.time() - t0
            hist.append(m)
            print("central", m, flush=True)
    return hist


if __name__ == "__main__":
    cases = test_set(60)
    which = sys.argv[1]
    if which == "astar":
        out = {"name": "astar", "metrics": evaluate_astar(cases)}
    elif which == "local":
        out = {"name": "local-only", "history": local_only(cases)}
    else:
        out = {"name": "centralized", "history": centralized(cases)}
    json.dump(out, open(f"results/{which}.json", "w"), indent=1)

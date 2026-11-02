import json, torch, numpy as np
from uavfl.task import QNet, collect, epsilon, client_density
from uavfl.env import OBS_DIM
net = QNet(); net.load_state_dict(torch.load("results/fedavg.pt"))
params = sum(p.numel() for p in net.parameters())
model_mb = params * 4 / 1e6
n = []
for r in range(1, 41):
    for pid in range(5):
        d, _, _ = collect(net, 20, epsilon(r, 40), client_density(pid, True), 1000 * pid + r)
        n.append(len(d))
trans_bytes = (2 * OBS_DIM + 3) * 4
fed = 40 * 5 * 2 * model_mb
cen = sum(n) * trans_bytes / 1e6
out = {"params": params, "model_mb": model_mb, "avg_transitions_per_client_round": float(np.mean(n)),
       "federated_total_mb": fed, "centralized_raw_data_mb": cen}
json.dump(out, open("results/comm.json", "w"), indent=1); print(out)

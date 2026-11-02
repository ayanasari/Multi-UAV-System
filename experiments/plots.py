import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Inter", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COL = {"FedAvg (Flower)": "#2F5D9A", "FedProx (Flower)": "#3B7D3B", "Local-only": "#C77A1E", "Centralized": "#7A8798"}
FILES = {"FedAvg (Flower)": "fedavg", "FedProx (Flower)": "fedprox", "Local-only": "local", "Centralized": "centralized"}


def load(f):
    p = f"results/{f}.json"
    return json.load(open(p)) if os.path.exists(p) else None


astar = load("astar")["metrics"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, f in FILES.items():
    d = load(f)
    if not d:
        continue
    h = [x for x in d["history"] if x["round"] > 0]
    r = [x["round"] for x in h]
    axes[0].plot(r, [x["uav_success"] * 100 for x in h], "-o", ms=3, lw=2, color=COL[name], label=name)
    c = [x["collisions"] for x in h]
    k = 5 if len(c) > 20 else 1
    sm = [sum(c[max(0, i - k + 1): i + 1]) / len(c[max(0, i - k + 1): i + 1]) for i in range(len(c))]
    axes[1].plot(r, sm, "-", lw=2, color=COL[name], label=name)
axes[0].axhline(astar["uav_success"] * 100, ls="--", color="#B3261E", lw=1.5, label="Independent A* (full map)")
axes[1].axhline(astar["collisions"], ls="--", color="#B3261E", lw=1.5, label="Independent A* (full map)")
axes[0].set(title="UAVs reaching their goal (%)", xlabel="Round", ylim=(0, 100))
axes[1].set(title="Collisions per mission (lower is better, smoothed)", xlabel="Round", ylim=(0, 7))
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/learning_curves.png", dpi=170)

rows = [("Independent A* (full map)", astar)]
for name, f in FILES.items():
    d = load(f)
    if d:
        last = d["history"][-3:]
        rows.append((name, {k: sum(x[k] for x in last) / len(last) for k in ["uav_success", "mission_success", "collisions", "path_ratio"]}))
with open("results/summary.md", "w") as fh:
    fh.write("| Method | UAV success | Mission success | Collisions | Path length / optimal |\n|---|---|---|---|---|\n")
    for n, m in rows:
        fh.write(f"| {n} | {m['uav_success']*100:.1f}% | {m['mission_success']*100:.1f}% | {m['collisions']:.2f} | {m['path_ratio']:.2f} |\n")
print(open("results/summary.md").read())

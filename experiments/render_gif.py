import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from uavfl.env import MultiUAVGrid, ACTIONS
from uavfl.task import QNet, test_set, GRID, N_UAVS, MAX_STEPS

COLORS = ["#2F5D9A", "#C77A1E", "#3B7D3B"]


def frame(env, trails, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.6), dpi=90)
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.set_aspect("equal"); ax.axis("off")
    for r, c in np.argwhere(env.grid):
        ax.add_patch(Rectangle((c, r), 1, 1, color="#4A5A70"))
    ax.add_patch(Rectangle((0, 0), GRID, GRID, fill=False, lw=1.5, ec="#1B2A41"))
    for i in range(N_UAVS):
        g = env.goal[i]
        ax.add_patch(Rectangle((g[1] + .15, g[0] + .15), .7, .7, fill=False, lw=2.2, ec=COLORS[i]))
        t = np.array(trails[i])
        ax.plot(t[:, 1] + .5, t[:, 0] + .5, color=COLORS[i], lw=2, alpha=.5)
        p = env.pos[i]
        ax.add_patch(Circle((p[1] + .5, p[0] + .5), .35, color=COLORS[i]))
    ax.set_title(title, fontsize=10)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


def main(model_path, out, n_cases=4):
    net = QNet(); net.load_state_dict(torch.load(model_path)); net.eval()
    rng = np.random.default_rng(3)
    frames, ends = [], []
    for k, (grid, pos, goal) in enumerate(test_set(n_cases, seed=4242)):
        env = MultiUAVGrid(GRID, N_UAVS, 0.15, MAX_STEPS)
        env.reset(grid.copy()); env.pos, env.goal = pos.copy(), goal.copy()
        trails = [[tuple(p)] for p in env.pos]
        obs = env.observe()
        for t in range(MAX_STEPS):
            frames.append(frame(env, trails, f"Mission {k+1}  ·  step {t}  ·  shared Flower policy"))
            with torch.no_grad():
                a = net(torch.from_numpy(obs)).argmax(1).numpy()
            m = rng.random(N_UAVS) < .05
            a[m] = rng.integers(0, len(ACTIONS), m.sum())
            obs, _, done, timeout = env.step(a)
            for i in range(N_UAVS):
                trails[i].append(tuple(env.pos[i]))
            if done.all() or timeout:
                break
        end = frame(env, trails, f"Mission {k+1}  ·  {int(env.done.sum())}/{N_UAVS} UAVs reached goal")
        ends.append(end)
        frames += [end] * 6
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=160, loop=0)
    w, h = ends[0].size
    strip = Image.new("RGB", (w * len(ends), h), "white")
    for k, e in enumerate(ends):
        strip.paste(e, (k * w, 0))
    strip.save(out.replace(".gif", "_strip.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

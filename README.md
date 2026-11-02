# Distributed Path Planning for Multi-UAV Systems with Flower

Course project: Distributed Analytics for Industrial Automation (IM60210), IIT Kharagpur.

Several UAVs learn **one shared path-planning policy** with **federated reinforcement learning**. Each UAV is a Flower `ClientApp`. It flies training episodes in its own area and trains a DQN locally, then sends only its model weights to a ground-station `ServerApp`, which combines them with FedAvg or FedProx. Raw flight data never leaves a UAV. At run time, every UAV plans its own moves from a local 7×7 view, so there is no central planner.

## Setup

| Item | Value |
|---|---|
| World | 12×12 grid, random obstacles, 3 UAVs per mission, each with its own start and goal |
| Observation | Local 7×7 window (obstacles, other UAVs, goal, visited cells) + goal direction |
| Actions | stay / N / S / W / E |
| Reward | +2 goal, −0.5 collision, −0.05 per step, + progress toward goal |
| Learner | Double DQN, MLP 199→128→128→5; at the start of each round the target network is the global model |
| Federation | 5 UAV clients, 40 rounds, 20 episodes + 500 gradient steps per client per round |
| Non-IID | Client *i* trains in areas with obstacle density 5%, 10%, 15%, 20% or 25% |
| Test | 60 held-out missions with mixed densities |

## Run

```bash
pip install -e .
flwr run . --stream                                   # FedAvg
flwr run . --stream --run-config 'prox-mu=0.01 name="fedprox" out="results/fedprox.json"'
PYTHONPATH=. python experiments/baselines.py astar    # also: local, central
PYTHONPATH=. python experiments/plots.py
PYTHONPATH=. python experiments/render_gif.py results/fedavg.pt figures/uav_demo.gif
```

## Layout

```
uavfl/env.py          multi-UAV grid world (collisions, observations, rewards)
uavfl/planners.py     A* baseline
uavfl/task.py         Q-network, experience collection, DQN training, evaluation
uavfl/client_app.py   Flower ClientApp (one per UAV)
uavfl/server_app.py   Flower ServerApp (FedAvg / FedProx + global evaluation)
experiments/          baselines, plots, demo GIF
results/  figures/    outputs
```

## Results so far

See `results/summary.md` and `figures/learning_curves.png`.

import json
import os
import time
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from .task import QNet, evaluate, test_set

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    num_rounds = int(rc["num-server-rounds"])
    torch.manual_seed(int(rc["seed"]))
    global_model = QNet()
    arrays = ArrayRecord(global_model.state_dict())
    mu = float(rc["prox-mu"])
    common = dict(fraction_train=float(rc["fraction-train"]), fraction_evaluate=1.0,
                  min_train_nodes=1, min_available_nodes=int(rc["num-uavs"]))
    strategy = FedProx(proximal_mu=mu, **common) if mu > 0 else FedAvg(**common)
    cases = test_set(60)
    history, t0 = [], time.time()

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = QNet()
        net.load_state_dict(arrays.to_torch_state_dict())
        m = evaluate(net, cases)
        m["round"], m["time"] = server_round, time.time() - t0
        history.append(m)
        return MetricRecord({k: float(v) for k, v in m.items()})

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": float(rc["learning-rate"])}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    out = rc["out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"name": rc["name"], "history": history}, open(out, "w"), indent=1)
    torch.save(result.arrays.to_torch_state_dict(), out.replace(".json", ".pt"))

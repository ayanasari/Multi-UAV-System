import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .task import QNet, client_density, collect, epsilon, train as train_fn, evaluate, test_set

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = QNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    cfg = msg.content["config"]
    pid = context.node_config["partition-id"]
    rc = context.run_config
    rnd, total = int(cfg["server-round"]), int(rc["num-server-rounds"])
    density = client_density(pid, bool(rc["non-iid"]))
    seed = 1000 * pid + rnd
    data, ret, succ = collect(model, int(rc["local-episodes"]), epsilon(rnd, total), density, seed)
    loss = train_fn(model, data, int(rc["local-steps"]), float(cfg["lr"]), mu=float(rc["prox-mu"]), seed=seed)
    metrics = {"train_loss": loss, "train_return": ret, "train_success": succ, "num-examples": len(data)}
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_(msg: Message, context: Context):
    model = QNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    pid = context.node_config["partition-id"]
    cases = test_set(20, seed=777 + pid)
    m = evaluate(model, cases)
    metrics = {"eval_uav_success": m["uav_success"], "eval_collisions": m["collisions"], "num-examples": len(cases)}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

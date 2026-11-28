from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ParquetBatchDataset, load_data_spec, load_lightgcn_graph, split_path
from .model import build_model
from .utils import atomic_json_dump, resolve_device, set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int | None]:
    model.eval()
    labels, scores = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        labels.append(y.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    if not labels:
        return {"rows": 0, "auc": None, "logloss": None}
    y_true, y_score = np.concatenate(labels), np.concatenate(scores)
    auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None
    return {"rows": int(y_true.size), "auc": auc, "logloss": float(log_loss(y_true, y_score, labels=[0, 1]))}


def run_experiment(config: ExperimentConfig) -> dict:
    set_seed(config.seed)
    features, vocab_sizes = load_data_spec(config.data_dir)
    device = resolve_device(config.device)
    run_name = f"{config.model}__{config.task}__{config.regime}__seed{config.seed}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), run_dir / "config.json")

    effective_batch_size = config.lightgcn_batch_size if config.model.lower() == "lightgcn" else config.batch_size
    datasets = {
        split: ParquetBatchDataset(
            split_path(config.data_dir, config.regime, config.task, split), features,
            config.task, effective_batch_size, split == "train", config.seed,
            drop_singleton=split == "train",
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        split: DataLoader(ds, batch_size=None, num_workers=config.num_workers, pin_memory=device.type == "cuda")
        for split, ds in datasets.items()
    }
    graph_metadata = None
    edge_users = edge_items = None
    if config.model.lower() == "lightgcn":
        edge_users, edge_items, graph_metadata = load_lightgcn_graph(config.data_dir, config.task)
        if not edge_users.numel():
            raise ValueError(f"LightGCN graph for {config.task} has no train-positive edges")
    model = build_model(
        config.model,
        vocab_sizes,
        config.embedding_dim,
        config.hidden_dims,
        edge_users=edge_users,
        edge_items=edge_items,
        lightgcn_layers=config.lightgcn_layers,
    ).to(device)
    print(
        f"[START] model={config.model} task={config.task} regime={config.regime} "
        f"device={device} train_batch_size={effective_batch_size}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, stale, history = -float("inf"), 0, []
    started = time.time()
    checkpoint = run_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        total_loss, total_rows = 0.0, 0
        for step, (x, y) in enumerate(loaders["train"], start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            total_rows += y.numel()
            if config.max_train_steps and step >= config.max_train_steps:
                break
        valid = evaluate(model, loaders["valid"], device)
        record = {"epoch": epoch, "train_rows": total_rows, "train_loss": total_loss / max(total_rows, 1), **{f"valid_{k}": v for k, v in valid.items()}}
        history.append(record)
        atomic_json_dump(history, run_dir / "history.json")
        print(
            f"[EPOCH {epoch}/{config.epochs}] model={config.model} task={config.task} "
            f"regime={config.regime} train_loss={record['train_loss']:.6f} "
            f"valid_auc={valid['auc']} valid_logloss={valid['logloss']}",
            flush=True,
        )
        score = valid["auc"] if valid["auc"] is not None else -float("inf")
        if score > best_auc:
            best_auc, stale = score, 0
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "features": features, "vocab_sizes": vocab_sizes}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
    if not checkpoint.exists():
        torch.save({"model": model.state_dict(), "config": config.to_dict(), "features": features, "vocab_sizes": vocab_sizes}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    test = evaluate(model, loaders["test"], device)
    result = {
        "run_name": run_name, "model": config.model, "task": config.task, "regime": config.regime,
        "seed": config.seed, "best_valid_auc": None if best_auc == -float("inf") else best_auc,
        **{f"test_{k}": v for k, v in test.items()}, "epochs_ran": len(history),
        "lightgcn_graph": graph_metadata,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json_dump(result, run_dir / "metrics.json")
    print(
        f"[DONE] model={config.model} task={config.task} regime={config.regime} "
        f"test_auc={result['test_auc']} test_logloss={result['test_logloss']} "
        f"elapsed_seconds={result['elapsed_seconds']}",
        flush=True,
    )
    return result

"""GNN training loop: AdamW + ReduceLROnPlateau + early stopping on val Pearson R."""

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plb.data.dataset import PDBbindGraphDataset, make_dataloader
from plb.eval import format_metrics_row, regression_metrics
from plb.models.gnn import AffinityModel, GNNConfig

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:

    model: GNNConfig = field(default_factory=GNNConfig)
    batch_size: int = 64
    max_epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    early_stop_patience: int = 15
    seed: int = 42
    device: str = "auto"
    amp: bool = False
    deterministic: bool = False
    num_workers: int = 0
    name: str = "gnn_default"
    notes: str = ""


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def train_one_epoch(
    model: AffinityModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    loss_fn = torch.nn.MSELoss(reduction="sum")
    total_loss, n_samples = 0.0, 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.esm_pocket,
                    batch.esm_whole,
                )
                loss = loss_fn(pred, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.esm_pocket,
                batch.esm_whole,
            )
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach())
        n_samples += int(batch.y.numel())

    return {"loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def evaluate(
    model: AffinityModel,
    loader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    pdb_ids: list[str] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pred = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.esm_pocket,
            batch.esm_whole,
        )
        y_true_chunks.append(batch.y.detach().cpu().numpy())
        y_pred_chunks.append(pred.detach().cpu().numpy())
        ids = batch.pdb_id
        if isinstance(ids, str):
            pdb_ids.append(ids)
        else:
            pdb_ids.extend(ids)

    y_true = np.concatenate(y_true_chunks).ravel()
    y_pred = np.concatenate(y_pred_chunks).ravel()
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean((y_pred - y_true) ** 2))
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "pdb_ids": pdb_ids}


def _save_run_config(cfg: TrainConfig, path: Path) -> None:
    blob = asdict(cfg)
    path.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")


def train(
    cfg: TrainConfig,
    train_ds: PDBbindGraphDataset,
    val_ds: PDBbindGraphDataset,
    test_ds: PDBbindGraphDataset | None = None,
    run_dir: Path | str = Path("runs") / "gnn",
) -> dict[str, object]:
    run_root = Path(run_dir) / cfg.name
    run_root.mkdir(parents=True, exist_ok=True)
    _save_run_config(cfg, run_root / "config.json")

    _set_seed(cfg.seed, cfg.deterministic)
    device = _resolve_device(cfg.device)
    log.info(
        "training '%s' on %s (n_train=%d, n_val=%d)", cfg.name, device, len(train_ds), len(val_ds)
    )

    train_loader = make_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = make_dataloader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = AffinityModel(cfg.model).to(device)
    log.info("model parameters: %d", model.n_parameters())

    # TODO: try cosine annealing - plateau scheduler is finicky with small val sets
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    scaler = torch.amp.GradScaler(device="cuda") if cfg.amp and device.type == "cuda" else None

    log_path = run_root / "log.jsonl"
    best_path = run_root / "best.pt"
    best_pearson = -float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_eval = evaluate(model, val_loader, device)
            val_metrics = val_eval["metrics"]
            scheduler.step(val_metrics["pearson_r"])
            elapsed = time.perf_counter() - t0

            improved = val_metrics["pearson_r"] > best_pearson + 1e-4
            if improved:
                best_pearson = val_metrics["pearson_r"]
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_cfg": asdict(cfg.model),
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
            else:
                epochs_since_improve += 1

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_pearson_r": val_metrics["pearson_r"],
                "val_spearman_r": val_metrics["spearman_r"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "elapsed_sec": elapsed,
                "improved": improved,
            }
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

            log.info(
                "ep %3d/%d | train MSE %.3f | val MSE %.3f | %s | %.1fs%s",
                epoch,
                cfg.max_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                format_metrics_row(val_metrics),
                elapsed,
                "  *" if improved else "",
            )

            if epochs_since_improve >= cfg.early_stop_patience:
                log.info(
                    "early stop at epoch %d (no val improvement for %d epochs)",
                    epoch,
                    cfg.early_stop_patience,
                )
                break

    summary: dict[str, object] = {
        "run_dir": str(run_root),
        "best_epoch": best_epoch,
        "best_val_pearson_r": best_pearson,
        "log_path": str(log_path),
        "best_checkpoint": str(best_path),
    }

    if test_ds is not None and best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loader = make_dataloader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        test_eval = evaluate(model, test_loader, device)
        np.savez(
            run_root / "test_predictions.npz",
            y_true=test_eval["y_true"],
            y_pred=test_eval["y_pred"],
            pdb_ids=np.asarray(test_eval["pdb_ids"]),
        )
        (run_root / "metrics.json").write_text(
            json.dumps(
                {
                    "val": {k: float(v) for k, v in ckpt["val_metrics"].items()},
                    "test": {k: float(v) for k, v in test_eval["metrics"].items()},
                    "best_epoch": best_epoch,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "n_test": len(test_ds),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["test_metrics"] = test_eval["metrics"]
        summary["test_predictions"] = str(run_root / "test_predictions.npz")
        log.info("CASF-2016 test  | %s", format_metrics_row(test_eval["metrics"]))

    return summary

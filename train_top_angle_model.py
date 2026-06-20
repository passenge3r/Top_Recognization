import argparse
import csv
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from top_angle_estimator import (
    IMAGE_EXTENSIONS,
    build_resnet50_regression,
    get_preprocessor,
    parse_label_from_filename,
)


@dataclass
class Sample:
    path: Path
    label: float
    split: str


def iter_labeled_images(root: Path) -> list[tuple[Path, float]]:
    images = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )
    labeled: list[tuple[Path, float]] = []
    for path in images:
        label = parse_label_from_filename(path)
        if label is not None:
            labeled.append((path, label))
    return labeled


def label_bin(label: float) -> int:
    if label < 10:
        return 0
    if label >= 70:
        return 8
    return int(label // 10)


def split_samples(
    labeled: list[tuple[Path, float]],
    val_ratio: float,
    seed: int,
) -> list[Sample]:
    rng = random.Random(seed)
    by_bin: dict[int, list[tuple[Path, float]]] = {}
    for path, label in labeled:
        by_bin.setdefault(label_bin(label), []).append((path, label))

    samples: list[Sample] = []
    for _, items in sorted(by_bin.items()):
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_ratio))) if len(items) >= 4 else 0
        for i, (path, label) in enumerate(items):
            split = "val" if i < val_count else "train"
            samples.append(Sample(path=path, label=label, split=split))
    return sorted(samples, key=lambda sample: str(sample.path).lower())


def prepare_tensors(samples: list[Sample], preprocessor, device):
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensors = []
    labels = []
    kept_samples = []
    failed = []
    for sample in samples:
        roi, _ = preprocessor.process(str(sample.path), view_type="top")
        if roi is None:
            failed.append(sample.path)
            continue
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb)))
        labels.append(sample.label)
        kept_samples.append(sample)

    if not tensors:
        raise RuntimeError("No trainable images after preprocessing.")

    x = torch.stack(tensors).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    return x, y, kept_samples, failed


def extract_features(model, x, batch_size: int):
    import torch
    import torch.nn as nn

    original_fc = model.fc
    model.fc = nn.Identity()
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            features.append(model(x[start : start + batch_size]))
    model.fc = original_fc
    return torch.cat(features, dim=0)


def mae(pred, target) -> float:
    return float((pred - target).abs().mean().item())


def weighted_smooth_l1(pred, target, edge_weight: float):
    import torch.nn.functional as F

    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if edge_weight <= 1.0:
        return loss.mean()
    weights = ((target <= 10.0) | (target >= 70.0)).float() * (edge_weight - 1.0) + 1.0
    return (loss * weights).mean()


def set_trainable_layers(model, train_mode: str) -> list:
    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return list(model.parameters())

    if train_mode == "layer4":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        return [param for param in model.parameters() if param.requires_grad]

    if train_mode == "head":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        return list(model.fc.parameters())

    raise ValueError(f"Unknown train mode: {train_mode}")


def train(args):
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    labeled = iter_labeled_images(data_dir)
    if len(labeled) < 10:
        raise RuntimeError(f"Not enough labeled images in {data_dir}: {len(labeled)}")

    samples = split_samples(labeled, args.val_ratio, args.seed)
    preprocessor = get_preprocessor(Path(args.top_model), Path(args.side_model))
    x, y, kept_samples, failed = prepare_tensors(samples, preprocessor, device)

    model = build_resnet50_regression(device)
    state = torch.load(str(args.init_model), map_location=device)
    model.load_state_dict(state)

    train_idx = torch.tensor(
        [i for i, sample in enumerate(kept_samples) if sample.split == "train"],
        dtype=torch.long,
        device=device,
    )
    val_idx = torch.tensor(
        [i for i, sample in enumerate(kept_samples) if sample.split == "val"],
        dtype=torch.long,
        device=device,
    )
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise RuntimeError("Train/val split is empty after preprocessing.")

    trainable_params = set_trainable_layers(model, args.train_mode)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best = {
        "epoch": 0,
        "val_mae": math.inf,
        "train_mae": math.inf,
        "state": None,
    }
    rows = []

    train_y = y[train_idx]
    val_y = y[val_idx]

    if args.train_mode == "head":
        features = extract_features(model, x, args.batch_size)
        train_x = features[train_idx]
        val_x = features[val_idx]
        train_step_model = model.fc
        eval_model = model.fc
    else:
        train_x = x[train_idx]
        val_x = x[val_idx]
        train_step_model = model
        eval_model = model

    for epoch in range(1, args.epochs + 1):
        train_step_model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        epoch_loss = 0.0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = train_step_model(train_x[idx])
            loss = weighted_smooth_l1(pred, train_y[idx], args.edge_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * idx.numel()
        scheduler.step()

        eval_model.eval()
        with torch.no_grad():
            train_mae = float("nan")
            if args.train_mae_every > 0 and (
                epoch == 1 or epoch % args.train_mae_every == 0 or epoch == args.epochs
            ):
                train_pred_parts = []
                for start in range(0, train_x.shape[0], args.eval_batch_size):
                    train_pred_parts.append(eval_model(train_x[start : start + args.eval_batch_size]))
                train_pred = torch.cat(train_pred_parts, dim=0)
                train_mae = mae(train_pred, train_y)
            val_pred_parts = []
            for start in range(0, val_x.shape[0], args.eval_batch_size):
                val_pred_parts.append(eval_model(val_x[start : start + args.eval_batch_size]))
            val_pred = torch.cat(val_pred_parts, dim=0)
            val_mae = mae(val_pred, val_y)

        avg_loss = epoch_loss / float(train_x.shape[0])
        lr = scheduler.get_last_lr()[0]
        rows.append([epoch, avg_loss, train_mae, val_mae, lr])
        if val_mae < best["val_mae"]:
            best = {
                "epoch": epoch,
                "val_mae": val_mae,
                "train_mae": train_mae,
                "state": deepcopy(model.state_dict()),
            }
            torch.save(best["state"], output_model)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"loss={avg_loss:.4f} train_mae={train_mae:.4f} "
                f"val_mae={val_mae:.4f} best_epoch={best['epoch']} "
                f"best_val_mae={best['val_mae']:.4f}"
            )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_mae", "val_mae", "lr"])
        writer.writerows(rows)

    split_path = Path(args.split_csv)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "split"])
        for sample in kept_samples:
            writer.writerow([str(sample.path.relative_to(data_dir)), f"{sample.label:.4f}", sample.split])

    print(f"device: {device}")
    print(f"train_mode: {args.train_mode}")
    print(f"labeled: {len(labeled)}, used: {len(kept_samples)}, failed: {len(failed)}")
    print(f"train: {train_idx.numel()}, val: {val_idx.numel()}")
    print(f"best_epoch: {best['epoch']}")
    print(f"best_train_mae: {best['train_mae']:.4f}")
    print(f"best_val_mae: {best['val_mae']:.4f}")
    print(f"saved_model: {output_model.resolve()}")
    print(f"log_csv: {log_path.resolve()}")
    print(f"split_csv: {split_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train the top-angle ResNet50 regression model.")
    parser.add_argument("--data-dir", default="data_samples/top")
    parser.add_argument("--top-model", default="top_best.pt")
    parser.add_argument("--side-model", default="side_best.pt")
    parser.add_argument("--init-model", default="outputs/top_angle_resnet50_retrained_300_best.pth")
    parser.add_argument("--output-model", default="outputs/top_angle_resnet50_retrained_350_best.pth")
    parser.add_argument("--log-csv", default="outputs/top_angle_train_350_log.csv")
    parser.add_argument("--split-csv", default="outputs/top_angle_train_350_split.csv")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--train-mae-every",
        type=int,
        default=25,
        help="Compute train MAE every N epochs. Use 0 to skip it except validation.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=1.5)
    parser.add_argument(
        "--train-mode",
        choices=["head", "layer4", "full"],
        default="head",
        help="head caches backbone features; layer4 unfreezes the final block; full updates all layers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

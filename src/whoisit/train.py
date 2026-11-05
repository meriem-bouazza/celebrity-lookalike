"""One training run. The result goes to results/runs/<loss>_n<N>_seed<S>_ep<E>/.

Example: python -m whoisit.train --loss ce --n 5 --seed 42 --epochs 3
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from whoisit.data.dataset import (load_aligned, load_train_people, load_val_photos,
                                  pick_aligned_photos)
from whoisit.data.images import FaceDataset
from whoisit.data.splits import SEED, load_identities
from whoisit.evaluate import top_k_accuracy
from whoisit.models.backbone import Backbone
from whoisit.models.heads import ARCFACE_MARGIN, ARCFACE_SCALE, HEADS, make_head
from whoisit.utils import set_seed

RUNS_DIR = Path("results/runs")
RESULT_FILE = "result.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=sorted(HEADS), default="ce")
    parser.add_argument("--n", type=int, default=5, help="photos per person")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED, help="only changes the training, not the photos")
    # in the container the shared memory (/dev/shm) is only 63 MB, more workers crash
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="run again if the result already exists")
    parser.add_argument("--margin", type=float, default=ARCFACE_MARGIN, help="only for arcface")
    parser.add_argument("--scale", type=float, default=ARCFACE_SCALE, help="only for arcface")
    return parser.parse_args(argv)


def run_folder(loss, n, seed, epochs):
    return RUNS_DIR / f"{loss}_n{n}_seed{seed}_ep{epochs}"


@torch.no_grad()
def validate(model, loader, device):
    # eval mode, so BatchNorm does not change and does not depend on the batch
    model.eval()
    parts = [model(images.to(device)).cpu().numpy() for images, _ in loader]
    model.train()
    accuracy = top_k_accuracy(np.concatenate(parts), loader.dataset.labels)
    return accuracy[1], accuracy[5]


def save_result(folder, result):
    # write to a temp file first, so a crash never leaves half a result.json
    temp = folder / (RESULT_FILE + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, folder / RESULT_FILE)


def main(argv=None):
    args = parse_args(argv)
    folder = run_folder(args.loss, args.n, args.seed, args.epochs)
    if (folder / RESULT_FILE).exists() and not args.force:
        print(f"skip {folder}: {RESULT_FILE} exists (use --force to run again)")
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # no seed here on purpose: the photos stay the same for every run seed
    aligned = load_aligned()
    photos = pick_aligned_photos(load_identities(), load_train_people(), args.n, aligned)
    data = FaceDataset(photos, train=True)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    print(f"train: {data.num_people} people, {len(data)} photos, device {device}")

    val_data = FaceDataset(load_val_photos(aligned), train=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    print(f"val: {val_data.num_people} people, {len(val_data)} photos")

    model = Backbone().to(device)
    head_options = {"margin": args.margin, "scale": args.scale} if args.loss == "arcface" else {}
    head = make_head(args.loss, data.num_people, **head_options).to(device)
    loss_fn = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # the heads use different amounts of random numbers, so we set the seed again here:
    # now every loss sees the same order of photos and the same flips
    set_seed(args.seed)

    epochs = []
    for epoch in range(1, args.epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        total = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(head(model(images), labels), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        seconds = time.time() - start
        row = {"epoch": epoch, "loss": total / len(loader), "seconds": seconds}
        line = f"epoch {epoch}: loss {row['loss']:.3f}, {seconds:.0f} s"
        if device == "cuda":
            # read before validation, so this is the memory of the training only
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            line += f", peak {row['peak_gb']:.1f} GB"
        row["top1"], row["top5"] = validate(model, val_loader, device)
        print(f"{line}, val top 1 {row['top1']:.2%}, top 5 {row['top5']:.2%}")
        epochs.append(row)

    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoints / "model.pt")

    # result.json comes last: if it exists, the run is complete
    save_result(folder, {
        "options": {"loss": args.loss, "n": args.n, "seed": args.seed, "epochs": args.epochs,
                    "batch_size": args.batch_size, "lr": args.lr, **head_options},
        "train": {"people": data.num_people, "photos": len(data)},
        "val": {"people": val_data.num_people, "photos": len(val_data)},
        "epochs": epochs,
        "final": {"top1": epochs[-1]["top1"], "top5": epochs[-1]["top5"]},
    })
    print("saved", folder)


if __name__ == "__main__":
    main()

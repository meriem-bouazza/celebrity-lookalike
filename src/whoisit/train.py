"""Train the network with cross-entropy.

Example: python -m whoisit.train --n 5 --epochs 3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from whoisit.data.dataset import load_aligned, load_train_people, pick_aligned_photos
from whoisit.data.images import FaceDataset
from whoisit.data.splits import SEED, load_identities
from whoisit.models.backbone import Backbone
from whoisit.models.heads import SoftmaxHead

CHECKPOINT_DIR = Path("checkpoints")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="photos per person")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    photos = pick_aligned_photos(load_identities(), load_train_people(), args.n, load_aligned())
    data = FaceDataset(photos, train=True)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    print(f"{data.num_people} people, {len(data)} photos, device {device}")

    model = Backbone().to(device)
    head = SoftmaxHead(data.num_people).to(device)  # one value per person
    loss_fn = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
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
        print(f"epoch {epoch}: loss {total / len(loader):.3f}, {seconds:.0f} s")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"ce_n{args.n}_seed{args.seed}.pt"
    torch.save(model.state_dict(), path)
    print("saved", path)


if __name__ == "__main__":
    main()

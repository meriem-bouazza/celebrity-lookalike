from pathlib import Path

import pandas as pd

IDENTITY_FILE = Path("data/celeba/identity_CelebA.txt")
OUTPUT_FILE = Path("results/celeba_counts.txt")

df = pd.read_csv(IDENTITY_FILE, sep=" ", header=None, names=["image", "identity"])

# images per identity
counts = df.groupby("identity").size()

lines = [
    f"images: {len(df)}",
    f"identities: {counts.shape[0]}",
    counts.describe().to_string(),
]
for t in (2, 5, 10, 20, 25, 30):
    lines.append(f">= {t} images: {(counts >= t).sum()} identities")

report = "\n".join(lines)
print(report)

# save result
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
from pathlib import Path

LFW_DIR = Path("data/lfw/lfw")
OUTPUT_FILE = Path("results/lfw_counts.txt")

# one folder per person
counts = {p.name: len(list(p.glob("*.jpg"))) for p in LFW_DIR.iterdir() if p.is_dir()}
values = list(counts.values())

lines = [
    f"people: {len(values)}",
    f"images: {sum(values)}",
    f"max images per person: {max(values)}",
]
for t in (1, 2, 5, 10, 20):
    lines.append(f">= {t} images: {sum(1 for v in values if v >= t)} people")

report = "\n".join(lines)
print(report)

# save result
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
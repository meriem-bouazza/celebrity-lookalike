# Wer ist das? – Celebrity look-alike finder with face embeddings

Celebrity look-alike finder with face embeddings: comparing cross-entropy and ArcFace when few photos per person are available. University project (HSD) with webcam demo.

**Authors:** Meriem Bouazza, Melina Tsigka · **Module:** Advances in Intelligent Systems, Hochschule Düsseldorf · **Status:** in development

> Research question: Which training method yields the best face embeddings when only a few photos per person are available?

## How it works

`face detection + alignment (MTCNN) → embedding (ResNet-18) → cosine similarity to gallery → top-3 celebrities`

*Pipeline figure and demo screenshot: TODO.*

## Requirements

- Python **3.11** (3.12 should work; 3.13+ is not supported by the pinned `torch` 2.2.2)
- Tested on: Windows 11 (CPU). *More: TODO.*
- GPU optional for training; CPU is enough for the demo.

## Installation

```bash
git clone <repo-url>
cd celebrity-lookalike
py -3.11 -m venv .venv            # macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes `whoisit` importable
python scripts/check_env.py       # checks imports, MTCNN and a ResNet forward pass
pytest
```

## Dataset

Not included in this repository. Source, license and setup: see [docs/datasets.md](docs/datasets.md) (TODO).

## Usage

*TODO:* start the demo, reproduce evaluation, retrain (one command each).

## Results

*TODO:* Table 1 and Figure 1 (top-1 over K), link to paper and poster.

## Project structure

```text
configs/   YAML configs for training      splits/   frozen identity-disjoint split CSVs
data/      datasets (not versioned)       results/  run logs, summary.csv, figures, tables
src/whoisit/  package (data, models, train, evaluate, gallery)
app/       Gradio demo                    scripts/  helper scripts
tests/     pytest                         docs/     decisions, datasets, meeting notes, AI usage
paper/     mini paper                     examples/ own example photos (with consent)
```

## Workflow

- Every change goes through a branch and a pull request; the **other** person reviews and merges.
- Run `pytest` before every merge.
- No dataset images, no weights (`*.pt`) in Git.

## Privacy

The demo runs locally only (`127.0.0.1`), starts the camera only on click and stores no visitor images.

## Third-party components

MTCNN via [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MIT), ResNet-18 from torchvision (BSD-3-Clause). *More: TODO.*

## AI usage and contributions

See `docs/ai_usage.md` and `docs/contributions.md` (TODO).

## License

Code: [MIT](LICENSE). Datasets and third-party weights are subject to their own licenses.

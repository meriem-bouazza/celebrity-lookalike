# Celebrity Look-Alike

Finds the celebrities whose faces are closest to yours. A neural network turns each face into an embedding (a vector), and a webcam demo shows the three most similar people from a celebrity gallery.

University project for *Advances in Intelligent Systems* by Meriem Bouazza and Melina Tsigka.

## Research question

When only a few photos per person are available, does ArcFace give better face embeddings than plain classification training with cross-entropy?

Our first assumption: ArcFace is better with few photos, and the gap shrinks with more photos. We test this and adjust the hypothesis as we learn more.

## Approach

1. Detect and align the face (MTCNN)
2. Compute an embedding with a ResNet (ImageNet weights, fine-tuned on faces)
3. Compare it with the gallery using cosine similarity
4. Show the top 3 matches

We train the same network on the same data twice and only change the loss (cross-entropy vs. ArcFace). Training and test people are split by person, so the test set contains people the model has not seen.

There is no single correct answer for "looks similar", so we measure whether the system finds other photos of the same person (top-1 and top-5). Whether the look-alikes are plausible is only checked on examples.

## Data

No photos are stored in this repository.

- Training and evaluation: CelebA (aligned images, `identity_CelebA.txt`, `list_attr_celeba.txt`)
- Demo gallery: LFW (people with at least 2 images), and a new gallery with well-known celebrities from the Pins Face Recognition dataset (Kaggle) plus 34 LFW people (see "Run")

Put the CelebA files here:

```
data/celeba/img_align_celeba/
data/celeba/identity_CelebA.txt
data/celeba/list_attr_celeba.txt
```

LFW is downloaded by a script. It lands in `data/sklearn/lfw_home/lfw_funneled/`, and the script writes the list of images to `data/lfw_meta.csv`:

```
python scripts/download_lfw.py
```

Dataset statistics are in `results/`. We may never show CelebA photos outside our team, and no LFW or Pins photos go into Git. The licenses and what they allow are in `docs/decisions.md`, entries "2026-11-03: Dataset licenses" and "2026-11-17: New demo gallery with well-known celebrities".

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11 on Windows and in the GPU container (Linux). The container setup (conda-forge, steps) is described in `docs/decisions.md`, entry "2026-10-30: Python environment".

## Run

Tests (the tests that need real faces are skipped if LFW is missing):

```
python -m pytest
```

Demo gallery with the ready-made face model, then the top 3 look-alikes for one photo:

```
python -m whoisit.data.preprocess --metadata data/lfw_meta.csv --out-dir data/aligned/lfw
python -m whoisit.gallery --aligned-dir data/aligned/lfw --out data/gallery/pretrained.npz
python -m whoisit.gallery --out data/gallery/pretrained.npz --photo me.jpg
python app/app.py
```

With our own trained model (ArcFace, N = 20, seed 42): copy its checkpoint from the container to `results/runs/arcface_n20_seed42_ep20/checkpoints/model.pt`, then:

```
python -m whoisit.gallery --model ours
python app/app.py
```

The demo starts with the ready-made model. If `data/gallery/ours.npz` exists, a button switches to our model. The camera starts only after a click, the uploaded photo is deleted when the next photo comes in (at the latest after 10 minutes), and Gradio's run history in the browser is off. Without a webcam, click one of the example photos: LFW people with only one photo, so they are not in the gallery.

New demo gallery with well-known celebrities: 85 people from the Pins Face Recognition dataset (Kaggle) and 34 well-known people from LFW, without people who are likely CelebA training people (see `docs/decisions.md`). The lists are in `configs/`.

1. Download the Pins dataset from https://www.kaggle.com/datasets/hereisburak/pins-face-recognition and unzip it to `data/pins/`. LFW must be in `data/sklearn/lfw_home/lfw_funneled/` (`python scripts/download_lfw.py`).
2. Pick the photos. This copies 10 photos per person to `data/celeb_raw/` and writes `data/celeb_raw/meta.csv`:

```
python -m whoisit.data.celeb_meta
```

3. Align the photos and build the galleries for both models (in the container, after uploading `data/celeb_raw`):

```
python -m whoisit.data.preprocess --metadata data/celeb_raw/meta.csv --image-root data/celeb_raw --out-dir data/aligned/celeb --device cuda
python -m whoisit.gallery --aligned-dir data/aligned/celeb --out data/gallery/celeb/pretrained.npz --device cuda
python -m whoisit.gallery --model ours --aligned-dir data/aligned/celeb --out data/gallery/celeb/ours.npz --device cuda
```

4. Start the demo with the new gallery (without the options it uses the LFW gallery):

```
python app/app.py --gallery-dir data/gallery/celeb --aligned-dir data/aligned/celeb
```

Add `--device cuda` to use the GPU. The split by person is made with `python -m whoisit.data.splits`.

## Status

Work in progress. Done: dataset statistics, split by person, face detection and alignment, evaluation (top-1 and top-5), the baseline without training, training with cross-entropy and ArcFace (24 runs), the final numbers on the test people, an error analysis with CelebA attributes, and the Gradio demo with the ready-made and our own model. In progress: ArcFace with its own settings, the new celebrity gallery in the app, the poster and the paper. The results and all decisions are in `docs/decisions.md`.

## Limitations

- CelebA and LFW are not representative (mostly celebrities and public figures, skewed in age, gender and skin tone). The Pins gallery has mostly young, western actors.
- In our error analysis, photos of men are found again more often than photos of women (see `docs/decisions.md`, "Error analysis with the attributes").
- The demo runs locally and does not store visitor photos.
- Similarity scores are cosine similarities, not probabilities.

## References

1. Schroff, Kalenichenko, Philbin: FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR 2015.
2. Deng, Guo, Xue, Zafeiriou: ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
3. Kaya, Bilge: Deep Metric Learning: A Survey. Symmetry 11(9), 2019.
4. Zhang, Zhang, Li, Qiao: Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters 23(10), 2016.
5. He, Zhang, Ren, Sun: Deep Residual Learning for Image Recognition. CVPR 2016.

Libraries, pre-trained models and datasets are from others. The training setup, the comparison, the evaluation and the demo were designed by us (see "Use of AI tools").

## Use of AI tools

We used Claude (Anthropic) and Claude Code as helpers, for texts (exposé, plan, README, decision log) and for writing and testing code. We planned every step ourselves, reviewed and ran all code, and made the decisions about the experiments, the settings and the interpretation of the results. This section lists what AI helped with, and we update it when that changes.

## License

MIT for our code (see `LICENSE`). Datasets and pre-trained models have their own terms.
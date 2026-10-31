# Decisions

One entry per decision. Newest at the end.

Template:

```
## YYYY-MM-DD: <decision>
- Decision:
- Why:
- Numbers:
- Alternatives:
- Risks:
- Who:
```

## 2026-10-29: Datasets

- Decision: We use CelebA for training and testing. We use LFW for the demo gallery.
- Why: CelebA has many photos per person, but no names. LFW has names, but most people have only 1 or 2 photos.
- Numbers (from `results/celeba_counts.txt` and `results/lfw_counts.txt`):

  | | CelebA | LFW |
  |---|---|---|
  | photos | 202599 | 13233 |
  | people | 10177 | 5749 |
  | mean photos per person | 19.9 | 2.3 |
  | most photos of one person | 35 | 530 |
  | people with 2 or more photos | 10133 | 1680 |
  | people with 5 or more photos | 9343 | 423 |
  | people with 20 or more photos | 6348 | 62 |

  The LFW mean is 13233 / 5749. It is not in the result file.
- Gallery: LFW people with at least 2 photos, up to 5 photos per person.
- Source: LFW comes from the figshare copy that scikit-learn uses (https://ndownloader.figshare.com/files/5976018). The UMass page did not work on 2026-10-29.
- Licenses: LICENSE_TODO
- Alternatives: VGGFace2 (license unclear). Small celebrity datasets from Kaggle.
- Risks:
  - Some LFW people may also be in CelebA. We cannot check by name, because CelebA has no names.
  - CelebA photos are already cropped. LFW photos are not.
  - Both datasets are not a fair sample of all people.
- Who: Meriem, Melina

## 2026-10-30: Python environment

- Decision: We use Python 3.11 and fixed package versions in `requirements.txt` (for example torch 2.2.2 and numpy 1.26.4).
- Why: The face detector (MTCNN from facenet-pytorch) only works with these older versions. The container has Python 3.13, which does not work with torch 2.2.2. So we use our own environment.
- Numbers: `scripts/check_env.py` runs on the GPU and finishes without errors.
- How to set it up in the container:

      conda create -n lookalike -c conda-forge --override-channels python=3.11 -y
      conda activate lookalike
      conda install -c conda-forge --override-channels pip -y
      python -m pip install -r requirements.txt
      python scripts/check_env.py

  Use conda-forge only (the normal channel asks to accept the Anaconda terms). Run `conda activate lookalike` in every new terminal.
- Alternatives: Newer packages with a different face detector.
- Risks:
  - The versions are old.
  - A new package can change numpy and break things. Check the numpy version after every install.
- Who: Meriem, Melina

## 2026-10-30: Where we work

- Decision: We write code on the laptop and push from there. The container only runs the code. The datasets are uploaded by hand and never go into Git.
- Why: The container has no Git login and runs out after some time. Changes made there cannot be pushed.
- Numbers: The GPU is an NVIDIA RTX 4070 Ti SUPER with 16 GB, shared by both of us. The container shows 96 CPU cores and 54 GB RAM, with no limit set. 
- Alternatives: Colab or Kaggle
- Risks:
  - The container runs out. Copy results to a laptop before that.
  - We share one GPU. Only one training at a time.
- Who: Meriem, Melina

## 2026-10-31: Experiment setup

- Decision: We train the same network two times, once with cross-entropy and once with ArcFace. Then we compare them with 2, 5, 10 and 20 photos per person.
- Why: We want to know if ArcFace helps when we only have a few photos per person.
- Numbers:
  - Network: ResNet-18 from ImageNet, photos 112x112, 512 numbers per face.
  - People (first guess): We only use CelebA people with 20 or more photos (6348 people). 2000 for training, 500 for validation, 1000 for testing. Nobody is in two groups.
  - Training photos: 2, 5, 10 or 20 per person. The smaller set is always part of the bigger set.
  - Seeds (first guess): 3 for each setting. That makes 2 x 4 x 3 = 24 runs.
  - Split: always the same, with seed 42.
  - How we measure: Every test photo is the question once. All other test photos are the gallery. We use cosine similarity. Top 1 is right if the most similar photo shows the same person. Top 5 is right if one of the 5 most similar photos does. Each test person has 10 photos.
  - Baseline: The ImageNet network without training on faces. We have to be better than this.
- Alternatives: ResNet-50 (slower, maybe later). More people or more seeds (more time).
- Risks:
  - We do not know yet how long one run takes. We will do a small test run first (time per epoch, better than the baseline?) and change the numbers if we have to. A change gets a new entry here.
  - The test people are celebrities. The result may be different for other people.
  - With only 3 seeds the numbers can jump around. We will show how much.
- Who: Meriem, Melina
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
- Licenses: see "2026-11-03: Dataset licenses" below.
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

## 2026-11-03: Dataset licenses

- Decision: We never show CelebA photos outside our team, only numbers. Photos in the demo, on the poster and in the paper come from LFW or from ourselves (webcam, with our own consent).
- Why: The CelebA agreement does not allow us to publish any part of the dataset. LFW has no license text, but it is a public benchmark and showing its photos in papers is common.
- Key sentences, CelebA (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, section "Agreement", read on 2026-11-03):
  > The CelebA dataset is available for non-commercial research purposes only.
  >
  > All images of the CelebA dataset are obtained from the Internet which are not property of MMLAB, The Chinese University of Hong Kong.
  >
  > You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
  >
  > You agree not to further copy, publish or distribute any portion of the CelebA dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.
  >
  > The MMLAB reserves the right to terminate your access to the CelebA dataset at any time.
- Key sentences, LFW (http://vis-www.cs.umass.edu/lfw/, section "Disclaimer", read on 2026-11-03 from the web.archive.org copy, because the page did not load):
  > Labeled Faces in the Wild is a public benchmark for face verification, also known as pair matching. No matter what the performance of an algorithm on LFW, it should not be used to conclude that an algorithm is suitable for any commercial purpose.
  >
  > LFW was published to help the research community make advances in face verification, not to provide a thorough vetting of commercial algorithms before deployment.

  The page has no license and no terms of use. The photos are news photos from the web, so the copyright stays with the photographers.
- What this means for us:
  - Training and tests on CelebA: allowed (non-commercial research).
  - The data on the shared container: allowed (copies for internal use at one site).
  - CelebA photos in the demo, on the poster or in the paper: not allowed ("publish"). Models and numbers are fine. Whether "derived data" also covers our trained models only matters if we want to share them, so we do not upload weights.
  - LFW photos in the demo gallery, on the poster and in the paper: OK for a university project. We cite LFW and only show a few photos.
  - The demo runs only locally (see `app/app.py`), so we do not publish the gallery.
- Alternatives: Ask MMLAB for permission to show CelebA photos (slow, not needed).
- Risks:
  - LFW has no clear license. If the university or a publisher asks, we replace LFW photos with our own photos.
  - A ready-made face model for the fallback (for example facenet-pytorch, trained on VGGFace2) has its own license. Check it before we use it.
- Who: Melina

## 2026-11-03: Face detection and alignment

- Decision: We align all photos with MTCNN and `detect_and_align()` (112x112, 5-point template), also the CelebA photos that are already cropped. Training, tests, gallery and demo then all look the same.
- Why: The CelebA crops (`img_align_celeba`) are 178x218 with hair and background around the face. Our template is a tight crop around eyes, nose and mouth. If we train on the CelebA crops and the demo uses MTCNN crops, the model sees different faces in the demo than in training.
- Numbers (`python -m whoisit.data.preprocess`, min confidence 0.9):

  | | photos | faces found | rate |
  |---|---|---|---|
  | CelebA (first 100 lines of `identity_CelebA.txt`, GPU) | 100 | 100 | 100% |
  | Webcam (Melina, light, angle, glasses varied, CPU) | 30 | 30 | 100% |
  | LFW (all, earlier run) | 13233 | 13232 | 99.99% |

  Speed on the GPU: 43 photos per second. All CelebA photos (202599) take about 80 minutes, the about 55000 photos we use take about 20 minutes.
  The webcam photos stay on the laptop and are not in the repository.
- Alternatives: Use the CelebA crops as they are and only resize them (faster, but different from the demo).
- Risks:
  - Faces seen from the side are found, but the alignment is tilted. In the demo we ask people to look into the camera.
  - If MTCNN misses a photo of a val or test person, that person has fewer than 10 photos. `failed.csv` shows these photos; we check it after the full run.
- Who: Melina

## 2026-11-05: Aligned CelebA photos

- Decision: We use CelebA photos that we aligned with MTCNN, like in the demo. Minimum confidence is 0.9.
- Why: Training photos and demo photos should look the same (see the entry from 2026-11-03).
- Numbers:
  - We aligned 61990 photos: up to 25 for each training person, 10 for each validation person and 10 for each test person.
  - 171 photos had no face (99.72 % found). We do not use them.
  - Training people with fewer than 20 aligned photos are left out for every N. That leaves 1983 of the 2000 people. All values of N use the same people.
- Alternatives: Use the CelebA crops as they are. This is simpler, but different from the demo.
- Risks:
  - The 171 photos without a face may be the hard ones (side view, hat). Our data is maybe a bit easier than real life.
  - Some validation and test people may have fewer than 10 photos now. We have not checked this yet.
- Who: Meriem

## 2026-11-05: First training run

- Decision: We train with cross-entropy first. A small test run works. Settings (first guess): ResNet-18 from ImageNet, AdamW, learning rate 0.0003, batch size 128. The only change to the photos is flipping them left to right.
- Why: We want to know that the code works before we start the 24 runs.
- Numbers:
  - Test run with N = 5 (1983 people, 9915 photos), 3 epochs: the loss goes from 7.41 to 5.06. One epoch takes about 12 seconds.
  - The GPU used about 1.7 GB of 16 GB (batch size 128).
  - In the container we have to use `--workers 0`. The shared memory is only 63 MB, so more workers crash.
- Risks: 3 epochs are not enough, the loss still falls fast. We do not know yet how many epochs we need. The pilot run will show this.
- Who: Meriem
## 2026-11-05: One LFW folder

- Decision: LFW lives in one folder everywhere: `data/sklearn/lfw_home/lfw_funneled/`, downloaded with `python scripts/download_lfw.py`. The old folder `data/lfw/lfw/` from the README is no longer used.
- Why: The README and `count_lfw.py` used `data/lfw/lfw/` (the plain LFW download), but the tests, the preprocessing and the gallery used the funneled version from scikit-learn. In the container the LFW tests were skipped, because they looked in the other folder.
- Numbers: Both versions have the same 5749 people and 13233 photos (`results/lfw_counts.txt` is the same). The gallery (top-1 95.92 %) was built from the funneled version.
- Alternatives: Keep `data/lfw/lfw/` and point the tests there with `WHOISIT_LFW_DIR`. That still works if the container cannot download.
- Risks: The funneled photos are already roughly aligned by the LFW authors. MTCNN aligns them again, so this should not matter.
- Who: Melina

## 2026-11-05: Pilot run (time and memory)

- Decision: We keep the numbers from the experiment setup (1983 people, 3 seeds, 24 runs). We use batch size 128 for all runs.
- Why: One epoch takes about the same time for every batch size, so a bigger batch does not help. 128 is in the middle.
- Numbers:
  - Test with N = 20 (1983 people, 39660 photos), 3 epochs for each batch size. Peak GPU memory is the value from PyTorch, not from nvidia-smi.
  - Batch size 64: about 45 s per epoch, peak 0.6 GB, loss 1.76 after 3 epochs.
  - Batch size 128: about 42 s per epoch, peak 0.9 GB, loss 2.01 after 3 epochs.
  - Batch size 256: about 42 s per epoch, peak 1.6 GB, loss 2.70 after 3 epochs.
  - The time grows with the number of photos. Compared to N = 20, N = 10 needs about half, N = 5 a quarter and N = 2 a tenth. All 24 runs together need about 8 minutes for one epoch (45 s x 11.1). With 20 epochs that is about 2.8 hours, with 30 epochs about 4.2 hours.
  - The GPU is idle most of the time, because we load the photos without workers (the shared memory in the container is too small). Loading the photos is the slow part.
- Alternatives: Keep all photos in memory to be faster (about 1.5 GB for N = 20). We do not need this yet.
- Risks:
  - We have not decided how many epochs we need. For this we need the validation numbers.
  - The GPU is shared. If somebody else uses it, our runs are slower.
- Who: Meriem
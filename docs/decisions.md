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
  - The GPU used about 1.7 GB in nvidia-smi (the peak value from PyTorch is 0.9 GB). Batch size is 128.
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

## 2026-11-05: One command per run

- Decision: One command starts one run, for example `python -m whoisit.train --loss ce --n 5 --seed 42 --epochs 20`. The result is saved in `results/runs/<loss>_n<N>_seed<S>_ep<E>/result.json`. After every epoch we measure Top 1 and Top 5 on the validation people. The last epoch counts, not the best one.
- Why: We need 24 runs and want the same numbers for every run.
- Numbers:
  - Validation: we use only photos that were aligned. 500 people and 4988 photos are left (of 5000). Top 1 and Top 5 are computed with the same code as the baseline.
  - Seeds (first guess): 42, 43 and 44. The seed only changes the training. The photos we pick stay the same.
  - Test run with N = 5, 3 epochs: Top 1 goes from 16 % to 27 %, Top 5 from 31 % to 48 %. The baseline without training had 12 % and 23 % (on the test people, not the validation people).
  - Peak GPU memory is 0.9 GB. Training takes about 10 seconds per epoch.
- Alternatives: Use the best epoch. This looks a bit better, but we would pick it with the validation people.
- Risks:
  - Top 1 still goes up after 3 epochs. We do not know yet how many epochs we need.
  - The seeds only change the training, not the photos. So the spread over the 3 seeds does not show how much the choice of photos matters.
  - A crashed run starts again from the beginning.
- Who: Meriem

## 2026-11-05: Add the ArcFace loss

- Decision: ArcFace is chosen with `--loss arcface`. Everything else is the same as for cross-entropy. First guess for the settings: margin 0.5 and scale 64 (the usual values from the ArcFace paper). We do not tune them now. They are saved in `result.json`.
- Why: We want to compare ArcFace with cross-entropy, so only the loss may be different. The same seed now gives the same photo order and the same flips for both losses. Only the loss and its head are different.
- Numbers:
  - The start loss is about 40 for ArcFace and about 7.4 for cross-entropy. The two numbers cannot be compared, so we compare Top 1 and Top 5.
  - Test run with N = 5, seed 42, 3 epochs on the validation people:
    - Cross-entropy: Top 1 25.9 %, Top 5 45.6 %.
    - ArcFace: Top 1 22.5 %, Top 5 40.4 %.
  - Because of the new seed handling, the cross-entropy test numbers changed a little (Top 1 from 26.9 % to 25.9 %).
  - ArcFace also runs with N = 2 (Top 1 13.5 %) and N = 20 (Top 1 59.8 %, about 40 s per epoch). The loss goes down in all cases. Peak GPU memory stays at 0.9 GB.
- Risks:
  - After 3 epochs ArcFace is behind cross-entropy at N = 5, but it still gains about 8 points per epoch, while cross-entropy slows down. Three epochs are too short to say which loss is better.
  - Margin and scale are not tuned. A weak ArcFace result could also come from these settings.
  - We tested only one seed.
- Who: Meriem

## 2026-11-05: Number of epochs

- Decision: Every run trains for 20 epochs, for both losses. The last epoch counts.
- Why: In the test runs the validation numbers stop improving before epoch 20. Cross-entropy is flat after about epoch 11, ArcFace with N = 2 after about epoch 15. We use the same number for both losses and all N, so the comparison is fair.
- Numbers:
  - Cross-entropy, N = 5, seed 42, 15 epochs: Top 1 is 32.5 % after epoch 5, 35.8 % after epoch 10 and 36.2 % after epoch 15. After epoch 11 it changes by less than 0.2 points. The training loss is 0.02 at the end.
  - Cross-entropy, N = 2, seed 42, 30 epochs: Top 1 is about 19 % from epoch 8 on and does not go up until epoch 30 (19.2 %).
  - ArcFace, N = 5, seed 42, 15 epochs: Top 1 is 30.0 % after epoch 5, 34.4 % after epoch 10 and 35.0 % after epoch 15. It goes up and down by about 1 point between epochs. The training loss is still 1.2 at epoch 15.
  - ArcFace, N = 2, seed 42, 30 epochs: Top 1 is about 21 % from epoch 15 on (20.9 % after epoch 20, 21.2 % after epoch 30). The training loss is 0.04 at epoch 20.
  - Time: about 10 s per epoch for N = 5. The 12 cross-entropy runs with 20 epochs need about 80 minutes of training (per run: N = 20 about 14 min, N = 10 about 7 min, N = 5 about 3.5 min, N = 2 about 1.5 min; 3 seeds each) plus the validation.
- Alternatives: 30 epochs. The runs with N = 2 show no gain after epoch 20, and it would take 1.5 times longer. Using the best epoch, but we decided against this before.
- Risks:
  - We tested long runs only with one seed and only for N = 2 and N = 5. N = 10 and N = 20 have more steps per epoch, so 20 epochs should be enough there too.
  - ArcFace with N = 5 may still improve a bit after epoch 15, because its loss was still going down. We will see this in the curves of the real runs.
  - ArcFace with N = 5 goes up and down by about 1 point between epochs. This is about as big as the difference between the two losses at N = 5 (1.2 points) and half of the difference at N = 2 (2.0 points). Because the last epoch counts, the 3 seeds will show how much this matters.
- Who: Meriem

## 2026-11-06: Learning rate schedule and cross-entropy runs

- Decision: The learning rate now goes down from 0.0003 to 0 over the 20 epochs (cosine schedule), for both losses. We ran the 12 cross-entropy runs again with it. The results are in `results/runs/ce_n<N>_seed<S>_ep20/result.json`.
- Why: In the first 12 runs without a schedule, all 3 seeds with N = 20 had a short drop around epochs 13 to 15. The loss jumped from about 0.02 to 0.36 to 0.49 and Top 1 fell by 5 to 8 points. Such a drop at epoch 20 would make the result too low, and the last epoch counts. With the schedule the end of the training is calm.
- Numbers (validation people, last epoch, mean of 3 seeds, spread in brackets):
  - N = 2: Top 1 19.2 % (19.0 to 19.3), Top 5 35.1 %
  - N = 5: Top 1 34.8 % (33.9 to 35.6), Top 5 56.0 %
  - N = 10: Top 1 55.7 % (55.2 to 56.1), Top 5 75.1 %
  - N = 20: Top 1 70.1 % (69.3 to 70.8), Top 5 85.3 %
  - With N = 20 the biggest drop between two epochs is now 0.4 points. Top 1 at epoch 20 differs from epoch 15 by at most 0.4 points, and within these epochs it moves by at most 0.6 points, so 20 epochs are enough.
  - Without the schedule the means were almost the same (19.9 %, 34.9 %, 55.7 %, 70.0 % Top 1). So the schedule makes the training stable, but does not change the level.
  - Time: about 2 hours for all 12 runs. The validation takes about 7 s after every epoch, so about 2.5 min per run, no matter how big N is.
- Alternatives: Keep the old runs and only write down the risk. This would have been faster, but the drop was bigger than the difference between the losses we want to measure.
- Risks:
  - The GPU does not give exactly the same numbers for the same seed. Two runs with seed 42 and N = 2 differed by about 1 point. So the same seed means the same photos and the same order, but not the same result.
  - The seeds only change the training, not the photos we pick.
  - The old runs without the schedule are only kept in the container, not in the repo.
- Who: Meriem

## 2026-11-14: ArcFace runs

- Decision: We ran the 12 ArcFace runs with the same settings as the cross-entropy runs (20 epochs, cosine schedule, seeds 42, 43 and 44, margin 0.5, scale 64). The results are in `results/runs/arcface_n<N>_seed<S>_ep20/result.json`, the table in `results/tables/summary.csv`. The checkpoints stay in the container.
- Why: Same settings for both losses, so only the loss is different (see "Add the ArcFace loss").
- Numbers (validation people, last epoch, mean of 3 seeds, spread in brackets):
  - N = 2: Top 1 18.7 % (18.5 to 18.9), Top 5 33.2 %. Cross-entropy: 19.2 %.
  - N = 5: Top 1 32.5 % (31.5 to 33.1), Top 5 51.1 %. Cross-entropy: 34.8 %.
  - N = 10: Top 1 54.1 % (53.0 to 55.2), Top 5 72.8 %. Cross-entropy: 55.7 %.
  - N = 20: Top 1 70.7 % (70.5 to 70.8), Top 5 84.9 %. Cross-entropy: 70.1 %.
  - So ArcFace is a bit worse than cross-entropy for N = 2, 5 and 10, and about the same for N = 20. Only for N = 5 the spreads of the two losses do not overlap.
  - ArcFace reaches its best epoch early (epoch 8 to 17) and then goes down a bit. Best epoch minus last epoch, mean of 3 seeds: 0.6 points (N = 2), 0.8 (N = 5), 1.4 (N = 10), 2.3 (N = 20). For cross-entropy this is at most 0.5 points. Between epochs 15 and 20, Top 1 moves by at most 0.8 points, so the end of the training is calm.
  - The training loss at epoch 20 is 0.16 to 0.52 for ArcFace (cross-entropy: about 0.004 for N = 20).
  - Time: like cross-entropy, about 2 hours for all 12 runs (training per run: N = 2 about 1.5 min, N = 5 about 3.5 min, N = 10 about 7 min, N = 20 about 15 min, plus about 2.5 min validation).
- Alternatives: Use the best epoch for ArcFace. This would make ArcFace better for N = 20 (72.9 %), but we would pick the epoch with the validation people, and we decided that the last epoch counts.
- Risks:
  - Margin and scale are not tuned. The drop after the best epoch could mean that ArcFace overfits with these settings.
  - These are validation numbers. The final numbers come from the test people (#46).
  - The first comparison with the hypothesis comes in #51.
- Who: Melina

## 2026-11-16: Final numbers on the test people

- Decision: The final numbers come from the test people (#46). We evaluated each of the 24 runs once, with the model from the last epoch. After this we do not change the training anymore (no new epochs, schedule, margin or scale).
- Why: We used the validation people to pick the epochs and the schedule, so their numbers are a bit too good. The test people were never used for any choice.
- Numbers (test people, mean of 3 seeds, spread of Top 1 in brackets):

  | N | CE Top 1 | ArcFace Top 1 | CE Top 5 | ArcFace Top 5 |
  |---|---|---|---|---|
  | 2 | 16.0 % (15.7 to 16.2) | 16.6 % (16.4 to 16.8) | 28.1 % | 28.2 % |
  | 5 | 30.0 % (28.7 to 31.3) | 28.3 % (27.2 to 29.1) | 47.8 % | 43.8 % |
  | 10 | 48.5 % (47.8 to 49.8) | 49.1 % (48.7 to 49.4) | 68.4 % | 66.7 % |
  | 20 | 63.8 % (63.1 to 64.6) | 64.4 % (64.1 to 64.7) | 80.2 % | 79.8 % |

  - Same test photos as the baseline: 1000 people, 9964 photos (`data/aligned/celeba_test`). The baseline without training has Top 1 12.2 % and Top 5 22.6 %.
  - The numbers are in `results/runs/<run>/test.json` and in `results/tables/summary.csv` (split "test").
- Result: The hypothesis is not supported. ArcFace is not clearly better with few photos, and the gap does not get smaller with more photos. In Top 1 most differences are smaller than the spread over the seeds. Only with N = 2 all ArcFace seeds are above all cross-entropy seeds, but only by about 0.6 points. In Top 5 ArcFace is worse with N = 5 and N = 10.
- Alternatives: Tune margin and scale and test again. We do not do this, because then the test numbers would no longer be clean.
- Risks:
  - Margin and scale were not tuned, so a better ArcFace setting may exist.
  - The test numbers are lower than the validation numbers (for example 63.8 % instead of 70.1 % for cross-entropy with N = 20), because they come from other people.
  - Only 3 seeds, and the GPU does not give exactly the same numbers for the same seed.
- Who: Meriem

## 2026-11-16: Attributes for the error analysis

- Decision: For the error analysis we look at 5 CelebA attributes: Eyeglasses, Wearing_Hat, Blurry, Smiling and Male.
- Why:
  - Eyeglasses and Wearing_Hat hide parts of the face.
  - Blurry stands for bad photo quality. CelebA has no attribute for light or pose, so this is the closest one.
  - Smiling changes the face. It splits the test photos almost in half.
  - Male lets us check if the model works equally well for both groups.
- Numbers (test people, 1000 people and 9964 photos, from `results/tables/test_attributes.csv`, made with `python -m whoisit.attributes`):

  | attribute | photos | percent | people with at least 2 such photos |
  |---|---|---|---|
  | Eyeglasses | 570 | 5.7 % | 93 |
  | Wearing_Hat | 436 | 4.4 % | 83 |
  | Blurry | 425 | 4.3 % | 103 |
  | Smiling | 4934 | 49.5 % | 864 |
  | Male | 3852 | 38.7 % | 391 |

  Every group has more than 80 people with at least 2 photos, so we can compute Top 1 for it.
- Alternatives:
  - Attractive or Heavy_Makeup: these are opinions of the people who labeled CelebA, not facts about the photo.
  - Pale_Skin: the only attribute about skin color, and much too rough for a fair statement. We name this as a limit in the paper instead.
  - Young: 81 % of the photos are labeled young, and the label is a rough guess.
- Risks:
  - The CelebA attributes were set by people and are not always right.
  - The groups with glasses, hats and blurry photos are small (about 5 % of the photos), so their numbers will jump more.
  - Several attributes can come together (for example a hat and glasses). We look at each attribute alone.
- Who: Meriem

## 2026-11-16: Does the hypothesis hold?

- Decision: The hypothesis is not supported. With 2 to 20 photos per person and the same settings for both losses, ArcFace was not better than cross-entropy, and the gap does not get smaller with more photos. This is our first result for the paper and the poster.
- Why (test people, ArcFace minus cross-entropy, from `results/tables/summary.csv`):
  - Top 1: +0.5 (N = 2), -1.7 (N = 5), +0.6 (N = 10), +0.6 (N = 20) points.
  - Top 5: +0.1, -4.0, -1.6, -0.4 points.
  - Only one difference is bigger than the spread over the 3 seeds: Top 5 with N = 5, where cross-entropy is better. All other differences are about as big as the noise. On the validation people some of the small differences point the other way.
- Reasons we can show with our numbers:
  - The settings were chosen with cross-entropy (learning rate, schedule, 20 epochs). ArcFace just got the same settings.
  - ArcFace reaches its best validation Top 1 earlier (with N = 20 around epoch 9, cross-entropy around epoch 19) and loses 0.6 to 2.3 points until epoch 20. Cross-entropy loses only 0.1 to 0.5 points. Because the last epoch counts, this costs ArcFace more.
  - The training loss of ArcFace stays higher (0.16 to 0.49 against 0.004 to 0.12 at epoch 20).
- Reasons we can only guess:
  - Margin 0.5 and scale 64 come from the ArcFace paper, which used far more people and photos. They may be too strict for 1983 people with 2 to 20 photos.
  - With fewer people there are fewer classes to push apart.
- What we do next: The first comparison gave ArcFace the settings we had chosen with cross-entropy. To make the comparison fairer, ArcFace gets its own settings, chosen only on the validation people (new issues). The result above stays our first result and stays in the paper.
- What we write: In the paper we say clearly that the hypothesis was not supported with the same settings for both losses, show the numbers with the spread over the seeds, and name the reasons above, split into shown and guessed. The second result (ArcFace with its own settings) comes as an extra section.
- Risks: Only 3 seeds, and the GPU does not give exactly the same numbers for the same seed (about 1 point). Small differences can be chance.
- Who: Meriem, Melina
## 2026-11-17: Rules for ArcFace with its own settings

- Decision: We write these rules down before any new run, so we cannot change them after we see the numbers.
  - We try 3 new settings: 10 epochs with margin 0.5, 20 epochs with margin 0.3, 10 epochs with margin 0.3. Our old setting (20 epochs, margin 0.5) is the fourth. Scale stays 64, everything else stays the same.
  - A setting with 10 epochs is a new run with the cosine schedule over 10 epochs. We do not use epoch 10 of the old 20-epoch runs.
  - First only seed 42 with N = 2 and N = 20, only on the validation people. For the old setting we use our existing runs with seed 42.
  - We pick the setting with the best mean validation Top 1 of N = 2 and N = 20, at the last epoch (like `summary.csv`). If two are equal, we take the shorter training.
  - Only the chosen setting runs for all N with 3 seeds, and only it goes once to the test people.
  - The first result (same settings for both losses) stays in the paper.
- Why: In the first comparison ArcFace got the settings we had chosen with cross-entropy (see 2026-11-16). ArcFace was best around epoch 9 and then got worse, so fewer epochs or a smaller margin may suit it better. Choosing only on the validation people keeps the test people clean.
- Numbers: 6 new tuning runs (3 settings, N = 2 and N = 20, seed 42), then 12 runs for the chosen setting (N = 2, 5, 10, 20, seeds 42, 43, 44).
- Alternatives:
  - Pick the best epoch of each run (early stopping): this would change the rule we used for cross-entropy.
  - Tune the scale or the learning rate too: more runs than we have time for.
- Risks:
  - Only one seed per setting, and the GPU gives about 1 point difference for the same seed. The chosen setting can win by chance.
  - Cross-entropy did not get the same tuning, so the second comparison favors ArcFace a bit. We say this in the paper.
- Who: Meriem, Melina

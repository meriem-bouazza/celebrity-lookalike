# Decisions

One entry per decision, newest at the end.

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

- Decision: CelebA for training and evaluation, LFW for the demo gallery (names).
- Why: CelebA has many images per identity but no names. LFW has names but too few images per person for training.
- Numbers (from `results/celeba_counts.txt` and `results/lfw_counts.txt`):

  | | CelebA | LFW |
  |---|---|---|
  | images | 202599 | 13233 |
  | identities | 10177 | 5749 |
  | mean images per identity | 19.9 | 2.3 |
  | max images per identity | 35 | 530 |
  | >= 2 images | 10133 | 1680 |
  | >= 5 images | 9343 | 423 |
  | >= 20 images | 6348 | 62 |

  The LFW mean is 13233 / 5749. It is not in the result file.
- Gallery: people with >= 2 LFW images, up to 5 images per person.
- Source note: LFW was downloaded from the figshare mirror used by scikit-learn (https://ndownloader.figshare.com/files/5976018) because the UMass page was not reachable on 2026-10-29.
- Licenses: LICENSE_TODO
- Alternatives:
  - VGGFace2 (license and availability unclear).
  - Smaller celebrity datasets from Kaggle.
- Risks:
  - LFW people may also be in CelebA. We cannot check by name because CelebA has no names.
  - CelebA images are aligned crops, LFW images are not.
  - Both datasets are not representative.
- Who: Meriem, Melina

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
- Demo gallery: LFW (people with at least 2 images)

Put the files here:

```
data/celeba/img_align_celeba/
data/celeba/identity_CelebA.txt
data/celeba/list_attr_celeba.txt
data/lfw/lfw/
```

LFW can be downloaded with:

```
curl.exe -L -C - -o data/lfw/lfw.tgz "https://ndownloader.figshare.com/files/5976018"
tar -xzf data/lfw/lfw.tgz -C data/lfw
```

Dataset statistics are in `results/`. Licenses: TODO, see `docs/decisions.md`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11 on Windows.

## Status

Work in progress. Dataset statistics are done, training and demo are not.

## Limitations

- CelebA and LFW are not representative (mostly celebrities and public figures, skewed in age, gender and skin tone).
- The demo runs locally and does not store visitor photos.
- Similarity scores are cosine similarities, not probabilities.

## References

1. Schroff, Kalenichenko, Philbin: FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR 2015.
2. Deng, Guo, Xue, Zafeiriou: ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
3. Kaya, Bilge: Deep Metric Learning: A Survey. Symmetry 11(9), 2019.
4. Zhang, Zhang, Li, Qiao: Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters 23(10), 2016.
5. He, Zhang, Ren, Sun: Deep Residual Learning for Image Recognition. CVPR 2016.

Libraries, pre-trained models and datasets are from others. The training setup, comparison, evaluation and demo are our own work.

## Use of AI tools

We used Claude (Anthropic) for the exposé text and layout, the project plan, templates for the README and decision log, the dataset counting scripts, and for help with error messages. We reviewed and adapted all of it ourselves. Code and text written from here on is marked in the commit history where AI helped.

## License

MIT for our code (see `LICENSE`). Datasets and pre-trained models have their own terms.

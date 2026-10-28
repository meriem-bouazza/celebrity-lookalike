# Celebrity Look-Alike

Find the celebrities you look most like. A neural network turns every face into an embedding (a vector of numbers), and a webcam demo shows the three most similar celebrities.

> University project for *Advances in Intelligent Systems*, Hochschule Düsseldorf (HSD), WiSe 2026/27.
> Authors: Meriem Bouazza, Melina Tsigka

<!-- TODO: add a screenshot or GIF of the demo here -->

## Research question

Does ArcFace produce better face embeddings than plain classification training (cross-entropy) when only a few photos per person are available?

**Hypothesis (first assumption):** ArcFace performs better with few photos per person, and the difference shrinks as more photos become available. We test this with measurements and refine it during the project.

## Why this project

- Embeddings are a core concept of modern AI (image search, language models). We want to apply and understand them hands-on.
- Face recognition affects everyday life but is controversial (privacy, fairness, biometric data). We discuss these aspects openly.
- The idea is playful and easy to try out. The value is mainly educational and societal.

## How it works

1. **Detect and align** the face (MTCNN)
2. **Compute the embedding** (ResNet, pre-trained on ImageNet, fine-tuned on faces)
3. **Compare** with the celebrity database using cosine similarity
4. **Show the top 3** most similar celebrities with a similarity score

We train the same network on the same data twice and change only the loss: cross-entropy vs. ArcFace. People are split into training and test sets, so we test on persons the model has never seen.

**Evaluation:** There is no single correct answer for "similar". We therefore measure whether the system finds other photos of the same person (top-1 and top-5 retrieval accuracy). Whether the results are plausible look-alikes is checked on examples only. This is a known limitation.

## Requirements

- Python 3.10 or newer (TODO: confirm version)
- A GPU for training (Google Colab or Kaggle is enough), CPU is sufficient for the demo
- A webcam for the live demo (example photos work as a fallback)

## Installation

```bash
git clone https://github.com/<github-user>/celebrity-lookalike.git
cd celebrity-lookalike
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repository does **not** contain any photos or datasets.

| Dataset | Used for | License / terms |
|---|---|---|
| TODO (e.g. VGGFace2, CelebA, LFW) | Training and testing | TODO: check and link |
| TODO | Celebrity database for the demo (names needed) | TODO |

Download instructions: TODO (script in `src/data/` or step by step here).

## Usage

```bash
# Train with cross-entropy
python src/train.py --loss cross_entropy      # TODO: adjust

# Train with ArcFace
python src/train.py --loss arcface            # TODO: adjust

# Evaluate
python src/evaluate.py                        # TODO: adjust

# Start the demo
python app/demo.py                            # TODO: adjust
```

## Results

TODO: add the results table and a link to the paper once available.

## Limitations and ethics

- The demo runs **locally**. Visitor images are not stored, and the camera starts only after a click.
- Celebrity datasets are not representative (age, ethnicity, make-up, photo style). We analyze errors for lighting, pose, glasses and groups where the dataset provides such attributes.
- Similarity scores are cosine similarities, not probabilities.
- This is a prototype for learning, not a tool for identifying people.

## Project structure

```
├── src/         data loading, model, training, evaluation
├── app/         webcam demo (Gradio or Streamlit)
├── notebooks/   Colab notebooks
├── configs/     experiment settings (loss, photos per person, seeds)
├── results/     tables and figures (no photos)
└── docs/        decisions, poster, notes
```

## References

1. Schroff, Kalenichenko, Philbin: *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR 2015.
2. Deng, Guo, Xue, Zafeiriou: *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
3. Kaya, Bilge: *Deep Metric Learning: A Survey.* Symmetry 11(9), 2019.
4. Zhang, Zhang, Li, Qiao: *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks (MTCNN).* IEEE Signal Processing Letters 23(10), 2016.
5. He, Zhang, Ren, Sun: *Deep Residual Learning for Image Recognition (ResNet).* CVPR 2016.
6. [deepface](https://github.com/serengil/deepface) and [insightface](https://github.com/deepinsight/insightface) (reference implementations)

**What is ours and what is not:** Libraries, pre-trained models and datasets come from others and are credited above. The training setup, the comparison, the evaluation and the demo are our own work.

## Use of AI tools

TODO: state honestly how you used AI tools (for example Claude by Anthropic for text and design in the exposé and for plans). Keep this consistent with the paper.

## License

MIT for our code, see [LICENSE](LICENSE). Datasets and pre-trained models have their own terms, see the Data section.

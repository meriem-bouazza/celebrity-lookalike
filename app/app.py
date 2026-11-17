"""Gradio demo: webcam or upload -> align -> embedding -> top-3 look-alikes from the LFW gallery.

Runs only locally (127.0.0.1, share=False, analytics off) and stores no visitor images:
the uploaded photo is deleted when the next photo comes in, run history in the browser is off,
and Gradio's file cache deletes files older than 5 minutes every 5 minutes.
The camera starts only after a click. The demo starts with the ready-made model; a button switches to ours.

Usage (the gallery must exist, see README "Run"):
    python app/app.py
    python app/app.py --device cuda
"Our model" is offered only when data/gallery/ours.npz exists.
Then open http://127.0.0.1:7860 in the browser.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # must be set before gradio is imported

import argparse
import csv
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np

from gradio.utils import get_upload_folder

from whoisit.data.align import detect_and_align, load_image
from whoisit import gallery as galleries
from whoisit.gallery import top_people

# The one setting that switches the model: name -> (load, embed, gallery built with the same model).
MODELS = {name: (load, embed, galleries.gallery_path(name)) for name, (load, embed) in galleries.MODELS.items()}
MODEL_LABELS = {"pretrained": "Ready-made model", "ours": "Our model (ArcFace)"}
UPLOAD_DIR = Path(get_upload_folder())  # where Gradio stores visitor photos; only files here are deleted
# The aligned face of the photo on screen. Its file is kept, because Gradio sends the same path again
# when the model is switched; it is deleted as soon as the next photo comes in.
last_face: dict[str, np.ndarray | None] = {}


def aligned_face(photo_path: str, device: str) -> np.ndarray | None:
    if photo_path not in last_face:
        for old_path in last_face:
            if Path(old_path).resolve().is_relative_to(UPLOAD_DIR.resolve()):
                Path(old_path).unlink(missing_ok=True)
        last_face.clear()
        last_face[photo_path] = detect_and_align(load_image(photo_path), device=device)
    return last_face[photo_path]


def make_match(model_name: str = "pretrained", device: str = "cpu",
               aligned_dir: Path = Path("data/aligned/lfw"), k: int = 3):
    """Load the model and gallery once and return the function the demo calls for every photo."""
    load, embed, gallery_path = MODELS[model_name]
    model = load(device)
    gallery = np.load(gallery_path)

    def match(photo_path: str | None) -> list[tuple[np.ndarray, str]]:
        if photo_path is None:
            raise gr.Error("Please take or upload a photo first.")
        face = aligned_face(photo_path, device)
        if face is None:
            raise gr.Error("No face found. Look straight into the camera and try again.")
        query = embed(model, face[None], device)[0]
        return [(load_image(aligned_dir / gallery["image_paths"][i]), f"{name.replace('_', ' ')} ({score:.2f})")
                for name, score, i in top_people(query, gallery["embeddings"], gallery["identities"], k)]

    return match


def example_photos(metadata: Path = Path("data/lfw_meta.csv"), k: int = 4) -> list[str]:
    """LFW photos of the first k people with only one photo: they are not in the gallery (2+ photos), never CelebA."""
    with open(metadata, newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row["identity"] for row in rows)
    return [row["image_path"] for row in rows if counts[row["identity"]] == 1 and Path(row["image_path"]).exists()][:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Celebrity look-alike demo (local only).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    matchers = {name: make_match(name, args.device) for name in MODEL_LABELS if MODELS[name][2].exists()}
    demo = gr.Interface(
        lambda photo, model_name: matchers[model_name](photo),
        [gr.Image(sources=["upload", "webcam"], type="filepath", label="Your Photo"),
         gr.Radio([(MODEL_LABELS[name], name) for name in matchers], value="pretrained", label="Model")],
        gr.Gallery(label="Your Top 3 Look-Alikes", columns=3, object_fit="contain"),
        title="Celebrity Look-Alike",
        description="The number is the cosine similarity (1 = same face), not a probability. "
                    "Your photo is not saved. It is deleted when the next photo comes in, at the latest after 10 minutes.",
        submit_btn="Find My Look-Alikes",
        examples=[[photo, "pretrained"] for photo in example_photos()],
        flagging_mode="never",
        delete_cache=(300, 300),
    )
    demo.launch(server_name="127.0.0.1", share=False, run_history=False)  # Gradio would keep every photo in the browser


if __name__ == "__main__":
    main()

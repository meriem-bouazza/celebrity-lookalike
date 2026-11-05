"""Gradio demo: webcam or upload -> align -> embedding -> top-3 look-alikes from the LFW gallery.

Runs only locally (127.0.0.1, share=False, analytics off) and stores no visitor images:
the photo is only kept in memory, and Gradio's file cache is cleared every 5 minutes.

Usage (the gallery must exist, see README "Run"):
    python app/app.py
    python app/app.py --model pretrained --device cuda
Then open http://127.0.0.1:7860 in the browser.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # must be set before gradio is imported

import argparse
from pathlib import Path

import gradio as gr
import numpy as np

from whoisit.data.align import detect_and_align, load_image
from whoisit.gallery import embed_pretrained, load_pretrained, top_people

# The one setting that switches the model: name -> (load, embed, gallery built with the same model).
# shortcut: only the ready-made model so far; add ours here once training saves a model (#20, #23).
MODELS = {
    "pretrained": (load_pretrained, embed_pretrained, Path("data/gallery/pretrained.npz")),
}


def make_match(model_name: str = "pretrained", device: str = "cpu",
               aligned_dir: Path = Path("data/aligned/lfw"), k: int = 3):
    """Load the model and gallery once and return the function the demo calls for every photo."""
    load, embed, gallery_path = MODELS[model_name]
    model = load(device)
    gallery = np.load(gallery_path)

    def match(image: np.ndarray | None) -> list[tuple[np.ndarray, str]]:
        if image is None:
            raise gr.Error("Please take or upload a photo first.")
        face = detect_and_align(image, device=device)
        if face is None:
            raise gr.Error("No face found. Look straight into the camera and try again.")
        query = embed(model, face[None], device)[0]
        return [(load_image(aligned_dir / gallery["image_paths"][i]), f"{name.replace('_', ' ')} ({score:.2f})")
                for name, score, i in top_people(query, gallery["embeddings"], gallery["identities"], k)]

    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Celebrity look-alike demo (local only).")
    parser.add_argument("--model", choices=sorted(MODELS), default="pretrained")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    demo = gr.Interface(
        make_match(args.model, args.device),
        gr.Image(sources=["webcam", "upload"], type="numpy", label="Your Photo"),
        gr.Gallery(label="Your Top 3 Look-Alikes", columns=3, object_fit="contain"),
        title="Celebrity Look-Alike",
        description="The number is the cosine similarity (1 = same face), not a probability. "
                    "Your photo is not saved. Temporary files are deleted after 5 minutes.",
        submit_btn="Find My Look-Alikes",
        flagging_mode="never",
        delete_cache=(300, 300),
    )
    demo.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()

import numpy as np
import torch
from PIL import Image

from whoisit.data.images import make_transform
from whoisit.gallery import (embed_ours, embed_pretrained, load_ours, load_pretrained, select_gallery,
                             top_people)


def rows(counts):
    # counts = {identity: number of aligned photos}
    return [{"identity": name, "image_path": f"images/{name}/{name}_{i:04d}.jpg"}
            for name, n in counts.items() for i in range(n, 0, -1)]


def test_select_gallery_min_and_max_photos():
    chosen = select_gallery(rows({"one": 1, "two": 2, "seven": 7}))
    per_person = {}
    for r in chosen:
        per_person.setdefault(r["identity"], []).append(r["image_path"])
    assert set(per_person) == {"two", "seven"}  # "one" has only 1 photo
    assert len(per_person["two"]) == 2
    assert per_person["seven"] == [f"images/seven/seven_{i:04d}.jpg" for i in range(1, 6)]  # first 5 by name


def test_embeddings_shape_and_norm():
    # untrained weights: no download, only checks resizing, scaling and output shape
    model = load_pretrained(pretrained=None)
    faces = np.random.default_rng(0).integers(0, 256, (3, 112, 112, 3), dtype=np.uint8)
    embeddings = embed_pretrained(model, faces)
    assert embeddings.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_top_people_scores_each_person_by_best_photo():
    embeddings = np.eye(4)
    identities = np.array(["a", "a", "b", "c"])
    query = np.array([0.6, 0.0, 0.8, 0.0])  # b's photo is the closest, then a's first photo
    assert top_people(query, embeddings, identities, k=2) == [("b", 0.8, 2), ("a", 0.6, 0)]


def test_embed_ours_scales_faces_like_training():
    # the demo must feed our model the same numbers as the validation during training
    torch.manual_seed(0)
    model = load_ours(checkpoint=None)
    faces = np.random.default_rng(0).integers(0, 256, (2, 112, 112, 3), dtype=np.uint8)
    with torch.no_grad():
        expected = model(torch.stack([make_transform(train=False)(Image.fromarray(f)) for f in faces]))
    expected = torch.nn.functional.normalize(expected, dim=1).numpy()
    np.testing.assert_allclose(embed_ours(model, faces), expected, atol=1e-5)

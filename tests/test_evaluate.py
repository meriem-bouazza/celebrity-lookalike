import numpy as np
import pytest

from whoisit.evaluate import top_k, top_k_accuracy


def angles(degrees):
    # one 2-d unit vector per angle, so cosine similarity is easy to see by hand
    rad = np.deg2rad(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1)


def test_top_k_excludes_the_image_itself():
    embeddings = np.eye(3, dtype=np.float32)
    embeddings[1] = embeddings[0]  # image 1 is a copy of image 0
    neighbours = top_k(embeddings, k=1)
    assert neighbours[0, 0] == 1 and neighbours[1, 0] == 0


def test_hand_made_example():
    # A: 0 and 10 degrees (close). B: 90 and 180. C: 100 and 270.
    # Nearest other photo: A0->A1 yes, A1->A0 yes, B90->C100 no, B180->C100 no,
    # C100->B90 no, C270->A0 or B180 (both 90 degrees away) no. So Top 1 = 2/6.
    embeddings = angles([0, 10, 90, 180, 100, 270])
    labels = ["A", "A", "B", "B", "C", "C"]
    result = top_k_accuracy(embeddings, labels)
    assert result[1] == pytest.approx(2 / 6)
    assert result[5] == 1.0  # with 6 photos, the 5 others are all photos


def test_perfect_embeddings_give_one():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(20), 5)
    embeddings = np.eye(20)[labels] + 0.01 * rng.normal(size=(100, 20))
    assert top_k_accuracy(embeddings, labels) == {1: 1.0, 5: 1.0}


def test_random_embeddings_are_near_chance():
    # 100 people x 10 photos: a random other photo is the same person with p = 9/999.
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(100), 10)
    result = top_k_accuracy(rng.normal(size=(1000, 128)), labels)
    assert result[1] < 0.05
    assert result[1] <= result[5] < 0.15


def test_scale_of_embeddings_does_not_matter():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(30, 8))
    labels = np.repeat(np.arange(10), 3)
    scaled = embeddings * rng.uniform(0.1, 10, size=(30, 1))
    assert top_k_accuracy(embeddings, labels) == top_k_accuracy(scaled, labels)


def test_wrong_input_is_rejected():
    with pytest.raises(ValueError):
        top_k_accuracy(np.eye(4), [0, 0, 1])  # 4 rows, 3 labels
    with pytest.raises(ValueError):
        top_k_accuracy(np.eye(3), [0, 0, 1])  # person 1 has only one photo

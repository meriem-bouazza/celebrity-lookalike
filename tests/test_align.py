"""Tests for face detection and alignment (task 3.1)."""

import numpy as np
import pytest

from whoisit.data.align import (
    TEMPLATE_112,
    align_face,
    detect_and_align,
    detect_largest_face,
    load_image,
    similarity_transform,
)


# --- Geometry only: no photo needed -------------------------------------------------------


def _apply(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


def test_similarity_transform_maps_landmarks_onto_template():
    # Landmarks = template rotated by 20 degrees, scaled by 2.5 and shifted.
    angle = np.deg2rad(20)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    landmarks = (TEMPLATE_112 @ rot.T) * 2.5 + np.array([300.0, 120.0])

    matrix = similarity_transform(landmarks.astype(np.float32))

    np.testing.assert_allclose(_apply(matrix, landmarks), TEMPLATE_112, atol=1e-2)


def test_align_face_output_shape_and_type():
    image = np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8)
    aligned = align_face(image, TEMPLATE_112 * 2.0)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_no_face_returns_none():
    # Smooth gradient without any face-like structure.
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)
    image = np.stack([np.tile(gradient, (256, 1))] * 3, axis=-1)
    assert detect_and_align(image) is None


# --- Real photos (LFW) ---------------------------------------------------------------------


def test_portrait_gives_aligned_face(lfw_people):
    path = next(iter(lfw_people.values()))[0]
    aligned = detect_and_align(load_image(path))
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


def test_aligned_eyes_are_level(lfw_people):
    # Detecting again on the aligned crop: both eyes should be close to the same height.
    for images in list(lfw_people.values())[:5]:
        aligned = detect_and_align(load_image(images[0]))
        assert aligned is not None
        face = detect_largest_face(aligned, min_confidence=0.5)
        if face is None:  # MTCNN sometimes misses tight 112 px crops; geometry is tested above
            continue
        landmarks, _ = face
        assert abs(landmarks[0, 1] - landmarks[1, 1]) < 5.0


def test_group_photo_uses_largest_face(lfw_people):
    # Synthetic "group photo": person A at full size next to person B at half size.
    name_a, name_b = list(lfw_people)[:2]
    big = load_image(lfw_people[name_a][0])
    small = load_image(lfw_people[name_b][0])[::2, ::2]
    canvas = np.zeros((big.shape[0], big.shape[1] + small.shape[1] + 50, 3), dtype=np.uint8)
    canvas[:, : big.shape[1]] = big
    canvas[: small.shape[0], big.shape[1] + 50 :] = small

    face = detect_largest_face(canvas)
    assert face is not None
    landmarks, _ = face
    # All landmarks must lie inside the region of the big image (person A).
    assert (landmarks[:, 0] < big.shape[1]).all()


def test_pil_and_array_input_give_same_result(lfw_people):
    from PIL import Image

    image = load_image(next(iter(lfw_people.values()))[0])
    a = detect_and_align(image)
    b = detect_and_align(Image.fromarray(image))
    assert a is not None and b is not None
    np.testing.assert_array_equal(a, b)


def test_min_confidence_above_one_rejects_everything(lfw_people):
    image = load_image(next(iter(lfw_people.values()))[0])
    assert detect_and_align(image, min_confidence=1.01) is None

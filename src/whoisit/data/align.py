"""Face detection (MTCNN) and 5-point similarity alignment to 112 x 112 pixels.

The same ``detect_and_align()`` is used for training, gallery and demo images so that
all inputs look alike (plan, section 0.5/6).

Pipeline:
    1. MTCNN (facenet-pytorch) finds face boxes, confidences and 5 landmarks.
    2. The largest face above ``min_confidence`` is kept.
    3. A similarity transform (rotation, uniform scale, translation) maps its 5 landmarks
       onto a fixed template in a 112 x 112 image.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

OUTPUT_SIZE = 112

# 5-point template for 112 x 112 crops: left eye, right eye, nose tip, left and right mouth
# corner (left/right as seen in the image). Taken from insightface, `arcface_dst` in
# python-package/insightface/utils/face_align.py (https://github.com/deepinsight/insightface).
# MTCNN returns its landmarks in the same order.
TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# One MTCNN instance per device, created on first use (loading takes ~1 s).
_detectors: dict[str, MTCNN] = {}


def _get_detector(device: str) -> MTCNN:
    """Return a cached MTCNN detector for ``device`` ("cpu" or "cuda")."""
    if device not in _detectors:
        _detectors[device] = MTCNN(keep_all=True, device=device)
    return _detectors[device]


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array of shape (H, W, 3).

    The EXIF orientation is applied, so portrait phone photos are not rotated by 90 degrees.
    """
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def _to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert a PIL image or an RGB array to a contiguous RGB uint8 array (H, W, 3)."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image of shape (H, W, 3), got {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def detect_largest_face(
    image: np.ndarray | Image.Image, min_confidence: float = 0.9, device: str = "cpu"
) -> tuple[np.ndarray, float] | None:
    """Detect faces with MTCNN and return the largest one above ``min_confidence``.

    Args:
        image: RGB image as array (H, W, 3) uint8 or PIL image.
        min_confidence: Minimum MTCNN face probability (0..1).
        device: "cpu" or "cuda".

    Returns:
        ``(landmarks, confidence)`` with landmarks of shape (5, 2) in pixel coordinates
        (x, y), or ``None`` if no face reaches ``min_confidence``.
    """
    rgb = _to_rgb_array(image)
    with torch.no_grad():
        boxes, probs, points = _get_detector(device).detect(Image.fromarray(rgb), landmarks=True)
    if boxes is None:
        return None

    keep = probs >= min_confidence
    if not keep.any():
        return None
    boxes, probs, points = boxes[keep], probs[keep], points[keep]

    # "Largest" = largest box area; in group photos this is usually the main person.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    return points[i].astype(np.float32), float(probs[i])


def similarity_transform(landmarks: np.ndarray) -> np.ndarray:
    """Estimate the 2 x 3 similarity transform that maps ``landmarks`` onto ``TEMPLATE_112``.

    Args:
        landmarks: Array (5, 2) with the detected points in the order of ``TEMPLATE_112``.

    Returns:
        Affine matrix (2, 3) for ``cv2.warpAffine``.
    """
    # estimateAffinePartial2D fits rotation + uniform scale + translation (4 degrees of freedom),
    # i.e. no shearing, so the face is not distorted. LMEDS is robust if one landmark is off.
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), TEMPLATE_112, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("could not estimate a transform from the given landmarks")
    return matrix


def align_face(image: np.ndarray | Image.Image, landmarks: np.ndarray) -> np.ndarray:
    """Warp ``image`` so that ``landmarks`` land on the template; returns (112, 112, 3) uint8."""
    rgb = _to_rgb_array(image)
    matrix = similarity_transform(landmarks)
    return cv2.warpAffine(rgb, matrix, (OUTPUT_SIZE, OUTPUT_SIZE), borderValue=(0, 0, 0))


def detect_and_align(
    image: np.ndarray | Image.Image, min_confidence: float = 0.9, device: str = "cpu"
) -> np.ndarray | None:
    """Detect the largest face and return it aligned to 112 x 112 pixels.

    Args:
        image: RGB image as array (H, W, 3) uint8 or PIL image.
        min_confidence: Minimum MTCNN face probability (0..1).
        device: "cpu" or "cuda".

    Returns:
        Aligned RGB face of shape (112, 112, 3) uint8, or ``None`` if no face was found.
    """
    face = detect_largest_face(image, min_confidence=min_confidence, device=device)
    if face is None:
        return None
    landmarks, _ = face
    return align_face(image, landmarks)

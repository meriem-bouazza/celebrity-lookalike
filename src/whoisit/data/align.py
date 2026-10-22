"""Face detection (MTCNN) and 5-point similarity alignment to 112 x 112 pixels.

The same ``detect_and_align()`` is used for training, gallery and webcam images so that
all inputs look alike (plan, section 0.5/6).

TODO (task 3.1, Meriem): implement ``detect_and_align(image, min_confidence) -> np.ndarray | None``.
"""

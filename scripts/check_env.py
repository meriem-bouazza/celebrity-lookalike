"""Checks that the environment from requirements.txt works (plan, task 2.4).

Usage: python scripts/check_env.py
Run it on both laptops and in the GPU environment (Colab/Kaggle).
"""

import platform
import sys
from importlib.metadata import version


def main() -> int:
    import cv2
    import gradio
    import numpy
    import torch
    import torchvision
    from facenet_pytorch import MTCNN
    from torchvision.models import resnet18

    print(f"Python       {platform.python_version()} ({sys.executable})")
    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"facenet      {version('facenet-pytorch')}")
    print(f"gradio       {gradio.__version__}")
    print(f"numpy        {numpy.__version__}")
    print(f"opencv       {cv2.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "-"
    print(f"device       {device} ({gpu})")

    # MTCNN ships its weights with the package, so this also works offline.
    MTCNN(device=device)

    # Forward pass with untrained weights: checks shapes without downloading ImageNet weights.
    net = resnet18(weights=None)
    net.fc = torch.nn.Identity()
    out = net.to(device).eval()(torch.randn(2, 3, 112, 112, device=device))
    assert tuple(out.shape) == (2, 512), out.shape
    print("forward pass (2, 3, 112, 112) -> (2, 512) ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

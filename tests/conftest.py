"""Shared pytest fixtures.

Tests that need real face photos use LFW (downloaded for development only, not versioned):
    python scripts/download_lfw.py
They are skipped if LFW is missing. Override the location with the environment variable
WHOISIT_LFW_DIR (folder that contains one subfolder per person).
"""

import os
from pathlib import Path

import pytest

DEFAULT_LFW_DIR = Path(__file__).resolve().parents[1] / "data" / "sklearn" / "lfw_home" / "lfw_funneled"


@pytest.fixture(scope="session")
def lfw_dir() -> Path:
    """Folder with one subfolder of JPEGs per person, or skip the test."""
    path = Path(os.environ.get("WHOISIT_LFW_DIR", DEFAULT_LFW_DIR))
    if not path.is_dir():
        pytest.skip(f"LFW not found at {path} (run scripts/download_lfw.py)")
    return path


@pytest.fixture(scope="session")
def lfw_people(lfw_dir) -> dict[str, list[Path]]:
    """Persons with at least 5 images, as {name: sorted image paths} (sorted by name)."""
    people = {}
    for person in sorted(p for p in lfw_dir.iterdir() if p.is_dir()):
        images = sorted(person.glob("*.jpg"))
        if len(images) >= 5:
            people[person.name] = images
    return people

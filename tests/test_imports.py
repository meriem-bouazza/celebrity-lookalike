"""Smoke test: the package and all modules import (keeps pytest green while modules are stubs)."""

import importlib

import pytest

MODULES = [
    "whoisit",
    "whoisit.utils",
    "whoisit.evaluate",
    "whoisit.gallery",
    "whoisit.train",
    "whoisit.data.align",
    "whoisit.data.preprocess",
    "whoisit.data.splits",
    "whoisit.data.dataset",
    "whoisit.models.backbone",
    "whoisit.models.heads",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_set_seed_is_reproducible():
    import torch

    from whoisit.utils import set_seed

    set_seed(0)
    a = torch.rand(3)
    set_seed(0)
    b = torch.rand(3)
    assert torch.equal(a, b)

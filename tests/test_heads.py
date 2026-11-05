"""Tests for the training heads, with hand-made embeddings."""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from whoisit.models.heads import ArcFaceHead, SoftmaxHead, make_head  # noqa: E402


def cosines(head, embeddings):
    return F.normalize(embeddings) @ F.normalize(head.weights).T


def test_output_shape():
    head = ArcFaceHead(num_people=10)
    logits = head(torch.randn(4, 512), torch.tensor([0, 3, 9, 3]))
    assert logits.shape == (4, 10)


def test_batch_where_all_labels_are_the_same():
    head = ArcFaceHead(num_people=5)
    logits = head(torch.randn(4, 512), torch.full((4,), 2))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_margin_0_gives_scale_times_cosine():
    head = ArcFaceHead(num_people=6, margin=0.0, scale=10.0)
    embeddings = torch.randn(4, 512)
    logits = head(embeddings, torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(logits, 10.0 * cosines(head, embeddings))


def test_margin_only_makes_the_correct_person_smaller():
    head = ArcFaceHead(num_people=6, margin=0.5, scale=10.0)
    embeddings = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 2, 3])
    plain = 10.0 * cosines(head, embeddings)

    logits = head(embeddings, labels)

    rows = torch.arange(4)
    assert (logits[rows, labels] < plain[rows, labels]).all()
    others = torch.ones_like(plain, dtype=torch.bool)
    others[rows, labels] = False
    assert torch.equal(logits[others], plain[others])


def angle_logits(degrees, margin):
    # two numbers per face and one person at 0 degrees, so we know the angle theta
    head = ArcFaceHead(num_people=1, embedding_size=2, margin=margin, scale=1.0)
    head.weights.data = torch.tensor([[1.0, 0.0]])
    theta = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32))
    embeddings = torch.stack([theta.cos(), theta.sin()], dim=1)
    return theta, head(embeddings, torch.zeros(len(theta), dtype=torch.long)).squeeze(1)


def test_logit_is_cos_of_theta_plus_margin():
    theta, logits = angle_logits([10.0, 45.0, 90.0, 120.0], margin=0.5)
    assert torch.allclose(logits, torch.cos(theta + 0.5), atol=1e-4)


def test_fallback_above_pi_keeps_going_down():
    # without the fallback the value goes up again when theta + margin is larger than pi
    _, logits = angle_logits(list(range(0, 181, 5)), margin=0.5)
    assert (logits[1:] < logits[:-1]).all()
    assert logits[-1].item() == pytest.approx(-1 - math.sin(math.pi - 0.5) * 0.5, abs=1e-4)


@pytest.mark.parametrize("degrees", [0.0, 180.0])
def test_gradient_is_finite_when_embedding_lies_on_the_weight(degrees):
    head = ArcFaceHead(num_people=2, embedding_size=2)
    head.weights.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    theta = math.radians(degrees)
    embeddings = torch.tensor([[math.cos(theta), math.sin(theta)]], requires_grad=True)

    F.cross_entropy(head(embeddings, torch.tensor([0])), torch.tensor([0])).backward()

    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(head.weights.grad).all()


def test_make_head():
    head = make_head("arcface", 10)
    assert isinstance(head, ArcFaceHead)
    assert head.weights.shape == (10, 512)
    assert head.scale == 64.0
    assert head.cos_m == pytest.approx(math.cos(0.5))

    head = make_head("arcface", 10, margin=0.3, scale=30.0)
    assert head.scale == 30.0
    assert head.cos_m == pytest.approx(math.cos(0.3))

    assert isinstance(make_head("ce", 10), SoftmaxHead)

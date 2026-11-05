import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from whoisit.data.images import SIZE, FaceDataset  # noqa: E402
from whoisit.models.backbone import EMBEDDING_SIZE, Backbone  # noqa: E402
from whoisit.models.heads import SoftmaxHead  # noqa: E402


def test_backbone_output_shape():
    model = Backbone(pretrained=False)
    out = model(torch.zeros(2, 3, SIZE, SIZE))
    assert out.shape == (2, EMBEDDING_SIZE)


def test_embedding_is_linear_and_batchnorm():
    linear, norm = Backbone(pretrained=False).embedding
    assert isinstance(linear, torch.nn.Linear)
    assert (linear.in_features, linear.out_features) == (512, 512)
    assert isinstance(norm, torch.nn.BatchNorm1d)


def test_head_output_shape():
    head = SoftmaxHead(num_people=10)
    logits = head(torch.zeros(4, EMBEDDING_SIZE), torch.zeros(4, dtype=torch.long))
    assert logits.shape == (4, 10)


def test_one_training_step_changes_the_weights():
    model = Backbone(pretrained=False)
    head = SoftmaxHead(num_people=3)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    before = model.embedding[0].weight.clone()

    images = torch.rand(4, 3, SIZE, SIZE)
    labels = torch.tensor([0, 1, 2, 0])
    loss = torch.nn.functional.cross_entropy(head(model(images), labels), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.embedding[0].weight)


def test_dataset(tmp_path):
    rows = []
    for person in (7, 9):
        (tmp_path / str(person)).mkdir()
        for i in range(3):
            name = f"{person}_{i}.jpg"
            Image.new("RGB", (112, 112)).save(tmp_path / str(person) / name)
            rows.append((name, person))
    photos = pd.DataFrame(rows, columns=["image", "person"])
    data = FaceDataset(photos, image_dir=tmp_path)
    image, label = data[0]
    assert len(data) == 6
    assert data.num_people == 2
    assert image.shape == (3, SIZE, SIZE)
    assert label in (0, 1)

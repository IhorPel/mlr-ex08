import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from leaf_pipeline import ROOT, LeafDataset, LeafNet, LeafNetBN, evaluate, image_transform


def test_dataset_agrees_with_imagefolder():
    ours = LeafDataset(ROOT / "data/leaves/train", image_transform())
    reference = datasets.ImageFolder(ROOT / "data/leaves/train", transform=image_transform())
    assert ours.classes == reference.classes == ["apple", "cherry", "chestnut", "maple"]
    assert len(ours) == len(reference) == 700
    for i in [0, 174, 175, 349, 699, -1]:
        x, y = ours[i]
        expected, label = reference[i]
        assert torch.equal(x, expected) and y == label


@pytest.mark.parametrize("model", [LeafNet(), LeafNetBN(), LeafNet(dropout=.3), LeafNet(global_pool=True)])
def test_network_shape_and_backward(model):
    torch.set_num_threads(4)
    model.train()
    result = model(torch.randn(5, 3, 96, 96))
    assert result.shape == (5, 4)
    torch.nn.functional.cross_entropy(result, torch.tensor([0, 1, 2, 3, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_normalization_and_channels():
    x = image_transform()(Image.new("RGB", (96, 96), (255, 0, 0)))
    assert x.shape == (3, 96, 96)
    np.testing.assert_allclose(x[:, 0, 0].numpy(), [(1-.485)/.229, -.456/.224, -.406/.225], atol=1e-6)


def test_batchnorm_evaluation_does_not_update_statistics():
    model = LeafNetBN()
    model.eval()
    before = model.bn1.running_mean.clone()
    with torch.no_grad():
        model(torch.randn(2, 3, 96, 96))
    assert torch.equal(before, model.bn1.running_mean)


def test_evaluation_is_sample_weighted_for_uneven_last_batch():
    dataset = torch.utils.data.TensorDataset(torch.randn(7, 3, 96, 96), torch.arange(7) % 4)
    model = LeafNet().eval()
    a = evaluate(model, DataLoader(dataset, batch_size=3), "cpu")
    b = evaluate(model, DataLoader(dataset, batch_size=7), "cpu")
    assert a["accuracy"] == b["accuracy"]
    assert abs(a["loss"] - b["loss"]) < 1e-6

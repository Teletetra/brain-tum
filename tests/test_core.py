import numpy as np
import torch

from brain_tumor_seg.data.windows import make_25d
from brain_tumor_seg.evaluation.metrics import dice_score, iou_score
from brain_tumor_seg.losses.dice import dice_loss
from brain_tumor_seg.models.edge_branch import EdgeBranch
from brain_tumor_seg.models.h_csaf import HCSAFFusion


def test_make_25d_shape_and_boundaries():
    volume = np.zeros((16, 16, 5), dtype=np.float32)
    sample = make_25d(volume, 0)
    assert sample.shape == (3, 16, 16)
    sample = make_25d(volume, 4)
    assert sample.shape == (3, 16, 16)


def test_metrics_perfect_prediction():
    target = np.array([[0, 1], [1, 0]])
    assert dice_score(target, target, 1) == 1.0
    assert iou_score(target, target, 1) == 1.0


def test_dice_loss_is_zero_for_perfect_logits():
    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]], [[-8.0, 8.0], [8.0, -8.0]]]])
    loss = dice_loss(logits, target)
    assert float(loss) < 1e-3


def test_edge_branch_shape():
    model = EdgeBranch(16)
    x = torch.randn(2, 16, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)


def test_h_csaf_shape():
    model = HCSAFFusion(32, num_heads=4)
    cnn = torch.randn(2, 32, 16, 16)
    vit = torch.randn(2, 32, 8, 8)
    y = model(cnn, vit)
    assert y.shape == cnn.shape

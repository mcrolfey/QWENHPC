"""Rare-class training helpers: per-class validation loss and softened balancing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers import TrainingArguments  # noqa: E402

from src.train import BalancedTrainer, answer_token_losses, balanced_mean  # noqa: E402


def _labels():
    # Row 0 scores tokens 3 and 4 (positions 2, 3 after the shift); row 1 scores token 2 only.
    return torch.tensor([[-100, -100, -100, 5, 6], [-100, -100, 7, -100, -100]])


def test_answer_token_losses_match_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = _labels()
    per, mean = answer_token_losses(logits, labels)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(mean, ref)                                   # same as the model's own loss
    row0 = F.cross_entropy(logits[0, 2:4], labels[0, 3:5])
    row1 = F.cross_entropy(logits[1, 1:2], labels[1, 2:3])
    assert torch.allclose(per, torch.stack([row0, row1]))


def test_balanced_mean_weights_classes_equally():
    # 3 common examples with loss 1, 1 rare example with loss 5: per-example mean 2, per-class mean 3.
    assert balanced_mean([1, 1, 1, 5], ["a", "a", "a", "b"]) == 3.0


def test_sample_weights_power():
    from src.dataset import AsbestosVLMDataset
    ds = AsbestosVLMDataset.__new__(AsbestosVLMDataset)
    ds.class_keys = lambda: [("type", 0)] * 4 + [("type", 1)]
    full, sqrt, none = ds.sample_weights(1.0), ds.sample_weights(0.5), ds.sample_weights(0.0)
    assert full[:4].sum() == pytest.approx(full[4])                   # classes drawn equally often
    assert sqrt[:4].sum() / sqrt[4] == pytest.approx(2.0)             # common class 2x, not 4x
    assert torch.all(none == 1)


class _Model(torch.nn.Module):
    """Predicts a fixed distribution; examples of class 1 get a much worse loss."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **_):
        logits = torch.zeros(*input_ids.shape, 4) + self.w
        logits[..., 0] = 3.0
        return SimpleNamespace(logits=logits)


def test_trainer_reports_balanced_loss(tmp_path):
    # Answer token is 0 (easy) for class 0 and 3 (hard) for class 1; 3 easy examples, 1 hard.
    data = [{"t": 0}, {"t": 0}, {"t": 0}, {"t": 3}]

    def collate(items):
        ids = torch.zeros(len(items), 3, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        labels[:, 2] = torch.tensor([it["t"] for it in items])
        return {"input_ids": ids, "labels": labels}

    args = TrainingArguments(output_dir=str(tmp_path), per_device_eval_batch_size=3, report_to=[],
                             use_cpu=True, remove_unused_columns=False, label_names=["labels"])
    groups = [("type", 0)] * 3 + [("type", 1)]
    trainer = BalancedTrainer(model=_Model(), args=args, eval_dataset=data, data_collator=collate,
                              eval_groups=groups)
    m = trainer.evaluate()
    easy = F.cross_entropy(torch.tensor([[3.0, 0, 0, 0]]), torch.tensor([0])).item()
    hard = F.cross_entropy(torch.tensor([[3.0, 0, 0, 0]]), torch.tensor([3])).item()
    assert m["eval_balanced_loss"] == pytest.approx((easy + hard) / 2, rel=1e-5)
    assert m["eval_loss"] == pytest.approx((3 * easy + hard) / 4, rel=1e-3)  # batches of 3 and 1

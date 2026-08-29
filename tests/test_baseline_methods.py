from __future__ import annotations

import torch

from src.baseline.clear import apply_clear_scores, clear_confidence
from src.baseline.common import IGNORE_INDEX
from src.baseline.ibft import VariationalBottleneck, compute_ibft_loss
from src.baseline.llm_code_cleaning import apply_llm_cleaning_rewrites
from src.baseline.token_cleaning import apply_token_cleaning_from_scores, token_quality_scores
from src.baseline.xtf import apply_xtf_from_scores


def _sample(uid: str = "s1"):
    return {"uid": uid, "input_ids": [1, 2, 3, 4, 5], "labels": [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]}


def test_token_cleaning_masks_low_score_labels():
    rows, report = apply_token_cleaning_from_scores(
        [_sample()],
        {"s1": {"uid": "s1", "scores": [0, 0, 0.1, 0.9, 0.8]}},
        keep_ratio=2 / 3,
    )
    assert rows[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]
    assert report["kept_tokens"] == 2


def test_token_quality_score_is_base_minus_reference():
    assert token_quality_scores([3.0, 1.0], [1.0, 2.0]) == [2.0, -1.0]


def test_xtf_masks_high_pcp_tokens():
    rows, report = apply_xtf_from_scores(
        [_sample()],
        {"s1": {"uid": "s1", "pcp_probs": [0, 0, 0.99, 0.2, 0.1]}},
        pcp_threshold=0.95,
    )
    assert rows[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]
    assert report["masked_tokens"] == 1


def test_clear_filters_and_replaces():
    rows, report = apply_clear_scores(
        [_sample("keep"), _sample("drop"), _sample("replace")],
        {
            "keep": {"uid": "keep", "confidence": 0.9},
            "drop": {"uid": "drop", "confidence": 0.1},
            "replace": {
                "uid": "replace",
                "confidence": 0.2,
                "candidate_response": "better()",
                "candidate_confidence": 0.8,
                "candidate_better_score": 0.95,
            },
        },
        gamma=0.5,
        eta=0.8,
    )
    assert len(rows) == 2
    assert report["filtered_samples"] == 1
    assert report["replaced_samples"] == 1


def test_clear_confidence_is_clamped():
    assert clear_confidence(2.0, -1.0) == 0.5


def test_llm_cleaning_rewrite_updates_response_fields():
    sample = {"uid": "s1", "response": "old", "messages": [{"role": "assistant", "content": "old"}]}
    rows, report = apply_llm_cleaning_rewrites([sample], {"s1": {"uid": "s1", "cleaned_response": "```go\nnew\n```"}})
    assert rows[0]["response"] == "new"
    assert rows[0]["messages"][0]["content"] == "new"
    assert report["replaced_samples"] == 1


def test_ibft_loss_shapes():
    hidden = (torch.randn(1, 5, 8),)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
    lm_head = torch.nn.Linear(8, 16)
    bottleneck = VariationalBottleneck(8, z_dim=4)
    out = compute_ibft_loss(hidden_states=hidden, labels=labels, lm_head=lm_head, bottleneck=bottleneck)
    assert out.num_tokens == 3
    assert out.loss.ndim == 0


class _FakeScoreOutput:
    def __init__(self, logits, attentions=None, hidden_states=None):
        self.logits = logits
        self.attentions = attentions
        self.hidden_states = hidden_states


class _FakeScoreModel(torch.nn.Module):
    def __init__(self, vocab_size=16, bias=0.0):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.bias = bias

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        logits[..., 3] = self.bias
        logits[..., 4] = self.bias / 2
        logits[..., 5] = self.bias / 3
        attentions = None
        hidden_states = None
        if output_attentions:
            base = torch.ones(batch, 1, length, length, device=input_ids.device)
            base = torch.tril(base)
            attentions = (base / base.sum(dim=-1, keepdim=True).clamp_min(1),)
        if output_hidden_states:
            hidden = torch.nn.functional.one_hot(input_ids.clamp_max(7), num_classes=8).float()
            hidden_states = (hidden,)
        return _FakeScoreOutput(logits, attentions=attentions, hidden_states=hidden_states)


def test_token_cleaning_score_generation_with_fake_models():
    from src.baseline.token_cleaning import compute_token_cleaning_score_rows

    rows = compute_token_cleaning_score_rows([_sample()], _FakeScoreModel(bias=0.0), _FakeScoreModel(bias=2.0))
    assert rows[0]["uid"] == "s1"
    assert len(rows[0]["scores"]) == 5
    assert rows[0]["scores"][2] is not None
    assert rows[0]["scores"][0] is None


def test_xtf_score_generation_with_fake_model():
    from src.baseline.xtf import compute_xtf_score_rows

    rows = compute_xtf_score_rows([_sample()], _FakeScoreModel(bias=1.0))
    assert rows[0]["uid"] == "s1"
    assert len(rows[0]["ri_scores"]) == 5
    assert rows[0]["ri_scores"][2] is not None
    assert rows[0]["pcp_probs"][2] is not None
    assert rows[0]["tr_scores"][2] is not None

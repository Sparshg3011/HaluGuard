"""
test_hccs_training.py - Tests for embedding truncation and ranking helpers.

Run with:
    pytest tests/test_hccs_training.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from haluguard.hccs import (
    HCCSScorer,
    batch_embed,
    build_pair_features,
    embed_code,
    get_pair_feature_dim,
)
from haluguard.training import (
    build_ranking_dataloaders,
    build_ranking_examples,
    compute_retrieval_metrics,
    create_data_splits,
    select_training_chunk_indices,
    split_ranking_examples,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.truncation_side = "right"
        self.recorded_sides = []
        self.recorded_lengths = []
        # ``batch_embed`` uses padding=True; ``_ensure_tokenizer_pad_token`` maps pad to eos if missing.
        self.eos_token = "<eos>"
        self.eos_token_id = 2

    def __call__(
        self,
        text_or_texts,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: bool,
    ):
        self.recorded_sides.append(self.truncation_side)
        self.recorded_lengths.append(max_length)
        batch_size = 1 if isinstance(text_or_texts, str) else len(text_or_texts)
        return {
            "input_ids": torch.ones(batch_size, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0] for _ in range(batch_size)],
                dtype=torch.long,
            ),
        }


class _DummyEncoder:
    def __call__(self, **inputs):
        batch_size, seq_len = inputs["input_ids"].shape
        last_hidden_state = torch.arange(
            batch_size * seq_len * 3,
            dtype=torch.float32,
        ).view(batch_size, seq_len, 3)
        return SimpleNamespace(last_hidden_state=last_hidden_state)


class TestEmbeddingTruncation:
    def test_embed_code_keeps_tail_by_default(self) -> None:
        tokenizer = _DummyTokenizer()
        model = _DummyEncoder()

        result = embed_code("print('hello')", tokenizer, model, device="cpu")

        assert result.shape == (3,)
        assert tokenizer.recorded_sides == ["left"]
        assert tokenizer.truncation_side == "right"

    def test_batch_embed_keeps_start_for_context_chunks(self) -> None:
        tokenizer = _DummyTokenizer()
        model = _DummyEncoder()

        result = batch_embed(
            ["def foo(): pass", "def bar(): pass"],
            tokenizer,
            model,
            device="cpu",
        )

        assert result.shape == (2, 3)
        assert tokenizer.recorded_sides == ["right"]
        assert tokenizer.truncation_side == "right"

    def test_embed_code_respects_max_length_and_normalize(self) -> None:
        tokenizer = _DummyTokenizer()
        model = _DummyEncoder()

        result = embed_code(
            "print('hello')",
            tokenizer,
            model,
            device="cpu",
            max_length=128,
            normalize=True,
        )

        assert tokenizer.recorded_lengths == [128]
        assert torch.isclose(torch.tensor(result).norm(), torch.tensor(1.0))

    def test_batch_embed_supports_mean_pooling(self) -> None:
        tokenizer = _DummyTokenizer()
        model = _DummyEncoder()

        result = batch_embed(
            ["def foo(): pass"],
            tokenizer,
            model,
            device="cpu",
            pooling="mean",
        )

        expected = torch.tensor([[3.0, 4.0, 5.0]])
        assert torch.allclose(torch.tensor(result), expected)

    def test_batch_embed_sanitizes_empty_snippets(self) -> None:
        """RepoBench may include empty/whitespace snippets; must not yield zero-length batches."""
        tokenizer = _DummyTokenizer()
        model = _DummyEncoder()

        result = batch_embed(["", "   ", "def x(): pass"], tokenizer, model, device="cpu")

        assert result.shape == (3, 3)


class TestScorer:
    def test_pair_features_include_interaction_terms(self) -> None:
        query = torch.tensor([[1.0, 2.0]])
        chunk = torch.tensor([[3.0, 5.0]])

        features = build_pair_features(query, chunk)

        expected = torch.tensor([[1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 3.0, 10.0]])
        assert torch.equal(features, expected)
        assert get_pair_feature_dim(2) == 8

    def test_forward_returns_probabilities(self) -> None:
        scorer = HCCSScorer(input_dim=6, hidden_dim=4, dropout=0.0)
        x = torch.randn(5, 6)

        scores = scorer(x)
        logits = scorer.forward_logits(x)

        assert scores.shape == (5, 1)
        assert logits.shape == (5, 1)
        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)


class TestRankingHelpers:
    def test_build_examples_and_split_by_example(self) -> None:
        query_embs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        chunk_embs = [
            torch.tensor([
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
        ]
        gold_indices = [0, 0]

        examples = build_ranking_examples(query_embs, chunk_embs, gold_indices)
        train_examples, val_examples = split_ranking_examples(
            examples,
            train_ratio=0.5,
            seed=7,
        )

        train_ids = {example.example_id for example in train_examples}
        val_ids = {example.example_id for example in val_examples}

        assert len(examples) == 2
        assert train_ids.isdisjoint(val_ids)

    def test_hard_negative_sampling_keeps_gold_first(self) -> None:
        query_embs = torch.tensor([[1.0, 0.0, 0.0]])
        chunk_embs = [torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ])]
        gold_indices = [0]

        example = build_ranking_examples(query_embs, chunk_embs, gold_indices)[0]
        selected = select_training_chunk_indices(
            example,
            num_hard_negatives=1,
            num_random_negatives=0,
        )

        assert selected == [0, 1]

    def test_all_chunk_training_loader_keeps_full_candidate_list(self) -> None:
        query_embs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        chunk_embs = [
            torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]),
            torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
        ]
        gold_indices = [1, 0]

        examples = build_ranking_examples(query_embs, chunk_embs, gold_indices)
        train_loader, _ = build_ranking_dataloaders(
            train_examples=examples,
            val_examples=examples,
            train_batch_size=2,
            eval_batch_size=2,
            use_all_chunks_for_training=True,
        )

        batch = next(iter(train_loader))

        assert batch["chunk_embs"].shape[1] == 3
        assert sorted(batch["targets"].tolist()) == [0, 1]
        assert [True, True, True] in batch["mask"].tolist()
        assert [True, True, False] in batch["mask"].tolist()

    def test_retrieval_metrics_match_rankings(self) -> None:
        logits = torch.tensor([
            [4.0, 1.0, 0.5],
            [0.1, 0.2, 3.0],
        ])
        targets = torch.tensor([0, 2])
        mask = torch.tensor([
            [True, True, True],
            [True, True, True],
        ])

        metrics = compute_retrieval_metrics(logits, targets, mask)

        assert metrics["acc@1"] == 1.0
        assert metrics["acc@3"] == 1.0
        assert metrics["acc@5"] == 1.0
        assert metrics["recall@1"] == 1.0
        assert metrics["recall@3"] == 1.0
        assert metrics["recall@5"] == 1.0
        assert metrics["mrr"] == 1.0


class TestCreateDataSplits:
    def test_no_overlap_and_full_coverage(self) -> None:
        splits = create_data_splits(100, train_ratio=0.8, val_ratio=0.1, seed=42)
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == 100
        assert len(set(all_ids)) == 100  # no overlap
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_deterministic_with_same_seed(self) -> None:
        a = create_data_splits(500, seed=7)
        b = create_data_splits(500, seed=7)
        assert a == b

    def test_different_seed_gives_different_split(self) -> None:
        a = create_data_splits(500, seed=1)
        b = create_data_splits(500, seed=2)
        assert a["train"] != b["train"]

    def test_indices_are_sorted(self) -> None:
        splits = create_data_splits(200, seed=42)
        for key in ("train", "val", "test"):
            assert splits[key] == sorted(splits[key])

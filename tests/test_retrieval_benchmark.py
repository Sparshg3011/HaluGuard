"""Tests for paper-style retrieval benchmark helpers."""

from __future__ import annotations

from haluguard.retrieval_benchmark import (
    BUCKET_EASY,
    BUCKET_HARD,
    QUERY_VIEW_FULL,
    QUERY_VIEW_LAST3,
    build_query_text,
    build_ranking_result,
    compute_accuracy_metrics,
    get_candidate_bucket,
    rank_indices_from_scores,
    summarise_rankings,
)


class TestQueryViews:
    def test_full_query_view_returns_original_code(self) -> None:
        cropped_code = "a = 1\nb = 2\nreturn a + b\n"

        assert build_query_text(cropped_code, QUERY_VIEW_FULL) == cropped_code

    def test_last3_query_view_keeps_only_last_three_lines(self) -> None:
        cropped_code = "line1\nline2\nline3\nline4\nline5"

        assert build_query_text(cropped_code, QUERY_VIEW_LAST3) == "line3\nline4\nline5"


class TestBuckets:
    def test_bucket_assignment_matches_paper_style(self) -> None:
        assert get_candidate_bucket(4) is None
        assert get_candidate_bucket(5) == BUCKET_EASY
        assert get_candidate_bucket(9) == BUCKET_EASY
        assert get_candidate_bucket(10) == BUCKET_HARD


class TestRankingMetrics:
    def test_rank_indices_from_scores_orders_descending(self) -> None:
        ranked = rank_indices_from_scores([0.1, 0.7, 0.4])

        assert ranked == [1, 2, 0]

    def test_accuracy_metrics_are_monotonic(self) -> None:
        metrics = compute_accuracy_metrics([1, 2, 5, 6], top_ks=[1, 3, 5])

        assert metrics["acc@1"] <= metrics["acc@3"] <= metrics["acc@5"]
        assert metrics["mrr"] > 0.0

    def test_summarise_rankings_reports_easy_and_hard_sections(self) -> None:
        rankings = [
            build_ranking_result(
                method="demo",
                example_id=0,
                query_view=QUERY_VIEW_LAST3,
                candidate_count=5,
                gold_index=1,
                ranked_indices=[1, 0, 2, 3, 4],
            ),
            build_ranking_result(
                method="demo",
                example_id=1,
                query_view=QUERY_VIEW_LAST3,
                candidate_count=10,
                gold_index=4,
                ranked_indices=[0, 4, 1, 2, 3, 5, 6, 7, 8, 9],
            ),
        ]

        table = summarise_rankings(rankings)

        assert len(table) == 1
        row = table[0]
        assert row["easy_count"] == 1
        assert row["hard_count"] == 1
        assert row["easy_acc@1"] == 1.0
        assert row["hard_acc@3"] == 1.0
        assert "easy_mrr" not in row
        assert "hard_mrr" not in row

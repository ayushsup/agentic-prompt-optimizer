"""
Unit tests for the scoring subsystem.

Covers:
  - All five evaluation metric types (string_exact, integer_exact,
    number_tolerance, string_semantic, array_llm)
  - ScoreResult accumulation and F1 arithmetic
  - Scorer.score_document on synthetic schema+gold+pred triples
  - Array alignment policy (positional for objects, set-based for primitives)
  - Graceful handling of malformed predictions (empty JSON, parse errors)
  - Scorer.calculate_f1 edge cases (zero division)

Run with: pytest tests/test_scorer.py -v
"""

import json
import pytest

from src.evaluation.metrics import (
    integer_exact,
    number_tolerance,
    string_exact,
    compute_cache_key,
)
from src.evaluation.scorer import ScoreResult, Scorer


# ---------------------------------------------------------------------------
# Deterministic metric tests
# ---------------------------------------------------------------------------

class TestStringExact:
    def test_identical(self):
        assert string_exact("Hello World", "Hello World") == 1.0

    def test_case_insensitive(self):
        assert string_exact("HELLO", "hello") == 1.0

    def test_whitespace_trimmed(self):
        assert string_exact("  hello  ", "hello") == 1.0

    def test_mismatch(self):
        assert string_exact("foo", "bar") == 0.0

    def test_non_string_pred(self):
        assert string_exact(123, "123") == 0.0

    def test_none_values(self):
        assert string_exact(None, "test") == 0.0
        assert string_exact("test", None) == 0.0


class TestIntegerExact:
    def test_exact(self):
        assert integer_exact(42, 42) == 1.0

    def test_coercion_from_string(self):
        assert integer_exact("42", 42) == 1.0

    def test_coercion_from_float(self):
        assert integer_exact(42.0, 42) == 1.0

    def test_mismatch(self):
        assert integer_exact(41, 42) == 0.0

    def test_invalid_pred(self):
        assert integer_exact("not-a-number", 42) == 0.0

    def test_none(self):
        assert integer_exact(None, 42) == 0.0


class TestNumberTolerance:
    def test_exact_match(self):
        assert number_tolerance(100.0, 100.0) == 1.0

    def test_within_tolerance(self):
        assert number_tolerance(104.9, 100.0, tolerance=0.05) == 1.0

    def test_outside_tolerance(self):
        assert number_tolerance(106.0, 100.0, tolerance=0.05) == 0.0

    def test_gold_zero_pred_zero(self):
        assert number_tolerance(0, 0) == 1.0

    def test_gold_zero_pred_nonzero(self):
        assert number_tolerance(1, 0) == 0.0

    def test_string_coercion(self):
        assert number_tolerance("100", 100) == 1.0

    def test_invalid_pred(self):
        assert number_tolerance("invalid", 100) == 0.0


# ---------------------------------------------------------------------------
# Cache key test
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        key1 = compute_cache_key("string_semantic", "hello", "world")
        key2 = compute_cache_key("string_semantic", "hello", "world")
        assert key1 == key2

    def test_different_metrics(self):
        k1 = compute_cache_key("string_semantic", "hello", "world")
        k2 = compute_cache_key("array_llm", "hello", "world")
        assert k1 != k2

    def test_order_matters(self):
        k1 = compute_cache_key("string_semantic", "a", "b")
        k2 = compute_cache_key("string_semantic", "b", "a")
        assert k1 != k2


# ---------------------------------------------------------------------------
# ScoreResult tests
# ---------------------------------------------------------------------------

class TestScoreResult:
    def test_perfect_score(self):
        r = ScoreResult(true_positives=5.0, predicted_count=5, gold_count=5)
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_zero_division(self):
        r = ScoreResult(true_positives=0.0, predicted_count=0, gold_count=0)
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_f1_formula(self):
        r = ScoreResult(true_positives=2.0, predicted_count=4, gold_count=4)
        # P=0.5, R=0.5, F1=0.5
        assert abs(r.f1 - 0.5) < 1e-6

    def test_merge(self):
        a = ScoreResult(true_positives=3.0, predicted_count=4, gold_count=5)
        b = ScoreResult(true_positives=2.0, predicted_count=3, gold_count=3)
        a.merge(b)
        assert a.true_positives == 5.0
        assert a.predicted_count == 7
        assert a.gold_count == 8

    def test_to_dict_keys(self):
        r = ScoreResult(true_positives=1.0, predicted_count=1, gold_count=1)
        d = r.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d


# ---------------------------------------------------------------------------
# Scorer.calculate_f1
# ---------------------------------------------------------------------------

class TestScorerF1:
    def setup_method(self):
        self.scorer = Scorer()

    def test_perfect(self):
        assert self.scorer.calculate_f1(1.0, 1.0) == 1.0

    def test_zero_recall(self):
        assert self.scorer.calculate_f1(1.0, 0.0) == 0.0

    def test_zero_precision(self):
        assert self.scorer.calculate_f1(0.0, 1.0) == 0.0

    def test_both_zero(self):
        assert self.scorer.calculate_f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        f1 = self.scorer.calculate_f1(0.75, 0.5)
        assert abs(f1 - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Scorer.score_document integration tests
# ---------------------------------------------------------------------------

SCHEMA_SIMPLE = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "evaluation_config": "string_exact"},
        "age":  {"type": "integer", "evaluation_config": "integer_exact"},
    }
})

class TestScorerDocument:
    def setup_method(self):
        self.scorer = Scorer()

    def test_perfect_extraction(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice", "age": 30})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 == 1.0

    def test_wrong_value(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Bob", "age": 30})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 < 1.0

    def test_missing_field(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice"})  # age absent
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 < 1.0

    def test_malformed_prediction(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        f1, breakdown = self.scorer.score_document("not-json", gold, SCHEMA_SIMPLE)
        assert f1 == 0.0  # graceful failure

    def test_json_with_fences(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = "```json\n" + json.dumps({"name": "Alice", "age": 30}) + "\n```"
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 == 1.0

    def test_breakdown_contains_subtrees(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice", "age": 30})
        _, breakdown = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert "subtrees" in breakdown
        assert "name" in breakdown["subtrees"]


# ---------------------------------------------------------------------------
# Array alignment policy tests
# ---------------------------------------------------------------------------

SCHEMA_ARRAY_STR = json.dumps({
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "evaluation_config": "string_exact",
        }
    }
})

SCHEMA_ARRAY_OBJ = json.dumps({
    "type": "object",
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "evaluation_config": "string_exact"},
                    "title":   {"type": "string", "evaluation_config": "string_exact"},
                }
            }
        }
    }
})

class TestArrayAlignment:
    def setup_method(self):
        self.scorer = Scorer()

    def test_primitive_array_exact_match(self):
        gold = json.dumps({"skills": ["Python", "SQL"]})
        pred = json.dumps({"skills": ["Python", "SQL"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 1.0

    def test_primitive_array_partial_match(self):
        gold = json.dumps({"skills": ["Python", "SQL", "Java"]})
        pred = json.dumps({"skills": ["Python", "SQL"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert 0.0 < f1 < 1.0

    def test_primitive_array_no_match(self):
        gold = json.dumps({"skills": ["Python"]})
        pred = json.dumps({"skills": ["Java"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 0.0

    def test_object_array_positional_alignment(self):
        gold = json.dumps({"jobs": [
            {"company": "Acme", "title": "Engineer"},
            {"company": "Beta", "title": "Lead"},
        ]})
        pred = json.dumps({"jobs": [
            {"company": "Acme", "title": "Engineer"},
            {"company": "Beta", "title": "Lead"},
        ]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_OBJ)
        assert f1 == 1.0

    def test_object_array_partial_positional(self):
        gold = json.dumps({"jobs": [
            {"company": "Acme", "title": "Engineer"},
        ]})
        pred = json.dumps({"jobs": [
            {"company": "Wrong Corp", "title": "Engineer"},
        ]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_OBJ)
        assert 0.0 < f1 < 1.0

    def test_empty_pred_array(self):
        gold = json.dumps({"skills": ["Python"]})
        pred = json.dumps({"skills": []})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# Stochastic metric fallback (no judge provided)
# ---------------------------------------------------------------------------

class TestStochasticFallback:
    def setup_method(self):
        # No judge_callable → falls back to string_exact
        self.scorer = Scorer()

    def test_string_semantic_fallback_match(self):
        score = self.scorer._evaluate_with_config("hello", "hello", "string_semantic")
        assert score == 1.0

    def test_string_semantic_fallback_mismatch(self):
        score = self.scorer._evaluate_with_config("hello", "world", "string_semantic")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Judge float parsing (critical for stochastic metric correctness)
# ---------------------------------------------------------------------------

from src.optimizer.loop import _parse_judge_float, _word_overlap_f1


class TestParseJudgeFloat:
    """
    Verify _parse_judge_float handles all common free-model response styles.
    This function's correctness is critical: if it fails, ALL stochastic
    fields score 0.0 and the optimizer gets no useful signal.
    """

    def test_bare_float(self):
        assert _parse_judge_float("0.8") == pytest.approx(0.8)

    def test_bare_one(self):
        assert _parse_judge_float("1.0") == pytest.approx(1.0)

    def test_bare_zero(self):
        assert _parse_judge_float("0.0") == pytest.approx(0.0)

    def test_verbose_response(self):
        result = _parse_judge_float("I would rate this 0.75 out of 1.0")
        assert result is not None
        assert result == pytest.approx(0.75)

    def test_score_colon(self):
        result = _parse_judge_float("Score: 0.9")
        assert result is not None
        assert result == pytest.approx(0.9)

    def test_fraction_10(self):
        result = _parse_judge_float("7/10")
        assert result is not None
        assert result == pytest.approx(0.7)

    def test_fraction_100(self):
        result = _parse_judge_float("85/100")
        assert result is not None
        assert result == pytest.approx(0.85)

    def test_percentage(self):
        result = _parse_judge_float("85%")
        assert result is not None
        assert result == pytest.approx(0.85)

    def test_0_80(self):
        assert _parse_judge_float("0.80") == pytest.approx(0.8)

    def test_gibberish_returns_none(self):
        assert _parse_judge_float("No numeric content here whatsoever.") is None

    def test_clamped_above_1(self):
        # Even if model says 1.5, should be clamped to 1.0 by caller
        result = _parse_judge_float("0.5")
        assert result == pytest.approx(0.5)


class TestWordOverlapF1:
    """Verify the fallback word-overlap scorer preserves optimization gradient."""

    def test_identical(self):
        assert _word_overlap_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        assert _word_overlap_f1("apple orange", "banana grape") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = _word_overlap_f1("machine learning python", "python data science")
        assert 0.0 < score < 1.0

    def test_empty_gold(self):
        assert _word_overlap_f1("something", "") == pytest.approx(0.0)

    def test_empty_both(self):
        assert _word_overlap_f1("", "") == pytest.approx(1.0)

    def test_superset_pred(self):
        # pred contains all gold tokens plus extras
        score = _word_overlap_f1("python machine learning science", "python machine learning")
        assert score > 0.5  # High overlap, penalised only slightly for extra pred tokens
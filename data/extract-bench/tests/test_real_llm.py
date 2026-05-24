import json
import os
from pathlib import Path

import pytest

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL")
DATA_DIR = Path(__file__).parent / "data"


def _load_json(path: Path) -> dict:
    with path.open("r") as file:
        return json.load(file)


def _metric_config(metric_id: str, pass_threshold: float) -> dict:
    params = {"pass_threshold": pass_threshold}
    if DEFAULT_LLM_MODEL:
        params["model"] = DEFAULT_LLM_MODEL
    return {"metrics": [{"metric_id": metric_id, "params": params}]}


@pytest.mark.asyncio
async def test_string_semantic_real_llm():
    schema = {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "evaluation_config": _metric_config("string_semantic", 0.6),
            }
        },
    }
    gold = {"company": "OpenAI"}
    predicted = {"company": "OpenAI"}

    evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
    result = await evaluator.evaluate_async(schema, gold, predicted)

    metric_result = result["results"]["$.properties.company"]["string_semantic"]
    assert metric_result.score >= 0.0
    assert metric_result.score <= 1.0
    assert metric_result.passed is True
    assert metric_result.details["structured_output"]["score"] >= 0.0


@pytest.mark.asyncio
async def test_array_llm_real_llm():
    schema = {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "evaluation_config": _metric_config("array_llm", 0.5),
                "items": {"type": "string"},
            }
        },
    }
    gold = {"skills": ["Python", "SQL", "Kubernetes"]}
    predicted = {"skills": ["Python", "SQL", "Kubernetes"]}

    evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
    result = await evaluator.evaluate_async(schema, gold, predicted)

    metric_result = result["results"]["$.properties.skills"]["array_llm"]
    assert metric_result.score >= 0.0
    assert metric_result.score <= 1.0
    assert metric_result.passed is True
    assert "matches_summary" in metric_result.details["structured_output"]


@pytest.mark.asyncio
async def test_custom_basic():
    base_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "evaluation_config": "string_semantic"},
            "age": {"type": "integer", "evaluation_config": "integer_exact"},
            "GPA": {
                "type": "number",
                "evaluation_config": {
                    "metrics": [
                        {"metric_id": "number_tolerance", "params": {"tolerance": 0.1}}
                    ]
                },
            },
            "is_student": {"type": "boolean"},
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "subject": {"type": "string"},
                    },
                },
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip": {
                        "type": "string",
                        "evaluation_config": "string_exact",
                    },
                },
            },
        },
    }

    # the evaluation framework only sees the predicted and gold output. no PDF!
    predicted_output = {
        "name": "Sam Doe",
        "age": 20,
        "GPA": 3.5,
        "is_student": True,
        "scores": [
            {"score": 90.1, "subject": "Math"},
            {"score": 92.3, "subject": "Science"},
            {"score": None, "subject": "History"},
            {"score": 94.45, "subject": "English"},
        ],
        "address": {
            "street": "123 main street",
            "city": "anytown",
            "state": "california",
            "zip": "12345",
        },
    }
    gold_output = {
        "name": "John Doe",
        "age": 20,
        "GPA": 3.6,
        "is_student": True,
        "scores": [
            {"score": 92.3, "subject": "Science"},
            {"score": 91.5, "subject": "Math"},
            {"score": 94.5, "subject": "English"},
        ],
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "state": "CA",
            "zip": "12345",
        },
    }
    structured_evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
    result = await structured_evaluator.evaluate_async(
        base_schema, gold_output, predicted_output
    )
    eval_results = result["results"]
    assert result is not None

    expected_metrics = {
        "$.properties.age": [("integer_exact", True)],
        "$.properties.GPA": [("number_tolerance", True)],
        "$.properties.is_student": [("boolean_exact", True)],
        "$.properties.address.properties.zip": [("string_exact", True)],
        "$.properties.address.properties.city": [("string_semantic", True)],
        "$.properties.address.properties.street": [("string_semantic", True)],
        "$.properties.address.properties.state": [("string_semantic", True)],
        "$.properties.name": [("string_semantic", False)],
        "$.properties.scores": [("array_llm", False)],
    }

    assert set(eval_results.keys()) == set(expected_metrics.keys())

    for path, metrics in expected_metrics.items():
        path_metrics = eval_results[path]
        expected_metric_ids = {metric_id for metric_id, _ in metrics}
        assert set(path_metrics.keys()) == expected_metric_ids
        for metric_id, expected_passed in metrics:
            metric_result = path_metrics[metric_id]
            assert metric_result.passed == expected_passed


@pytest.mark.asyncio
async def test_array_basic():
    json_schema = {
        "type": "object",
        "properties": {
            "education": {
                "type": "array",
                "evaluation_config": {
                    "metrics": [
                        {
                            "metric_id": "array_llm",
                            "params": {
                                "pass_threshold": 0.7,
                                "additional_instructions": (
                                    "Treat school names as semantic equivalents. "
                                    "For example, 'Stanford' and 'Stanford University' "
                                    "should be considered a match."
                                ),
                            },
                        }
                    ]
                },
                "items": {
                    "type": "object",
                    "properties": {
                        "degree": {
                            "type": "string",
                            "evaluation_config": "string_exact",
                        },
                        "school": {
                            "type": "string",
                            "evaluation_config": "string_semantic",
                        },
                        "year": {
                            "type": "integer",
                            "evaluation_config": "integer_exact",
                        },
                    },
                },
            }
        },
    }

    gold_json = {
        "education": [
            {"degree": "PhD", "school": "MIT", "year": 2020},
            {"degree": "BS", "school": "Stanford", "year": 2015},
        ]
    }
    extracted_json = {
        "education": [
            {"degree": "BS", "school": "Stanford University", "year": 2015},
            {"degree": "PhD", "school": "MIT", "year": 2020},
        ]
    }

    structured_evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
    result = await structured_evaluator.evaluate_async(
        json_schema, gold_json, extracted_json
    )
    eval_results = result["results"]

    assert set(eval_results.keys()) == {"$.properties.education"}
    metric_result = eval_results["$.properties.education"]["array_llm"]
    assert metric_result.passed is True
    assert metric_result.score >= 0.7

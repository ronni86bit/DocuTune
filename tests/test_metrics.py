"""Metric computation: perfect predictions, known errors, unit accounting."""

import copy
import json

from docutune.evaluation.bootstrap import summarize_bootstrap
from docutune.evaluation.error_analysis import categorize_prediction
from docutune.evaluation.metrics import (
    aggregate_example_metrics,
    evaluate_example,
    evaluate_predictions,
)


def _eval(raw_output: str, gold: dict):
    from docutune.evaluation.parser import parse_model_output

    result = parse_model_output(raw_output)
    parsed = result.parsed if result.is_object else None
    return evaluate_example(parsed, gold)


def test_perfect_prediction_scores_one(small_dataset):
    gold = small_dataset["examples"][0]["target"]
    ev = _eval(json.dumps(gold), gold)
    assert ev["exact_match"] is True
    assert ev["tp"] > 0
    assert ev["fp"] == 0 and ev["fn"] == 0
    assert ev["field_f1"] == 1.0
    assert ev["unsupported_units"] == 0


def test_missing_scalar_counts_fn_only(small_dataset):
    gold = copy.deepcopy(small_dataset["examples"][0]["target"])
    pred = copy.deepcopy(gold)
    pred["email"] = None
    ev = _eval(json.dumps(pred), gold)
    assert ev["exact_match"] is False
    assert ev["fn"] >= 1
    assert ev["per_field"]["email"]["fn"] == 1
    assert ev["per_field"]["email"]["fp"] == 0
    assert ev["missing_fields"] >= 1


def test_wrong_value_counts_fp_and_fn(small_dataset):
    gold = copy.deepcopy(small_dataset["examples"][0]["target"])
    pred = copy.deepcopy(gold)
    pred["name"] = "Completely Different"
    ev = _eval(json.dumps(pred), gold)
    assert ev["per_field"]["name"] == {"tp": 0, "fp": 1, "fn": 1}
    assert ev["unsupported_units"] == 1


def test_hallucinated_value_when_gold_null(small_dataset):
    gold = copy.deepcopy(small_dataset["examples"][0]["target"])
    gold["email"] = None
    pred = copy.deepcopy(gold)
    pred["email"] = "invented@example.com"
    ev = _eval(json.dumps(pred), gold)
    assert ev["per_field"]["email"] == {"tp": 0, "fp": 1, "fn": 0}
    assert ev["unsupported_units"] == 1


def test_extra_skill_counts_fp(small_dataset):
    gold = copy.deepcopy(small_dataset["examples"][0]["target"])
    pred = copy.deepcopy(gold)
    pred["skills"] = list(gold["skills"]) + ["MadeUpSkill"]
    ev = _eval(json.dumps(pred), gold)
    stats = ev["per_field"]["skills"]
    assert stats["fp"] == 1
    assert stats["tp"] == len(gold["skills"])


def test_date_format_normalization_not_penalized():
    gold = {
        "experience": [{
            "title": "MLE", "company": "Nova", "current": True,
            "start_date": "2022-06", "end_date": None, "responsibilities": [],
        }],
        "skills": [],
    }
    pred = {
        "experience": [{
            "title": "MLE", "company": "Nova", "current": True,
            "start_date": "June 2022", "end_date": "Present", "responsibilities": [],
        }],
        "skills": [],
    }
    ev = _eval(json.dumps(pred), gold)
    assert ev["exact_match"] is True


def test_aggregate_rates(small_dataset):
    gold = small_dataset["examples"][0]["target"]
    perfect = _eval(json.dumps(gold), gold)
    empty = evaluate_example(None, gold)
    aggregate = aggregate_example_metrics([perfect, empty])
    assert aggregate["n_examples"] == 2
    assert 0.0 < aggregate["field_recall"] < 1.0
    assert aggregate["unsupported_value_rate"] == 0.0
    assert aggregate["missing_value_rate"] > 0.0


def test_evaluate_predictions_end_to_end(small_dataset):
    examples = small_dataset["examples"]
    predictions = [
        {"id": ex["id"], "raw_output": json.dumps(ex["target"]), "latency_ms": 10.0}
        for ex in examples
    ]
    predictions.append({"id": examples[0]["id"], "raw_output": "garbage", "latency_ms": 5.0})
    metrics = evaluate_predictions(predictions, examples)
    assert metrics["json_validity"] > 0.5
    assert metrics["schema_validity"] < 1.0
    assert "latency_mean_ms" in metrics
    assert metrics["_versions"]["evaluator_version"] == "1.0"


def test_bootstrap_cis(small_dataset):
    gold = small_dataset["examples"][0]["target"]
    evals = [_eval(json.dumps(gold), gold)]
    evals[0]["schema_valid"] = True
    evals.append(evaluate_example(None, gold))
    evals[1]["schema_valid"] = False
    summary = summarize_bootstrap(evals, n_boot=100, seed=42)
    assert 0.0 <= summary["schema_validity"]["low"] <= summary["schema_validity"]["point"]
    assert summary["schema_validity"]["point"] == 0.5


def test_error_categories(small_dataset):
    gold = small_dataset["examples"][0]["target"]
    perfect = categorize_prediction(json.dumps(gold), gold)
    assert perfect["primary_category"] == "none"
    garbage = categorize_prediction("no json here", gold)
    assert garbage["primary_category"] == "malformed_json"
    wrong = categorize_prediction(json.dumps({**gold, "phone": "000"}), gold)
    assert "wrong_value" in wrong["categories"] or "unsupported_value" in wrong["categories"]

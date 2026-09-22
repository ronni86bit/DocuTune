# Evaluation Documentation

## Fairness Contract (the most important rule)

The base and fine-tuned models are evaluated under **identical** conditions:

| Element | Shared? | Where |
|---|---|---|
| Test set | yes — same 75 held-out examples | `data/splits/test.jsonl` |
| Prompt | yes — one canonical template, `PROMPT_VERSION=1.0` | `docutune/inference/prompt.py` |
| Decoding | yes — greedy (`do_sample=false`, `num_beams=1`, `max_new_tokens=700`) | `configs/eval.yaml` |
| Parser | yes — one robust parser, no per-model branches | `docutune/evaluation/parser.py` |
| Normalization | yes — one documented rule set | `docutune/evaluation/normalization.py` |
| Evaluator | yes — one metric engine | `docutune/evaluation/metrics.py` |

Only the model state differs (base weights vs. base weights + LoRA adapter). The cache
system enforces this structurally: predictions are fingerprinted by (model, revision,
adapter, dataset version, prompt version, evaluator version, generation config, test-file
hash) and stale caches are refused unless `--force` is given.

## Metric Definitions

Notation: `P` = predicted record, `G` = gold canonical record. Predictions first pass
the parser; outputs that fail to parse are treated as **empty extractions** (all recall
failures — never dropped silently).

### A. JSON Validity (`json_validity`)
`|{x : parse(x) succeeds}| / |examples|`. Purely syntactic. A wrapped-in-prose JSON that
the brace-recovery path salvages counts as valid; truncated JSON does not.

### B. Schema Validity (`schema_validity`)
`|{x : parse(x) succeeds ∧ PydanticResumeExtraction(x) succeeds}| / |examples|`.
Strict validation: correct types, all list items well-formed, **no extra keys**. This
measures whether the model respects the output contract.

### C. Exact Record Match (`exact_match`)
`|{x : normalize(P_x) == normalize(G_x)}| / |examples|` — full-record equality after
conservative normalization (below), over schema fields (R11: extra keys are penalized by
schema validity, not here).

### D–F. Field Precision / Recall / F1
Extraction units per field kind:
- **scalar fields** (name, email, phone, location, summary): 1 unit when non-null
- **string lists** (skills): 1 unit per item, multiset matching
- **record lists** (education, experience, projects, certifications): 1 unit per item,
  greedy matching by exact normalized-item equality

Then per field: `tp` = matched units; `fp` = predicted units with no gold match;
`fn` = gold units with no predicted match. A wrong scalar value counts as 1 fp **and**
1 fn. Micro P/R/F1 pool all fields; `results/per_field_metrics.csv` reports per-field.

### G. Unsupported-Value Rate (`unsupported_value_rate`)
`unsupported predicted units / predicted units`, where a unit is unsupported iff it has
no equal gold counterpart. Operationally this equals `Σfp / Σ(tp+fp)`. Because the
dataset is synthetic and the canonical record is known, this is well-defined. **This is
a benchmark-specific proxy for hallucination — not a universal hallucination measure.**
No semantic repair is ever applied that could artificially lower it. Where useful, raw
outputs are also checked for textual support against the source resume in analysis.

### H. Missing-Value Rate (`missing_value_rate`)
Field-level: `#{f : G[f] non-empty ∧ P[f] empty} / #{f : G[f] non-empty}` over the 10
top-level fields. Unparseable outputs miss every gold field by definition.

### I. Latency
`latency_ms` covers tokenize → generate → detokenize per example. Model loading is
excluded; warm-up generations run before measurement and are discarded; decoding is
identical for both models; device (CPU/GPU) and GPU name are recorded in the manifests.
Mean, p50 and p95 are reported.

### J. Output Length
Mean raw-output character length per model (diagnostic; detects degenerate rambling).

## Conservative Normalization Rules (R1–R12)

Implemented in `docutune/evaluation/normalization.py`; each rule exists to remove trivial
surface differences, never to rescue wrong values:

1. **R1** trim + collapse internal whitespace
2. **R2** empty-ish scalars (`""`, `n/a`, `none`, `null`, `-`, `?`) → null
3. **R3** case-insensitive comparison only for email/skills/degree (case-free fields);
   names/summaries stay case-sensitive
4. **R4** phones: strip spaces, dashes, dots, parentheses (keep leading `+`)
5. **R5** emails lowercased (via R3)
6. **R6** dates canonicalized to `YYYY-MM`/`YYYY`; month names/abbreviations,
   `MM/YYYY`, `YYYY-MM`, `YYYY` accepted; `present/current/now/till date` → PRESENT
   sentinel (not a date)
7. **R7** experience `end_date == PRESENT` equals gold `null` when `current=true`
8. **R8** lists are order-insensitive (multiset semantics)
9. **R9** grades: `CGPA: 8.5/10` → `8.5`; `85%` stays `85%`
10. **R10** years compare as integers
11. **R11** schema-unknown keys are ignored by exact-match comparison (schema validity
    penalizes them separately)
12. **R12** wrong JSON types are never coerced (`"skills": "Python"` is a field failure,
    not `["Python"]`)

## Bootstrap Confidence Intervals

Percentile bootstrap (1000 resamples, seed 42 — deterministic) over test examples for:
schema validity, exact match, field F1 (ratio of pooled counts), unsupported-value rate
(ratio-of-sums). Reported in `results/metrics.json` under `bootstrap`.

**Interpretation boundary:** these intervals quantify sampling uncertainty *on this
held-out synthetic benchmark*. They do not establish statistical significance for any
claim about other data, domains or models.

## Error Analysis Categories

`docutune/evaluation/error_analysis.py` categorizes each prediction (multiple categories
possible; primary = highest priority):

`malformed_json` → `schema_violation` → `list_item_error` → `date_normalization_error`
(raw format not normalized but semantically correct) → `section_association_error`
(content attached to the wrong employer) → `wrong_value` → `unsupported_value` →
`over_extraction` (field present in output, empty in gold) → `missing_field` →
`under_extraction` (partial list extraction). Perfect predictions get `none`.

Representative examples are selected deterministically (seeded random sample, biggest
improvement, worst regression, base-failure→fine-tuned-success, fine-tuned-failure→
base-success) and explicitly **include failures** — successes-only showcases are not
representative.

## Artifacts

`scripts/benchmark.py` writes into `results/`:
`baseline_predictions.jsonl`, `finetuned_predictions.jsonl`, `comparison.jsonl`,
`metrics.json`, `metrics.csv`, `per_field_metrics.csv`, `error_analysis.jsonl`,
`error_analysis_summary.csv`, `benchmark_manifest.json`, `benchmark_report.md/.html`,
`FINAL_REPORT.md`, `charts/*.png`, `reports/representative_examples.json` — and updates
the README benchmark table from measured values only.

## Resumability & Caching

- Predictions are appended row-by-row with flush+fsync (`--resume`)
- Cache fingerprints live in `results/*.cache.json`; mismatched fingerprints abort with
  instructions (`--force` recomputes deliberately)
- `scripts/run_experiment.py` chains validation → both passes → metrics → reports

"""Benchmark reporting: metrics files, charts, Markdown/HTML reports and the
README metric-table updater.

Hard rule honored here: values rendered into reports come ONLY from
measured results. When results are absent, the README keeps its TBD state.
"""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path
from typing import Any

from docutune.utils.io import read_json, write_text
from docutune.utils.logging import get_logger

logger = get_logger(__name__)

HEADLINE_KEYS = [
    ("json_validity", "JSON Validity", "pct"),
    ("schema_validity", "Schema Validity", "pct"),
    ("exact_match", "Exact Match", "pct"),
    ("field_precision", "Field Precision", "pct"),
    ("field_recall", "Field Recall", "pct"),
    ("field_f1", "Field F1", "pct"),
    ("unsupported_value_rate", "Unsupported Value Rate", "pct"),
    ("missing_value_rate", "Missing Value Rate", "pct"),
    ("latency_mean_ms", "Mean Latency (ms)", "ms"),
    ("latency_p95_ms", "P95 Latency (ms)", "ms"),
]


def _fmt(value: Any, kind: str) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "TBD"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    return f"{value:.1f}"


def _delta_fmt(base: Any, ft: Any, kind: str) -> str:
    if not isinstance(base, (int, float)) or not isinstance(ft, (int, float)):
        return "TBD"
    d = ft - base
    sign = "+" if d >= 0 else ""
    if kind == "pct":
        return f"{sign}{d * 100:.1f}pp"
    return f"{sign}{d:.1f}"


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------
def write_headline_metrics_csv(path: str, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "base", "finetuned", "delta"])
        for key, label, kind in HEADLINE_KEYS:
            b = metrics["base"].get(key)
            f = metrics["finetuned"].get(key)
            writer.writerow([label, b, f, _delta_fmt(b, f, kind)])


def write_per_field_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# README metric table + resume bullet
# ---------------------------------------------------------------------------
README_TABLE_START = "<!-- BENCHMARK-TABLE:START -->"
README_TABLE_END = "<!-- BENCHMARK-TABLE:END -->"
README_BULLET_START = "<!-- RESUME-BULLET:START -->"
README_BULLET_END = "<!-- RESUME-BULLET:END -->"


def render_readme_table(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return (
            "| Metric | Base | Fine-Tuned | Delta |\n"
            "|---|---:|---:|---:|\n"
            "| JSON Validity | TBD | TBD | TBD |\n"
            "| Schema Validity | TBD | TBD | TBD |\n"
            "| Exact Match | TBD | TBD | TBD |\n"
            "| Field Precision | TBD | TBD | TBD |\n"
            "| Field Recall | TBD | TBD | TBD |\n"
            "| Field F1 | TBD | TBD | TBD |\n"
            "| Unsupported Value Rate | TBD | TBD | TBD |\n"
            "| Mean Latency | TBD | TBD | TBD |"
        )
    lines = ["| Metric | Base | Fine-Tuned | Delta |", "|---|---:|---:|---:|"]
    for key, label, kind in HEADLINE_KEYS:
        b = metrics["base"].get(key)
        f = metrics["finetuned"].get(key)
        lines.append(f"| {label} | {_fmt(b, kind)} | {_fmt(f, kind)} | {_delta_fmt(b, f, kind)} |")
    return "\n".join(lines)


def render_resume_bullet(metrics: dict[str, Any] | None, metadata: dict[str, Any],
                         train_size: int = 450) -> str:
    if not metrics:
        return "TBD - generated automatically from measured benchmark results."
    b, f = metrics["base"], metrics["finetuned"]
    model = metadata.get("base_model", "the base model")
    test_size = metadata.get("test_size", "held-out")
    return (
        f"Fine-tuned {model} using LoRA/QLoRA on {train_size} synthetic resume-extraction "
        f"examples, improving schema validity from {_fmt(b.get('schema_validity'), 'pct')} to "
        f"{_fmt(f.get('schema_validity'), 'pct')} and field-level F1 from "
        f"{_fmt(b.get('field_f1'), 'pct')} to {_fmt(f.get('field_f1'), 'pct')} on a held-out "
        f"{test_size}-example benchmark (unsupported-value rate "
        f"{_fmt(b.get('unsupported_value_rate'), 'pct')} -> "
        f"{_fmt(f.get('unsupported_value_rate'), 'pct')})."
    )


def _replace_between(text: str, start: str, end: str, new_content: str) -> str:
    pattern = re.compile(re.escape(start) + r"\n(.*?)" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README markers not found: {start}")
    return pattern.sub(start + "\n" + new_content + "\n" + end, text)


def update_readme_metrics(metrics_path: str, readme_path: str) -> bool:
    """Update the README benchmark table from results/metrics.json.

    Returns True when the README was updated. Refuses to run (raises) when
    metrics.json does not exist - the README keeps TBD values until a real
    benchmark run has happened. Never invents numbers.
    """
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} not found - nothing measured yet; README stays TBD. "
            "Run scripts/benchmark.py (or scripts/run_experiment.py) first."
        )
    metrics = read_json(path)
    readme = Path(readme_path).read_text(encoding="utf-8")
    train_size = metrics.get("metadata", {}).get("train_size", 450)
    updated = _replace_between(readme, README_TABLE_START, README_TABLE_END,
                               render_readme_table(metrics))
    updated = _replace_between(updated, README_BULLET_START, README_BULLET_END,
                               render_resume_bullet(metrics, metrics.get("metadata", {}),
                                                    train_size))
    if updated != readme:
        write_text(readme_path, updated)
        logger.info("README metrics section updated from %s", metrics_path)
        return True
    return False


# ---------------------------------------------------------------------------
# Charts (matplotlib, Agg backend)
# ---------------------------------------------------------------------------
def generate_charts(metrics: dict[str, Any], charts_dir: str,
                    training_log_path: str | None = None) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - charts skipped")
        return []

    out: list[str] = []
    charts = Path(charts_dir)
    charts.mkdir(parents=True, exist_ok=True)
    base, ft = metrics.get("base", {}), metrics.get("finetuned", {})

    def bar_chart(key: str, title: str, filename: str, pct: bool) -> None:
        b, f = base.get(key), ft.get(key)
        if not isinstance(b, (int, float)) or not isinstance(f, (int, float)):
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        values = [b * 100 if pct else b, f * 100 if pct else f]
        bars = ax.bar(["Base", "Fine-Tuned"], values, color=["#94a3b8", "#4f46e5"])
        ax.set_title(title)
        for bar, val in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}" + ("%" if pct else ""), ha="center", va="bottom", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=150)
        plt.close(fig)
        out.append(str(charts / filename))

    bar_chart("schema_validity", "Schema Validity", "schema_validity.png", pct=True)
    bar_chart("exact_match", "Exact Record Match", "exact_match.png", pct=True)
    bar_chart("field_f1", "Field F1 (micro)", "field_f1.png", pct=True)
    bar_chart("unsupported_value_rate", "Unsupported Value Rate",
              "unsupported_value_rate.png", pct=True)

    # Latency chart (mean / p50 / p95 per model)
    lat_keys = [
        ("latency_mean_ms", "mean"), ("latency_p50_ms", "p50"), ("latency_p95_ms", "p95"),
    ]
    if all(isinstance(base.get(k), (int, float)) for k, _ in lat_keys):
        import numpy as np

        x = np.arange(3)
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, [base[k] for k, _ in lat_keys], width,
               label="Base", color="#94a3b8")
        ax.bar(x + width / 2, [ft[k] for k, _ in lat_keys], width,
               label="Fine-Tuned", color="#4f46e5")
        ax.set_xticks(x, [lbl for _, lbl in lat_keys])
        ax.set_ylabel("ms")
        ax.set_title("Latency (same environment, warm model)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(charts / "latency.png", dpi=150)
        plt.close(fig)
        out.append(str(charts / "latency.png"))

    # Per-field F1 grouped bars
    per_field_b = base.get("per_field", {})
    per_field_f = ft.get("per_field", {})
    fields = [f for f in per_field_b if f in per_field_f]
    if fields:
        import numpy as np

        y = np.arange(len(fields))
        height = 0.38
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(fields) + 1.6))
        ax.barh(y - height / 2, [per_field_b[f]["f1"] * 100 for f in fields], height,
                label="Base", color="#94a3b8")
        ax.barh(y + height / 2, [per_field_f[f]["f1"] * 100 for f in fields], height,
                label="Fine-Tuned", color="#4f46e5")
        ax.set_yticks(y, fields)
        ax.set_xlabel("F1 (%)")
        ax.set_title("Per-Field F1")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(charts / "per_field_f1.png", dpi=150)
        plt.close(fig)
        out.append(str(charts / "per_field_f1.png"))

    # Training loss curve, when a training log exists
    if training_log_path and Path(training_log_path).is_file():
        try:
            history = read_json(training_log_path)
            steps = [e["step"] for e in history if "loss" in e and "step" in e]
            losses = [e["loss"] for e in history if "loss" in e and "step" in e]
            if steps:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(steps, losses, color="#4f46e5")
                ax.set_xlabel("step")
                ax.set_ylabel("training loss")
                ax.set_title("Training Loss")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                fig.savefig(charts / "training_loss.png", dpi=150)
                plt.close(fig)
                out.append(str(charts / "training_loss.png"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not render training_loss chart: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Markdown + HTML reports
# ---------------------------------------------------------------------------
def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render_benchmark_report(metrics: dict[str, Any], error_summary: list[dict[str, Any]] | None,
                            manifest: dict[str, Any], final: bool = False) -> str:
    meta = metrics.get("metadata", {})
    base, ft = metrics.get("base", {}), metrics.get("finetuned", {})
    lines: list[str] = []
    title = "# DocuTune Before/After Evaluation" if final else "# DocuTune Benchmark Report"
    lines.append(title)
    lines.append("")
    lines.append("## Experiment")
    lines.append("")
    lines.append(
        "Untouched base model vs. LoRA/QLoRA fine-tuned model on the exact same held-out "
        "synthetic test set, with the exact same prompt, decoding configuration, parser, "
        "normalization and evaluator. Only the model state differs."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"- dataset_version: {meta.get('dataset_version')}, "
        f"schema_version: {meta.get('schema_version')}"
    )
    lines.append(f"- test size: {meta.get('test_size')} (held-out rendering templates 11-12)")
    lines.append(f"- seed: {meta.get('seed')}")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(
        f"- base model: `{meta.get('base_model')}` "
        f"(revision: {meta.get('base_model_revision') or 'default'})"
    )
    lines.append(f"- adapter: `{meta.get('adapter') or 'n/a'}`")
    lines.append(
        f"- prompt_version: {meta.get('prompt_version')}, "
        f"evaluator_version: {meta.get('evaluator_version')}"
    )
    lines.append("")
    lines.append("## Fine-Tuning Configuration")
    lines.append("")
    lines.append("See `artifacts/training/training_manifest.json` for the recorded run.")
    lines.append("")
    lines.append("## Baseline Results")
    lines.append("")
    lines.append(_md_table(["Metric", "Base"],
                           [[label, _fmt(base.get(k), kind)] for k, label, kind in HEADLINE_KEYS]))
    lines.append("")
    lines.append("## Fine-Tuned Results")
    lines.append("")
    lines.append(_md_table(["Metric", "Fine-Tuned"],
                           [[label, _fmt(ft.get(k), kind)] for k, label, kind in HEADLINE_KEYS]))
    lines.append("")
    lines.append("## Metric Deltas (fine-tuned - base)")
    lines.append("")
    lines.append(_md_table(
        ["Metric", "Base", "Fine-Tuned", "Delta"],
        [[label, _fmt(base.get(k), kind), _fmt(ft.get(k), kind),
          _delta_fmt(base.get(k), ft.get(k), kind)] for k, label, kind in HEADLINE_KEYS]))
    lines.append("")
    lines.append("## Per-Field Metrics")
    lines.append("")
    per_field = metrics.get("per_field") or []
    if per_field:
        lines.append(_md_table(
            ["Field", "Base F1", "Fine-Tuned F1", "Delta F1"],
            [[r["field"], _fmt(r.get("base_f1"), "pct"), _fmt(r.get("finetuned_f1"), "pct"),
              _delta_fmt(r.get("base_f1"), r.get("finetuned_f1"), "pct")] for r in per_field]))
    else:
        lines.append("_No per-field metrics available._")
    lines.append("")
    lines.append("## Error Analysis")
    lines.append("")
    if error_summary:
        lines.append(_md_table(
            ["Category", "Base", "Fine-Tuned", "Base %", "Fine-Tuned %"],
            [[r["category"], r["base_count"], r["finetuned_count"],
              f"{r['base_pct']:.1f}%", f"{r['finetuned_pct']:.1f}%"] for r in error_summary]))
    else:
        lines.append("_No error analysis available._")
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append(
        f"- device: {meta.get('device')}, gpu: {meta.get('gpu')}"
    )
    lines.append(
        "- warm model, warm-up generations excluded; identical decoding for both models"
    )
    lines.append("")
    boot = metrics.get("bootstrap") or {}
    lines.append("## Confidence Intervals (95% bootstrap, this benchmark only)")
    lines.append("")
    for model_name in ("base", "finetuned"):
        model_boot = boot.get(model_name) or {}
        if model_boot:
            lines.append(f"- **{model_name}**: " + "; ".join(
                f"{k} [{v['low']:.3f}, {v['high']:.3f}]" for k, v in model_boot.items()))
    lines.append("")
    lines.append("These intervals quantify sampling uncertainty on this held-out synthetic "
                 "benchmark only; they do not establish universal superiority.")
    lines.append("")
    lines.append("## Representative Examples")
    lines.append("")
    lines.append("See `results/reports/representative_examples.json` - selected deterministically "
                 "and including failures: random (seeded), biggest improvement, regression, and "
                 "both base/fine-tuned failure-swap directions.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Synthetic dataset and synthetic language bias; real-resume domain shift is likely."
    )
    lines.append("- Small test set (75 examples by default).")
    lines.append(
        "- 'Unsupported-value rate' is defined against the gold canonical record "
        "(see docs/EVALUATION.md)."
    )
    lines.append("- Results apply to this benchmark configuration only.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/generate_data.py --seed 42")
    lines.append("python scripts/validate_data.py")
    lines.append("python scripts/run_baseline.py")
    lines.append("python -m docutune.training.train --config configs/train.yaml")
    lines.append("python scripts/run_finetuned.py")
    lines.append("python scripts/benchmark.py")
    lines.append("```")
    lines.append("")
    lines.append(f"_Generated: {manifest.get('timestamp')}_")
    return "\n".join(lines)


def markdown_to_html(md: str) -> str:
    """Tiny Markdown -> HTML converter covering the constructs used in reports:
    headings, tables, fenced code, lists, bold, inline code, hr, paragraphs."""
    out: list[str] = []
    lines = md.split("\n")
    i = 0

    def inline(text: str) -> str:
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        return text

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-{3,}", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head, *body = rows
                t = ["<table>", "<thead><tr>",
                     "".join(f"<th>{inline(c)}</th>" for c in head), "</tr></thead><tbody>"]
                for r in body:
                    t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
                t.append("</tbody></table>")
                out.append("".join(t))
            continue
        elif re.match(r"^#{1,4} ", line):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level + 1:])}</h{level}>")
        elif line.strip() == "---":
            out.append("<hr/>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>DocuTune Benchmark Report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;color:#1e293b}table{border-collapse:collapse;margin:1rem 0}"
        "th,td{border:1px solid #cbd5e1;padding:.35rem .7rem;text-align:left}"
        "th{background:#f1f5f9}code{background:#f1f5f9;padding:.1rem .3rem;border-radius:4px}"
        "pre code{display:block;padding:1rem;overflow-x:auto}h1{color:#312e81}</style>"
        "</head><body>" + "\n".join(out) + "</body></html>"
    )

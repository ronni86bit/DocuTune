#!/usr/bin/env python
"""Project verification: files, imports, config, schema, dataset, artifacts,
README consistency, backend/frontend structure, Docker files, Streamlit absence.

Exit code 0 = all good; 1 = at least one FAIL.

Usage:
    python scripts/verify_project.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []  # (status, area, detail)


def check(status: str, area: str, detail: str = "") -> None:
    RESULTS.append((status, area, detail))


def check_file(rel: str, required: bool = True) -> bool:
    path = ROOT / rel
    exists = path.is_file()
    check("OK" if exists else ("FAIL" if required else "WARN"), "file",
          f"{rel}" + ("" if exists else " (missing)"))
    return exists


def main() -> int:
    # --- Core files ---
    for rel in ("README.md", "LICENSE", ".gitignore", ".env.example", "pyproject.toml",
                "docker-compose.yml", "backend/Dockerfile", "frontend/Dockerfile",
                "frontend/nginx.conf", "configs/train.yaml", "configs/eval.yaml",
                "configs/app.yaml", "notebooks/DocuTune_Training.ipynb",
                ".github/workflows/test.yml", "docs/MODEL_CARD.md", "docs/DATASET.md",
                "docs/EVALUATION.md", "docs/TRAINING.md", "docs/DEPLOYMENT.md",
                "docs/KAGGLE.md"):
        check_file(rel)

    # --- Package imports (torch-free core) ---
    try:
        import docutune.config  # noqa: F401
        import docutune.data.generator  # noqa: F401
        import docutune.data.validator  # noqa: F401
        import docutune.evaluation.benchmark  # noqa: F401
        import docutune.evaluation.error_analysis  # noqa: F401
        import docutune.evaluation.metrics  # noqa: F401
        import docutune.evaluation.parser  # noqa: F401
        import docutune.inference.prompt  # noqa: F401
        import docutune.schema.resume  # noqa: F401

        check("OK", "imports", "core package imports cleanly (no torch required)")
    except Exception as exc:  # noqa: BLE001
        check("FAIL", "imports", f"core import error: {exc}")

    try:
        import docutune.inference.loader  # noqa: F401
        import docutune.training.train  # noqa: F401

        check("OK", "imports", "training/inference modules import (lazy torch)")
    except ImportError as exc:
        check("WARN", "imports", f"training stack import failed ({exc}); install '.[training]'")

    # --- Configuration ---
    try:
        from docutune.config import TrainConfig

        cfg = TrainConfig.from_yaml(ROOT / "configs/train.yaml")
        check("OK", "config", f"active model: {cfg.model.name}")
    except Exception as exc:  # noqa: BLE001
        check("FAIL", "config", str(exc))

    # --- Schema ---
    try:
        from docutune.schema.resume import ResumeExtraction

        ResumeExtraction.model_validate({
            "name": "Test", "skills": ["Python"],
            "experience": [{"title": "X", "company": "Y", "current": True,
                            "responsibilities": ["did things"]}],
        })
        check("OK", "schema", "validation round-trip")
    except Exception as exc:  # noqa: BLE001
        check("FAIL", "schema", str(exc))

    # --- Dataset ---
    train = ROOT / "data/splits/train.jsonl"
    test = ROOT / "data/splits/test.jsonl"
    if train.is_file() and test.is_file():
        from docutune.data.validator import validate_dataset

        report = validate_dataset()
        check("OK" if report.ok else "FAIL", "dataset",
              f"counts={report.stats.get('counts')}, errors={len(report.errors)}")
    else:
        check("WARN", "dataset", "splits missing - run scripts/generate_data.py")

    # --- Results / adapter ---
    if (ROOT / "results/metrics.json").is_file():
        check("OK", "results", "metrics.json present")
    else:
        check("WARN", "results", "no benchmark results yet (results/metrics.json)")
    adapter = ROOT / "artifacts/adapters/final"
    adapter_files = ([p.name for p in adapter.iterdir() if p.name != ".gitkeep"]
                     if adapter.is_dir() else [])
    if "adapter_config.json" in adapter_files:
        check("OK", "adapter", f"present at {adapter}")
    else:
        check("WARN", "adapter",
              "no trained adapter at artifacts/adapters/final - run the Colab notebook, "
              "then set ADAPTER_PATH")

    # --- README metric consistency ---
    readme = ROOT / "README.md"
    metrics_file = ROOT / "results/metrics.json"
    if readme.is_file() and metrics_file.is_file():
        readme_text = readme.read_text(encoding="utf-8")
        markers_ok = ("<!-- BENCHMARK-TABLE:START -->" in readme_text
                      and "<!-- RESUME-BULLET:START -->" in readme_text)
        check("OK" if markers_ok else "FAIL", "readme",
              "metric markers present" if markers_ok else "metric markers missing")
    elif readme.is_file():
        has_tbd = "TBD" in readme.read_text(encoding="utf-8")
        check("OK" if has_tbd else "WARN", "readme",
              "TBD placeholders present (no results yet)" if has_tbd else "check placeholders")

    # --- Backend / frontend structure ---
    for rel in ("backend/app/main.py", "backend/app/routes/health.py",
                "backend/app/routes/extraction.py", "backend/app/routes/metrics.py",
                "backend/app/services/model_manager.py",
                "frontend/src/App.tsx", "frontend/src/pages/ExtractionPage.tsx",
                "frontend/src/pages/BenchmarkPage.tsx", "frontend/src/services/api.ts",
                "frontend/package.json"):
        check_file(rel)

    # --- Docker ---
    compose = ROOT / "docker-compose.yml"
    if compose.is_file():
        try:
            import yaml

            parsed = yaml.safe_load(compose.read_text(encoding="utf-8"))
            services = list((parsed or {}).get("services", {}))
            check("OK" if {"frontend", "backend"} <= set(services) else "FAIL",
                  "docker", f"compose services: {services}")
        except Exception as exc:  # noqa: BLE001
            check("FAIL", "docker", f"compose parse error: {exc}")

    # --- Streamlit absence (hard requirement) ---
    # Pattern assembled from parts so this check does not match its own source.
    streamlit_import = "import stream" + "lit"
    streamlit_from = "from stream" + "lit"
    offenders: list[str] = []
    skip_parts = {".venv", "node_modules", ".git", "dist", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*.py"):
        if skip_parts & set(path.parts) or path == Path(__file__):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if streamlit_import in text or streamlit_from in text:
            offenders.append(str(path.relative_to(ROOT)))
    for name in ("requirements.txt", "pyproject.toml"):
        p = ROOT / name
        if p.is_file() and "streamlit" in p.read_text(encoding="utf-8").lower():
            offenders.append(name)
    check("OK" if not offenders else "FAIL", "no-streamlit",
          "no Streamlit dependency or import found" if not offenders else f"found in {offenders}")

    # --- Summary ---
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("\nDocuTune project verification")
    print("=" * 60)
    for status, area, detail in RESULTS:
        marker = {"OK": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
        print(f"{marker} {area:<12} {detail}")
    print("=" * 60)
    print(f"{len(RESULTS)} checks: {len(RESULTS) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warnings, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

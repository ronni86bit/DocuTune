import { Card } from '../components/ui'

export function AboutPage() {
  return (
    <div className="page page-narrow">
      <div className="page-heading">
        <h2>About DocuTune</h2>
        <p>
          A controlled fine-tuning experiment: does parameter-efficient fine-tuning improve
          structured resume extraction for a small open-source instruction model? Measured on a
          held-out benchmark, not asserted.
        </p>
      </div>

      <Card title="Why fine-tuning?">
        <p>
          General instruction models often produce plausible but inconsistent structured output:
          invented keys, hallucinated fields, unstable date formats. Fine-tuning on task-specific
          supervised examples teaches the model the exact output contract. Whether that actually
          helps is an empirical question — this project measures it with a controlled before/after
          comparison rather than assuming it.
        </p>
      </Card>

      <Card title="Why LoRA / QLoRA?">
        <p>
          LoRA (Low-Rank Adaptation) freezes the base model weights and trains small low-rank
          adapters instead — a few percent of the parameters. QLoRA combines it with 4-bit NF4
          quantization of the frozen base, which brings training of a 3–4B parameter model within
          reach of a free Google Colab T4 GPU. The result is a ~50 MB adapter instead of a
          multi-GB model copy.
        </p>
      </Card>

      <Card title="Why structured extraction?">
        <p>
          Resume-to-JSON is a bounded, verifiable task: the schema is strict, gold labels can be
          checked exactly, and quality can be measured field by field (precision, recall, F1,
          schema validity, unsupported values). That makes it a good vehicle for demonstrating a
          rigorous evaluation methodology — much more informative than a vibes-based demo.
        </p>
      </Card>

      <Card title="How was the dataset created?">
        <p>
          600 fully synthetic examples generated deterministically (seed 42) and locally — no real
          people's resumes, no paid APIs. Canonical structured records are generated first, then
          rendered into messy resume text by 12 distinct templates (varying section order,
          headings, date formats, noise, fragmentation). The canonical record is the gold label.
        </p>
        <p>
          The split is <strong>template-based</strong>: training uses templates 1–8, validation
          templates 9–10, and the test set uses <strong>held-out templates 11–12</strong> that the
          model never sees during training. This measures whether the model learned the extraction
          task rather than memorizing surface formats.
        </p>
      </Card>

      <Card title="How was the benchmark designed?">
        <p>
          Both models see the exact same held-out test set with the exact same canonical prompt,
          the same deterministic decoding (greedy), the same parser, the same normalization and
          the same evaluator. Only the model state differs. Latency is measured on a warm model
          with warm-up generations excluded.
        </p>
      </Card>

      <Card title="What does schema validity mean?">
        <p>
          The share of outputs that (a) parse as JSON and (b) satisfy the strict Pydantic schema —
          correct types, no extra keys. It is the model's most basic reliability metric: a
          prediction that cannot be validated is unusable downstream no matter how good its
          content looks.
        </p>
      </Card>

      <Card title="What does unsupported-value rate mean?">
        <p>
          The share of predicted non-empty values that are not supported by the gold canonical
          record (wrong values, invented values, extra list items). It is a proxy for
          hallucination on this benchmark — <em>not</em> a universal hallucination measure.
          Because the dataset is synthetic, the canonical record is known exactly, which makes the
          comparison well-defined.
        </p>
      </Card>

      <Card title="Limitations">
        <ul>
          <li>Synthetic data has synthetic language bias; real resumes will differ.</li>
          <li>Small test set (75 examples) — estimates carry non-trivial uncertainty.</li>
          <li>Held-out templates reduce but do not eliminate template-style leakage.</li>
          <li>Results apply to this benchmark and configuration only.</li>
          <li>CPU inference of a 3–4B model is slow; GPU serving is recommended.</li>
        </ul>
      </Card>

      <Card title="Architecture">
        <pre>{`React/Vite (this UI)
   │  same-origin /api
   ▼
FastAPI backend
   ▼
Inference service (model manager, loads once)
   ▼
Base model + LoRA adapter → structured JSON

Training (offline, Google Colab GPU):
synthetic records → rendered resumes → train/val/test
   → QLoRA fine-tune → LoRA adapter → benchmark`}</pre>
      </Card>
    </div>
  )
}

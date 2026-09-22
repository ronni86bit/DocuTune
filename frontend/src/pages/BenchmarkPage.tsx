import type { BootstrapSummary, BenchmarkSummary, MetricSet } from '../types'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import { HeadlineCharts, LatencyChart, PerFieldChart } from '../components/Charts'
import { MetricCard } from '../components/MetricCard'
import { Card, EmptyState, ErrorBanner, InfoBanner, Spinner } from '../components/ui'
import { formatDelta, formatMs, formatPercent } from '../utils/format'

function CIList({ summary }: { summary: BootstrapSummary | undefined }) {
  if (!summary) return null
  const entries = Object.entries(summary) as Array<[string, { low: number; high: number }]>
  return (
    <ul className="ci-list">
      {entries.map(([key, ci]) => (
        <li key={key}>
          <strong>{key}</strong>: [{formatPercent(ci.low)}, {formatPercent(ci.high)}]
        </li>
      ))}
    </ul>
  )
}

export function BenchmarkPage() {
  const { data, loading, error } = useApi<BenchmarkSummary>(() => api.benchmarkSummary(), [])

  if (loading) return <Spinner label="Loading benchmark results…" />
  if (error) return <ErrorBanner message={error} />
  if (!data) return null

  if (!data.available) {
    return (
      <div className="page">
        <div className="page-heading">
          <h2>Benchmark Dashboard</h2>
        </div>
        <EmptyState title="Benchmark results have not been generated yet.">
          <p>
            Run the evaluation pipeline to produce real, reproducible before/after numbers:
          </p>
          <pre>{`python scripts/run_baseline.py
python -m docutune.training.train --config configs/train.yaml   # Colab GPU
python scripts/run_finetuned.py
python scripts/benchmark.py`}</pre>
          <p>
            This dashboard never shows placeholder numbers — metrics appear here only after a
            real benchmark run has written <code>results/metrics.json</code>.
          </p>
        </EmptyState>
      </div>
    )
  }

  const base = data.base as MetricSet
  const ft = data.finetuned as MetricSet

  return (
    <div className="page">
      <div className="page-heading">
        <h2>Benchmark Dashboard</h2>
        <p>
          Base model vs. fine-tuned model on the same held-out synthetic test set
          {data.metadata?.test_size ? ` (${data.metadata.test_size} examples)` : ''} — identical
          prompt, decoding, parser and evaluator.
        </p>
      </div>

      {data.partial && (
        <InfoBanner>
          {data.message ?? 'Only baseline results are available so far.'}
        </InfoBanner>
      )}

      <div className="metric-grid">
        <MetricCard
          label="Schema Validity"
          base={base.schema_validity}
          finetuned={ft?.schema_validity}
          hint="Share of outputs that parse and satisfy the resume schema."
        />
        <MetricCard
          label="Exact Match"
          base={base.exact_match}
          finetuned={ft?.exact_match}
          hint="Share of outputs exactly matching the normalized gold record."
        />
        <MetricCard
          label="Field F1 (micro)"
          base={base.field_f1}
          finetuned={ft?.field_f1}
          hint="Micro-averaged precision/recall over all extraction units."
        />
        <MetricCard
          label="Unsupported Value Rate"
          base={base.unsupported_value_rate}
          finetuned={ft?.unsupported_value_rate}
          higherIsBetter={false}
          hint="Predicted values not supported by the gold record (lower is better)."
        />
        <MetricCard
          label="JSON Validity"
          base={base.json_validity}
          finetuned={ft?.json_validity}
        />
        <MetricCard
          label="Missing Value Rate"
          base={base.missing_value_rate}
          finetuned={ft?.missing_value_rate}
          higherIsBetter={false}
        />
      </div>

      {ft && <HeadlineCharts base={base} finetuned={ft} />}

      <div className="chart-grid">
        <section className="card chart-card">
          <h3>Latency</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Statistic</th>
                <th>Base</th>
                <th>Fine-Tuned</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {(['latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms'] as const).map((key) => (
                <tr key={key}>
                  <td>{key.replace('latency_', '').replace('_ms', '')}</td>
                  <td>{formatMs(base[key])}</td>
                  <td>{ft ? formatMs(ft[key]) : '–'}</td>
                  <td>
                    {ft && base[key] !== undefined && ft[key] !== undefined
                      ? formatDelta((ft[key] as number) - (base[key] as number), false)
                      : '–'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {ft && <LatencyChart base={base} finetuned={ft} />}
        </section>

        <Card title="95% Bootstrap Confidence Intervals">
          <p className="card-subtitle">
            Percentile bootstrap over this held-out test set (deterministic seed). These
            intervals do not establish universal significance.
          </p>
          <h4>Base</h4>
          <CIList summary={data.bootstrap?.base} />
          {ft && (
            <>
              <h4>Fine-Tuned</h4>
              <CIList summary={data.bootstrap?.finetuned} />
            </>
          )}
        </Card>
      </div>

      {data.per_field && data.per_field.length > 0 && (
        <>
          <PerFieldChart rows={data.per_field} />
          <Card title="Per-Field Metrics" subtitle="Field | Base F1 | Fine-Tuned F1 | Delta">
            <table className="table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Base F1</th>
                  <th>Fine-Tuned F1</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {data.per_field.map((row) => (
                  <tr key={row.field}>
                    <td>{row.field}</td>
                    <td>{formatPercent(row.base_f1)}</td>
                    <td>{formatPercent(row.finetuned_f1)}</td>
                    <td className={(row.delta_f1 ?? 0) >= 0 ? 'ok-text' : 'bad-text'}>
                      {formatDelta(row.delta_f1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}

      {data.metadata && (
        <Card title="Run Metadata">
          <ul className="meta-list">
            <li>
              Base model: <code>{data.metadata.base_model}</code>
            </li>
            <li>Adapter: <code>{data.metadata.adapter ?? 'n/a'}</code></li>
            <li>
              Dataset v{data.metadata.dataset_version} · schema v{data.metadata.schema_version} ·
              prompt v{data.metadata.prompt_version} · evaluator v{data.metadata.evaluator_version}
            </li>
            <li>
              Device: {data.metadata.device} · GPU: {data.metadata.gpu}
            </li>
            <li>
              Decoding: {JSON.stringify(data.metadata.generation_config)} (identical for both
              models)
            </li>
            <li>Generated: {data.metadata.timestamp}</li>
          </ul>
        </Card>
      )}
    </div>
  )
}

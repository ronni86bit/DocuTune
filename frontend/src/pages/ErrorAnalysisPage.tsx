import { useMemo, useState } from 'react'
import type { ErrorRecord } from '../types'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import { Card, EmptyState, ErrorBanner, Spinner } from '../components/ui'
import { JsonViewer } from '../components/JsonViewer'

type ModelFilter = 'both' | 'base' | 'finetuned'

function RecordCard({ record, showBase, showFinetuned }: {
  record: ErrorRecord
  showBase: boolean
  showFinetuned: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="error-record">
      <button type="button" className="error-record-header" onClick={() => setExpanded(!expanded)}>
        <span className="error-record-id">{record.id}</span>
        <span className="chip">{record.difficulty}</span>
        <span className="chip">{record.template_id}</span>
        <span className={`chip ${record.base.primary_category !== 'none' ? 'chip-bad' : 'chip-ok'}`}>
          base: {record.base.primary_category}
        </span>
        <span
          className={`chip ${record.finetuned.primary_category !== 'none' ? 'chip-bad' : 'chip-ok'}`}
        >
          fine-tuned: {record.finetuned.primary_category}
        </span>
        <span className="error-expand">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="error-record-body">
          <div className="error-section">
            <h5>Source resume (synthetic)</h5>
            <pre className="error-text">{record.resume_text}</pre>
          </div>
          <div className="error-section">
            <h5>Gold JSON</h5>
            <JsonViewer data={record.gold} />
          </div>
          {showBase && (
            <div className="error-section">
              <h5>Base output — categories: {record.base.categories.join(', ') || 'none'}</h5>
              {record.base.details.length > 0 && (
                <ul className="error-details">
                  {record.base.details.map((d, i) => (
                    <li key={i}>{d}</li>
                  ))}
                </ul>
              )}
              <pre className="error-text">{record.base_output ?? '(no output)'}</pre>
            </div>
          )}
          {showFinetuned && (
            <div className="error-section">
              <h5>
                Fine-tuned output — categories: {record.finetuned.categories.join(', ') || 'none'}
              </h5>
              {record.finetuned.details.length > 0 && (
                <ul className="error-details">
                  {record.finetuned.details.map((d, i) => (
                    <li key={i}>{d}</li>
                  ))}
                </ul>
              )}
              <pre className="error-text">{record.finetuned_output ?? '(no output)'}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function ErrorAnalysisPage() {
  const { data, loading, error } = useApi(() => api.benchmarkSummary(), [])
  const [category, setCategory] = useState('all')
  const [model, setModel] = useState<ModelFilter>('both')
  const [difficulty, setDifficulty] = useState('all')

  const records = useMemo(() => data?.error_records ?? [], [data])
  const summary = useMemo(() => data?.error_summary ?? [], [data])

  const categories = useMemo(
    () => ['all', ...summary.map((row) => row.category).filter((c) => c !== 'none')],
    [summary],
  )

  const filtered = useMemo(
    () =>
      records.filter((record) => {
        if (difficulty !== 'all' && record.difficulty !== difficulty) return false
        if (category === 'all') return true
        const inBase = model !== 'finetuned' && record.base.categories.includes(category)
        const inFt = model !== 'base' && record.finetuned.categories.includes(category)
        if (model === 'both') return inBase || inFt
        return model === 'base' ? inBase : inFt
      }),
    [records, category, model, difficulty],
  )

  if (loading) return <Spinner label="Loading error analysis…" />
  if (error) return <ErrorBanner message={error} />
  if (!data?.available) {
    return (
      <div className="page">
        <div className="page-heading">
          <h2>Error Analysis</h2>
        </div>
        <EmptyState title="Benchmark results have not been generated yet.">
          <p>
            Error analysis is produced together with the benchmark by{' '}
            <code>scripts/benchmark.py</code>. Failures are categorized automatically
            (malformed JSON, schema violations, wrong values, unsupported values, …) and both
            successful and failed cases are shown.
          </p>
        </EmptyState>
      </div>
    )
  }

  const totals = data.error_totals

  return (
    <div className="page">
      <div className="page-heading">
        <h2>Error Analysis</h2>
        <p>
          Automatic failure categorization across the held-out test set — failures are analyzed,
          not hidden. A prediction can hit several categories; the primary one is shown per model.
        </p>
      </div>

      {totals && (
        <div className="metric-grid metric-grid-3">
          <div className="metric-card">
            <h4>Test Examples</h4>
            <span className="metric-number metric-solo">{totals.total_examples}</span>
          </div>
          <div className="metric-card">
            <h4>Base Model Failures</h4>
            <span className="metric-number metric-solo">
              {totals.base_failures}
              <small> / {totals.total_examples}</small>
            </span>
          </div>
          <div className="metric-card">
            <h4>Fine-Tuned Failures</h4>
            <span className="metric-number metric-solo">
              {totals.finetuned_failures}
              <small> / {totals.total_examples}</small>
            </span>
          </div>
        </div>
      )}

      <Card title="Error Distribution by Category">
        <table className="table">
          <thead>
            <tr>
              <th>Category</th>
              <th>Base</th>
              <th>Base %</th>
              <th>Fine-Tuned</th>
              <th>Fine-Tuned %</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => (
              <tr key={row.category}>
                <td>{row.category}</td>
                <td>{row.base_count}</td>
                <td>{row.base_pct.toFixed(1)}%</td>
                <td>{row.finetuned_count}</td>
                <td>{row.finetuned_pct.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card
        title="Representative Examples"
        subtitle="Selected deterministically — random (seeded), biggest improvement, regression, and both failure-swap directions. Failures included by design."
      >
        {data.representative_examples && (
          <div className="representative-grid">
            {Object.entries(data.representative_examples).map(([group, groupRecords]) =>
              groupRecords.length > 0 ? (
                <div key={group} className="representative-group">
                  <h4>{group.replace(/_/g, ' ')}</h4>
                  <ul>
                    {groupRecords.map((record) => (
                      <li key={record.id}>
                        {record.id} <span className="chip">{record.difficulty}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null,
            )}
          </div>
        )}
      </Card>

      <Card title="All Records" subtitle={`${filtered.length} record(s) match the filters`}>
        <div className="filters">
          <label>
            Category{' '}
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model{' '}
            <select value={model} onChange={(e) => setModel(e.target.value as ModelFilter)}>
              <option value="both">both</option>
              <option value="base">base</option>
              <option value="finetuned">fine-tuned</option>
            </select>
          </label>
          <label>
            Difficulty{' '}
            <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
              <option value="all">all</option>
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
            </select>
          </label>
        </div>
        <div>
          {filtered.map((record) => (
            <RecordCard
              key={record.id}
              record={record}
              showBase={model !== 'finetuned'}
              showFinetuned={model !== 'base'}
            />
          ))}
        </div>
      </Card>
    </div>
  )
}

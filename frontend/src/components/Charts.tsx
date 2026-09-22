import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MetricSet, PerFieldRow } from '../types'

const BASE_COLOR = '#94a3b8'
const FT_COLOR = '#4f46e5'

function GroupedBar({
  data,
  bars,
  layout = 'horizontal',
  percent = true,
}: {
  data: Array<Record<string, string | number>>
  bars: Array<{ key: string; name: string }>
  layout?: 'horizontal' | 'vertical'
  percent?: boolean
}) {
  const vertical = layout === 'vertical'
  const tickFormat = (v: number) => (percent ? `${(v * 100).toFixed(0)}%` : `${v}`)
  return (
    <ResponsiveContainer width="100%" height={vertical ? Math.max(260, data.length * 42) : 260}>
      <BarChart data={data} layout={layout} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        {vertical ? (
          <>
            <XAxis type="number" tickFormatter={tickFormat} domain={[0, 1]} />
            <YAxis type="category" dataKey="name" width={110} />
          </>
        ) : (
          <>
            <XAxis type="category" dataKey="name" />
            <YAxis type="number" tickFormatter={tickFormat} domain={[0, 'auto']} />
          </>
        )}
        <Tooltip
          formatter={(value: number | string) =>
            percent && typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : String(value)
          }
        />
        <Legend />
        {bars.length === 1 ? (
          <Bar dataKey={bars[0].key} name={bars[0].name} radius={4}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.name === 'Base' ? BASE_COLOR : FT_COLOR} />
            ))}
          </Bar>
        ) : (
          bars.map((bar) => (
            <Bar
              key={bar.key}
              dataKey={bar.key}
              name={bar.name}
              fill={bar.name === 'Base' ? BASE_COLOR : FT_COLOR}
              radius={4}
            />
          ))
        )}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function HeadlineCharts({
  base,
  finetuned,
}: {
  base: MetricSet
  finetuned: MetricSet
}) {
  const headline = [
    { key: 'schema_validity', label: 'Schema Validity' },
    { key: 'exact_match', label: 'Exact Match' },
    { key: 'field_f1', label: 'Field F1' },
    { key: 'unsupported_value_rate', label: 'Unsupported Value Rate' },
  ]
  return (
    <div className="chart-grid">
      {headline.map(({ key, label }) => {
        const b = base[key as keyof MetricSet]
        const f = finetuned[key as keyof MetricSet]
        if (typeof b !== 'number' || typeof f !== 'number') return null
        return (
          <section className="card chart-card" key={key}>
            <h3>{label}</h3>
            <GroupedBar
              data={[
                { name: 'Base', value: b },
                { name: 'Fine-Tuned', value: f },
              ]}
              bars={[{ key: 'value', name: label }]}
            />
          </section>
        )
      })}
    </div>
  )
}

export function LatencyChart({ base, finetuned }: { base: MetricSet; finetuned: MetricSet }) {
  const data = [
    { name: 'Mean', Base: base.latency_mean_ms, 'Fine-Tuned': finetuned.latency_mean_ms },
    { name: 'P50', Base: base.latency_p50_ms, 'Fine-Tuned': finetuned.latency_p50_ms },
    { name: 'P95', Base: base.latency_p95_ms, 'Fine-Tuned': finetuned.latency_p95_ms },
  ].filter((d) => typeof d.Base === 'number' && typeof d['Fine-Tuned'] === 'number')
  if (data.length === 0) return null
  return (
    <section className="card chart-card">
      <h3>Latency (ms, warm model)</h3>
      <GroupedBar
        data={data as Array<Record<string, string | number>>}
        bars={[
          { key: 'Base', name: 'Base' },
          { key: 'Fine-Tuned', name: 'Fine-Tuned' },
        ]}
        percent={false}
      />
    </section>
  )
}

export function PerFieldChart({ rows }: { rows: PerFieldRow[] }) {
  const data = rows
    .filter((r) => r.base_f1 !== null && r.finetuned_f1 !== null)
    .map((r) => ({ name: r.field, Base: r.base_f1 as number, 'Fine-Tuned': r.finetuned_f1 as number }))
  if (data.length === 0) return null
  return (
    <section className="card chart-card chart-card-wide">
      <h3>Per-Field F1</h3>
      <GroupedBar
        data={data}
        bars={[
          { key: 'Base', name: 'Base' },
          { key: 'Fine-Tuned', name: 'Fine-Tuned' },
        ]}
        layout="vertical"
      />
    </section>
  )
}

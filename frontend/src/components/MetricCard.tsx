import { formatDelta, formatPercent } from '../utils/format'

export function MetricCard({
  label,
  base,
  finetuned,
  higherIsBetter = true,
  hint,
}: {
  label: string
  base: number | null | undefined
  finetuned: number | null | undefined
  higherIsBetter?: boolean
  hint?: string
}) {
  const delta =
    base !== null && base !== undefined && finetuned !== null && finetuned !== undefined
      ? finetuned - base
      : null
  const good = delta === null ? null : higherIsBetter ? delta > 0 : delta < 0
  return (
    <div className="metric-card" title={hint}>
      <h4>{label}</h4>
      <div className="metric-values">
        <div className="metric-side">
          <span className="metric-model">Base</span>
          <span className="metric-number">{formatPercent(base)}</span>
        </div>
        <div className="metric-side metric-side-ft">
          <span className="metric-model">Fine-Tuned</span>
          <span className="metric-number">{formatPercent(finetuned)}</span>
        </div>
      </div>
      {delta !== null && (
        <span className={`metric-delta ${good ? 'delta-good' : 'delta-bad'}`}>
          {formatDelta(delta)} {good ? 'improvement' : 'change'}
        </span>
      )}
    </div>
  )
}

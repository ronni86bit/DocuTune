/** Formatting + comparison helpers. */

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  return `${(value * 100).toFixed(digits)}%`
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  return `${value.toFixed(0)} ms`
}

export function formatDelta(value: number | null | undefined, asPercent = true): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  const v = asPercent ? value * 100 : value
  const sign = v >= 0 ? '+' : ''
  const unit = asPercent ? 'pp' : ''
  return `${sign}${v.toFixed(1)}${unit}`
}

/** Stable JSON string with sorted keys (and sorted string arrays) for diffing. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null'
  if (Array.isArray(value)) {
    const items = value.map(canonicalJson)
    // Order-insensitive comparison for string arrays (skills etc.)
    const allStrings = value.every((v) => typeof v === 'string')
    if (allStrings) items.sort()
    return `[${items.join(',')}]`
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`)
  return `{${entries.join(',')}}`
}

export type FieldRelation =
  | 'match'
  | 'differs'
  | 'only-base'
  | 'only-finetuned'
  | 'both-empty'

export function fieldRelation(
  baseValue: unknown,
  finetunedValue: unknown,
): FieldRelation {
  const baseEmpty =
    baseValue === null ||
    baseValue === undefined ||
    baseValue === '' ||
    (Array.isArray(baseValue) && baseValue.length === 0)
  const ftEmpty =
    finetunedValue === null ||
    finetunedValue === undefined ||
    finetunedValue === '' ||
    (Array.isArray(finetunedValue) && finetunedValue.length === 0)
  if (baseEmpty && ftEmpty) return 'both-empty'
  if (baseEmpty) return 'only-finetuned'
  if (ftEmpty) return 'only-base'
  return canonicalJson(baseValue) === canonicalJson(finetunedValue) ? 'match' : 'differs'
}

export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

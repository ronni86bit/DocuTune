import type { ExtractionResult } from '../types'
import { fieldRelation } from '../utils/format'

const TOP_FIELDS = [
  'name',
  'email',
  'phone',
  'location',
  'summary',
  'skills',
  'education',
  'experience',
  'projects',
  'certifications',
] as const

function summarize(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? '' : 's'}`
  }
  if (typeof value === 'object') return '{…}'
  const text = String(value)
  return text.length > 60 ? `${text.slice(0, 57)}…` : text
}

const RELATION_LABEL: Record<string, string> = {
  match: 'match',
  differs: 'differs',
  'only-base': 'only base',
  'only-finetuned': 'only fine-tuned',
  'both-empty': 'both empty',
}

/**
 * Field-level difference view between the two model outputs.
 *
 * Note: this compares the models against EACH OTHER, not against ground
 * truth (no gold labels exist for user-provided text), so "match" means
 * "the models agree", never "correct".
 */
export function FieldDiffView({
  base,
  finetuned,
}: {
  base: ExtractionResult
  finetuned: ExtractionResult
}) {
  const baseParsed = base.json_valid && base.schema_valid ? base.parsed_output : null
  const ftParsed = finetuned.json_valid && finetuned.schema_valid ? finetuned.parsed_output : null

  const rows = TOP_FIELDS.map((field) => {
    const baseValue = baseParsed ? baseParsed[field] : null
    const ftValue = ftParsed ? ftParsed[field] : null
    return { field, relation: fieldRelation(baseValue, ftValue), baseValue, ftValue }
  })

  return (
    <section className="card diff-card">
      <h3>Field Differences (base vs fine-tuned)</h3>
      <p className="diff-note">
        This view compares the two model outputs against each other. It does not judge
        correctness — no gold labels exist for your input.
      </p>
      <table className="table diff-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Relation</th>
            <th>Base</th>
            <th>Fine-Tuned</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ field, relation, baseValue, ftValue }) => (
            <tr key={field} className={`diff-${relation}`}>
              <td className="diff-field">{field}</td>
              <td>
                <span className={`relation relation-${relation}`}>
                  {RELATION_LABEL[relation]}
                </span>
              </td>
              <td className="diff-value">{summarize(baseValue)}</td>
              <td className="diff-value">{summarize(ftValue)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

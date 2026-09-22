import type { ExtractionResult } from '../types'
import { downloadJson, formatMs } from '../utils/format'
import { JsonViewer } from './JsonViewer'
import { Badge, ValidityBadge } from './ui'

export function ResultPanel({
  result,
  title,
  accent,
}: {
  result: ExtractionResult
  title: string
  accent?: 'base' | 'finetuned'
}) {
  const parsed = result.json_valid && result.schema_valid ? result.parsed_output : null
  return (
    <section className={`result-panel panel-${accent ?? 'neutral'}`}>
      <header className="result-header">
        <h3>{title}</h3>
        <div className="result-badges">
          <Badge tone="neutral">{result.model}</Badge>
          <Badge tone="neutral">{formatMs(result.latency_ms)}</Badge>
          <ValidityBadge label="JSON" valid={result.json_valid} />
          <ValidityBadge label="Schema" valid={result.schema_valid} />
        </div>
      </header>
      {result.model_name && (
        <p className="result-model-name">
          model: <code>{result.model_name}</code>
          {result.adapter_path && (
            <>
              {' '}
              + adapter <code>{result.adapter_path}</code>
            </>
          )}
        </p>
      )}
      <div className="result-actions">
        <button
          type="button"
          className="btn btn-small"
          onClick={() => navigator.clipboard?.writeText(result.raw_output)}
        >
          Copy
        </button>
        <button
          type="button"
          className="btn btn-small"
          onClick={() => downloadJson(`${result.model}-extraction.json`, result.parsed_output ?? result.raw_output)}
        >
          Download JSON
        </button>
      </div>
      {parsed !== null ? (
        <JsonViewer data={parsed} />
      ) : (
        <div className="json-invalid">
          {!result.json_valid && (
            <p className="json-invalid-message">Model output is not valid JSON.</p>
          )}
          {result.json_valid && !result.schema_valid && (
            <p className="json-invalid-message">
              Output is JSON but does not satisfy the resume schema.
            </p>
          )}
          {result.error && <p className="json-error-detail">{result.error}</p>}
          <pre className="json-raw">{result.raw_output || '(empty output)'}</pre>
        </div>
      )}
    </section>
  )
}

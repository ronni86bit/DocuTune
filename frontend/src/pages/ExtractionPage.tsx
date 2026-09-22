import { useState } from 'react'
import { api, ApiError } from '../services/api'
import type { CompareResult, ExtractionResult, MetadataResponse, ModelMode } from '../types'
import { useApi } from '../hooks/useApi'
import { FieldDiffView } from '../components/FieldDiffView'
import { ResultPanel } from '../components/ResultPanel'
import { ErrorBanner, InfoBanner, Spinner } from '../components/ui'
import { SAMPLE_RESUME } from '../utils/sample'

type UIMode = ModelMode | 'compare'

export function ExtractionPage() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState<UIMode>('compare')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [single, setSingle] = useState<ExtractionResult | null>(null)
  const [comparison, setComparison] = useState<CompareResult | null>(null)

  const { data: metadata } = useApi<MetadataResponse>(() => api.metadata(), [])

  const run = async () => {
    if (!text.trim()) {
      setError('Please enter or load some resume text first.')
      return
    }
    setBusy(true)
    setError(null)
    setSingle(null)
    setComparison(null)
    try {
      if (mode === 'compare') {
        setComparison(await api.compare(text))
      } else {
        setSingle(await api.extract(text, mode))
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Unexpected error.')
    } finally {
      setBusy(false)
    }
  }

  const defaultMode = metadata?.default_model_mode ?? 'finetuned'

  return (
    <div className="page">
      <div className="page-heading">
        <h2>Resume Extraction</h2>
        <p>
          Turn unstructured resume text into validated structured JSON with the base model,
          the fine-tuned model, or both side by side.
        </p>
      </div>

      <InfoBanner>
        Demo environments: <strong>do not enter sensitive personal information</strong> into
        this demo. Use synthetic or anonymized text — try the sample below.
      </InfoBanner>

      <div className="extraction-layout">
        <section className="card input-card">
          <div className="input-toolbar">
            <h3>Resume Text</h3>
            <button type="button" className="btn btn-small" onClick={() => setText(SAMPLE_RESUME)}>
              Load sample
            </button>
            <button type="button" className="btn btn-small" onClick={() => setText('')}>
              Clear
            </button>
          </div>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={'Paste a resume here, or load the synthetic sample…'}
            rows={18}
            spellCheck={false}
          />
          <div className="input-controls">
            <fieldset className="mode-picker">
              <legend>Model</legend>
              {(
                [
                  ['base', 'Base Model'],
                  ['finetuned', 'Fine-Tuned Model'],
                  ['compare', 'Compare Both'],
                ] as Array<[UIMode, string]>
              ).map(([value, label]) => (
                <label key={value} className="radio">
                  <input
                    type="radio"
                    name="model-mode"
                    value={value}
                    checked={mode === value}
                    onChange={() => setMode(value)}
                  />
                  {label}
                </label>
              ))}
            </fieldset>
            <button type="button" className="btn btn-primary" onClick={run} disabled={busy}>
              {busy ? 'Extracting…' : mode === 'compare' ? 'Compare Models' : 'Extract'}
            </button>
          </div>
          {metadata && (
            <p className="model-note">
              Active base model: <code>{metadata.base_model}</code>
              {' · '}server default: <code>{defaultMode}</code>
              {' · '}adapter:{' '}
              {metadata.adapter_available ? (
                <code className="ok-text">{metadata.adapter_path}</code>
              ) : (
                <span className="warn-text">not found</span>
              )}
            </p>
          )}
        </section>

        <div className="results-area">
          {busy && <Spinner label="Running extraction (first request loads the model)…" />}
          {error && <ErrorBanner message={error} />}
          {!busy && single && (
            <ResultPanel
              result={single}
              title={single.model === 'base' ? 'Base Model' : 'Fine-Tuned Model'}
              accent={single.model === 'base' ? 'base' : 'finetuned'}
            />
          )}
          {!busy && comparison && (
            <>
              <div className="compare-grid">
                <ResultPanel result={comparison.base} title="Base Model" accent="base" />
                {comparison.finetuned ? (
                  <ResultPanel
                    result={comparison.finetuned}
                    title="Fine-Tuned Model"
                    accent="finetuned"
                  />
                ) : (
                  <section className="result-panel panel-finetuned">
                    <h3>Fine-Tuned Model</h3>
                    <InfoBanner>
                      {comparison.finetuned_error ??
                        'Fine-tuned model unavailable.'}
                    </InfoBanner>
                  </section>
                )}
              </div>
              {comparison.finetuned && (
                <FieldDiffView base={comparison.base} finetuned={comparison.finetuned} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

import { useState } from 'react'

type JsonValue = unknown

function Primitive({ value }: { value: JsonValue }) {
  if (value === null) return <span className="json-null">null</span>
  if (typeof value === 'string') return <span className="json-string">"{value}"</span>
  if (typeof value === 'number') return <span className="json-number">{String(value)}</span>
  if (typeof value === 'boolean') return <span className="json-bool">{String(value)}</span>
  return <span className="json-null">{String(value)}</span>
}

function JsonNode({ name, value, depth }: { name?: string; value: JsonValue; depth: number }) {
  const [open, setOpen] = useState(depth < 1)
  const isObject = value !== null && typeof value === 'object'
  const entries = isObject
    ? Array.isArray(value)
      ? value.map((v, i) => [String(i), v] as const)
      : Object.entries(value as Record<string, JsonValue>)
    : []

  if (!isObject) {
    return (
      <div className="json-row" style={{ paddingLeft: depth * 16 }}>
        {name !== undefined && <span className="json-key">"{name}"</span>}
        {name !== undefined && <span className="json-colon">: </span>}
        <Primitive value={value} />
      </div>
    )
  }

  const isArray = Array.isArray(value)
  return (
    <div className="json-node" style={{ paddingLeft: depth * 16 }}>
      <button type="button" className="json-toggle" onClick={() => setOpen(!open)}>
        <span className="json-caret">{open ? '▾' : '▸'}</span>
        {name !== undefined && <span className="json-key">"{name}"</span>}
        {name !== undefined && <span className="json-colon">: </span>}
        <span className="json-brace">{isArray ? '[' : '{'}</span>
        {!open && (
          <span className="json-preview">
            {entries.length} {isArray ? 'items' : 'keys'}
            <span className="json-brace">{isArray ? ']' : '}'}</span>
          </span>
        )}
      </button>
      {open && (
        <>
          {entries.map(([key, child]) => (
            <JsonNode key={key} name={key} value={child} depth={0} />
          ))}
          <div className="json-row json-close">
            <span className="json-brace">{isArray ? ']' : '}'}</span>
          </div>
        </>
      )}
    </div>
  )
}

export function JsonViewer({ data, rawOutput }: { data: unknown; rawOutput?: string }) {
  if (data === null || data === undefined) {
    return (
      <div className="json-invalid">
        <p className="json-invalid-message">Model output is not valid JSON.</p>
        {rawOutput !== undefined && <pre className="json-raw">{rawOutput}</pre>}
      </div>
    )
  }
  return (
    <div className="json-viewer">
      <JsonNode value={data} depth={0} />
    </div>
  )
}

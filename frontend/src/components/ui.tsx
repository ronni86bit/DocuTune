import type { ReactNode } from 'react'

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="spinner-wrap" role="status">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  )
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="banner banner-error" role="alert">
      <strong>Error:</strong> {message}
    </div>
  )
}

export function InfoBanner({ children }: { children: ReactNode }) {
  return <div className="banner banner-info">{children}</div>
}

type Tone = 'ok' | 'bad' | 'warn' | 'neutral'

export function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function ValidityBadge({ label, valid }: { label: string; valid: boolean }) {
  return (
    <Badge tone={valid ? 'ok' : 'bad'}>
      {label}: {valid ? 'yes' : 'no'}
    </Badge>
  )
}

export function Card({ title, subtitle, children, className = '' }: {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card ${className}`}>
      {title && (
        <header className="card-header">
          <h3>{title}</h3>
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {children}
    </div>
  )
}

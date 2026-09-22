import { NavLink } from 'react-router-dom'

const tabs = [
  { to: '/', label: 'Extraction' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/errors', label: 'Error Analysis' },
  { to: '/about', label: 'About' },
]

export function Header() {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <span className="brand-mark">DT</span>
          <div>
            <h1>DocuTune</h1>
            <p>Fine-Tuned Structured Extraction</p>
          </div>
        </div>
        <nav className="nav" aria-label="Main navigation">
          {tabs.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              end={tab.to === '/'}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}

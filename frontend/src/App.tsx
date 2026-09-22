import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from './components/Header'
import { ExtractionPage } from './pages/ExtractionPage'
import { BenchmarkPage } from './pages/BenchmarkPage'
import { ErrorAnalysisPage } from './pages/ErrorAnalysisPage'
import { AboutPage } from './pages/AboutPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Header />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<ExtractionPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/errors" element={<ErrorAnalysisPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <p>
            DocuTune — synthetic data, held-out evaluation, no fabricated numbers. Results shown
            are produced exclusively by the benchmark pipeline.
          </p>
        </footer>
      </div>
    </BrowserRouter>
  )
}

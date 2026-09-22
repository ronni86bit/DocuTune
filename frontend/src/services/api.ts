// Centralized API layer: base path, request handling, error normalization.

const API_BASE: string = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the DocuTune backend. Is it running?')
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body && typeof body.detail === 'string') detail = body.detail
    } catch {
      /* keep default detail */
    }
    throw new ApiError(response.status, detail)
  }
  return response.json() as Promise<T>
}

import type {
  CompareResult,
  ExtractionResult,
  HealthResponse,
  MetadataResponse,
  MetricsResponse,
  BenchmarkSummary,
  ModelMode,
} from '../types'

export const api = {
  health: () => request<HealthResponse>('/health'),
  metadata: () => request<MetadataResponse>('/metadata'),
  extract: (text: string, model: ModelMode) =>
    request<ExtractionResult>('/extract', {
      method: 'POST',
      body: JSON.stringify({ text, model }),
    }),
  compare: (text: string) =>
    request<CompareResult>('/compare', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
  metrics: () => request<MetricsResponse>('/metrics'),
  benchmarkSummary: () => request<BenchmarkSummary>('/benchmark/summary'),
}

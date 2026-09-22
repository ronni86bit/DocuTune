// Shared TypeScript types mirroring the FastAPI response schemas.

export type ModelMode = 'base' | 'finetuned'

export interface ParsedResume {
  name: string | null
  email: string | null
  phone: string | null
  location: string | null
  summary: string | null
  skills: string[]
  education: EducationEntry[]
  experience: ExperienceEntry[]
  projects: ProjectEntry[]
  certifications: CertificationEntry[]
  [key: string]: unknown
}

export interface EducationEntry {
  degree: string | null
  institution: string | null
  field: string | null
  start_year: number | null
  end_year: number | null
  grade: string | null
}

export interface ExperienceEntry {
  title: string | null
  company: string | null
  location: string | null
  start_date: string | null
  end_date: string | null
  current: boolean
  responsibilities: string[]
}

export interface ProjectEntry {
  name: string | null
  technologies: string[]
  description: string | null
}

export interface CertificationEntry {
  name: string | null
  issuer: string | null
  year: number | null
}

export interface ExtractionResult {
  model: string
  raw_output: string
  parsed_output: ParsedResume | null
  json_valid: boolean
  schema_valid: boolean
  latency_ms: number
  error?: string | null
  model_name?: string | null
  adapter_path?: string | null
}

export interface CompareResult {
  input: string
  base: ExtractionResult
  finetuned: ExtractionResult | null
  finetuned_error: string | null
}

export interface HealthResponse {
  status: string
  service: string
  device: string
  adapter_available: boolean
  model_mode: string
}

export interface MetadataResponse {
  base_model: string
  model_revision: string | null
  default_model_mode: string
  adapter_path: string
  adapter_available: boolean
  schema_version: string
  prompt_version: string
  evaluator_version: string
  dataset_version: string
  device: string
  quantized: boolean
  max_input_chars: number
}

export interface MetricSet {
  n_examples?: number
  json_validity?: number
  schema_validity?: number
  exact_match?: number
  field_precision?: number
  field_recall?: number
  field_f1?: number
  unsupported_value_rate?: number
  missing_value_rate?: number
  latency_mean_ms?: number
  latency_p50_ms?: number
  latency_p95_ms?: number
  output_length_mean_chars?: number
  per_field?: Record<string, FieldStats>
}

export interface FieldStats {
  tp: number
  fp: number
  fn: number
  precision: number
  recall: number
  f1: number
}

export interface PerFieldRow {
  field: string
  base_precision: number | null
  finetuned_precision: number | null
  base_recall: number | null
  finetuned_recall: number | null
  base_f1: number | null
  finetuned_f1: number | null
  delta_f1: number | null
}

export interface BootstrapCI {
  low: number
  high: number
  point: number
}

export interface BootstrapSummary {
  schema_validity?: BootstrapCI
  exact_match?: BootstrapCI
  field_f1?: BootstrapCI
  unsupported_value_rate?: BootstrapCI
}

export interface BenchmarkMetadata {
  timestamp?: string
  base_model?: string
  base_model_revision?: string | null
  adapter?: string | null
  dataset_version?: string
  schema_version?: string
  prompt_version?: string
  evaluator_version?: string
  test_size?: number
  train_size?: number
  seed?: number
  device?: string
  gpu?: string
  generation_config?: Record<string, unknown>
}

export interface MetricsResponse {
  available: boolean
  partial?: boolean
  message: string | null
  metadata?: BenchmarkMetadata
  base?: MetricSet
  finetuned?: MetricSet | null
  delta?: Record<string, number | null>
  bootstrap?: Record<string, BootstrapSummary>
}

export interface ErrorCategoryRow {
  category: string
  base_count: number
  finetuned_count: number
  base_pct: number
  finetuned_pct: number
}

export interface ErrorRecord {
  id: string
  difficulty: 'easy' | 'medium' | 'hard'
  template_id: string
  resume_text: string
  gold: ParsedResume
  base_output: string | null
  finetuned_output: string | null
  base: { categories: string[]; primary_category: string; details: string[] }
  finetuned: { categories: string[]; primary_category: string; details: string[] }
}

export interface BenchmarkSummary extends MetricsResponse {
  per_field?: PerFieldRow[] | null
  error_summary?: ErrorCategoryRow[]
  error_records?: ErrorRecord[]
  error_totals?: {
    total_examples: number
    base_failures: number
    finetuned_failures: number
  }
  representative_examples?: Record<string, ErrorRecord[]> | null
}

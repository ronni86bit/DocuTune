/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API base path; defaults to "/api" (same-origin behind Nginx / Vite proxy). */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

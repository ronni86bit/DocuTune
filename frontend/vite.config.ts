import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the frontend talks to the FastAPI backend through this proxy,
// so browser code can always use the same-origin "/api" base path (exactly like
// the production Nginx setup). Override with VITE_API_BASE if needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 300000,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})

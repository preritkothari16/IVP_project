import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Overridable so the app can run alongside another instance; defaults preserve the
// documented ports. FRONTEND_PORT must be paired with BACKEND_PORT in start_servers.py.
const FRONTEND_PORT = Number(process.env.FRONTEND_PORT || 5173)
const BACKEND_PORT = Number(process.env.BACKEND_PORT || 8000)

export default defineConfig({
  plugins: [react()],
  server: {
    port: FRONTEND_PORT,
    // fail rather than silently incrementing, so the printed URL is always the real one
    strictPort: true,
    host: true,
    proxy: {
      // keeps the browser same-origin so CORS is never exercised in dev
      '/api': {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})

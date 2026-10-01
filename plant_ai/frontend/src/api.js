const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function req(path, opts) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, opts)
  } catch {
    throw new ApiError(
      'Could not reach the server. Make sure the backend is running on port 8000.',
      0,
    )
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json()
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export const getHealth = () => req('/health')
export const getClasses = () => req('/classes')

export const predict = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  return req('/predict', { method: 'POST', body: fd })
}

export const formatBytes = (n) => {
  if (n < 1024) return `${n} B`
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1048576).toFixed(1)} MB`
}

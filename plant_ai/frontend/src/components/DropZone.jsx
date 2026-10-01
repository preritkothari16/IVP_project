import { useCallback, useRef, useState } from 'react'
import { formatBytes } from '../api'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']

export default function DropZone({ onFile, busy }) {
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const cameraRef = useRef(null)

  const validate = useCallback((file) => {
    if (!file) return 'No file selected.'
    if (!ACCEPT.includes(file.type)) {
      return `"${file.name}" is not a JPG, PNG or WebP image. Please pick a photo.`
    }
    if (file.size > MAX_BYTES) {
      return `That photo is ${formatBytes(file.size)}. The limit is 10 MB - try a smaller or
              compressed image.`
    }
    if (file.size === 0) return 'That file is empty.'
    return null
  }, [])

  const handle = useCallback(
    (file) => {
      const err = validate(file)
      setError(err)
      if (!err) onFile(file)
    },
    [validate, onFile],
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDrag(false)
      if (busy) return
      handle(e.dataTransfer.files?.[0])
    },
    [busy, handle],
  )

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!busy) setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={[
          'relative rounded-2xl border-2 border-dashed transition-all duration-200',
          drag
            ? 'border-moss-500 bg-moss-100 scale-[1.01]'
            : 'border-moss-300 bg-white/70 hover:border-moss-400 hover:bg-white',
          busy ? 'opacity-60 pointer-events-none' : '',
        ].join(' ')}
      >
        <div className="px-6 py-9 sm:py-11 text-center">
          <svg
            className="mx-auto mb-3 h-11 w-11 text-moss-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5V18a2 2 0 002 2h14a2 2 0 002-2v-1.5M3.5 8.5l3.2-3.2a2 2 0 012.8 0l1 1a2 2 0 002.8 0l1-1a2 2 0 012.8 0l3.2 3.2M3.5 8.5V18a2 2 0 002 2h13a2 2 0 002-2V8.5L17.6 4a2 2 0 00-2.8 0l-1 1a2 2 0 01-2.8 0l-1-1a2 2 0 00-2.8 0L3.5 8.5z"
            />
          </svg>

          <p className="text-base font-semibold text-moss-900">
            {drag ? 'Drop it here' : 'Drag a photo here'}
          </p>
          <p className="mt-1 text-sm text-moss-700/80">
            or choose a file &mdash; JPG, PNG or WebP, up to 10&nbsp;MB
          </p>

          <div className="mt-5 flex flex-col sm:flex-row items-center justify-center gap-2.5">
            <button
              type="button"
              className="btn-primary w-full sm:w-auto"
              onClick={() => inputRef.current?.click()}
              disabled={busy}
            >
              Choose photo
            </button>
            <button
              type="button"
              className="btn-ghost w-full sm:w-auto"
              onClick={() => cameraRef.current?.click()}
              disabled={busy}
            >
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 8.5A2 2 0 015 6.5h1.6l1-1.7A1 1 0 018.4 4h7.2a1 1 0 01.8.8l1 1.7H19a2 2 0 012 2V17a2 2 0 01-2 2H5a2 2 0 01-2-2V8.5z" />
                <circle cx="12" cy="12.5" r="3.2" />
              </svg>
              Use camera
            </button>
          </div>

          <p className="mt-4 text-xs text-moss-700/70">
            Best results: one leaf or one fruit filling the frame, in focus, in natural light.
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="sr-only"
          onChange={(e) => {
            handle(e.target.files?.[0])
            e.target.value = ''
          }}
        />
        {/* capture: opens the rear camera on phones */}
        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          onChange={(e) => {
            handle(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>

      {error && (
        <div
          role="alert"
          className="mt-3 animate-fade-up rounded-xl border border-berry/30 bg-berry/10 px-4 py-3
                     text-sm text-berry whitespace-pre-line"
        >
          {error}
        </div>
      )}
    </div>
  )
}

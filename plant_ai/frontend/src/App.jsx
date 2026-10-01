import { useCallback, useEffect, useRef, useState } from 'react'
import { getClasses, getHealth, predict, ApiError } from './api'
import DropZone from './components/DropZone'
import { ConfidenceBar, SeverityBadge } from './components/Confidence'
import DiseaseCard from './components/DiseaseCard'
import LowConfidence from './components/LowConfidence'

const pretty = (n) => n.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

function Preview({ url, name, previewOk, onClear, busy }) {
  return (
    <div className="card animate-fade-up p-4">
      <div className="flex items-start gap-4">
        {previewOk ? (
          <img
            src={url}
            alt={name ? `Uploaded: ${name}` : 'Uploaded photo'}
            className="h-28 w-28 shrink-0 rounded-xl border border-moss-200 object-cover sm:h-32 sm:w-32"
          />
        ) : (
          <div
            className="flex h-28 w-28 shrink-0 items-center justify-center rounded-xl border
                       border-dashed border-berry/50 bg-berry/5 p-2 text-center text-[11px]
                       leading-tight text-berry/80 sm:h-32 sm:w-32"
          >
            No preview
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-moss-900" title={name}>
            {name}
          </p>
          {!busy && (
            <button type="button" onClick={onClear} className="btn-ghost mt-3">
              Choose a different photo
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Loading({ url }) {
  return (
    <section className="card overflow-hidden" aria-busy="true" aria-live="polite">
      <div className="flex items-center gap-4 border-b border-moss-200/70 px-5 py-4">
        {url && (
          <img
            src={url}
            alt=""
            className="h-16 w-16 shrink-0 rounded-lg border border-moss-200 object-cover opacity-70"
          />
        )}
        <div className="flex-1 space-y-2.5">
          <div className="shimmer relative h-4 w-2/3 overflow-hidden rounded bg-moss-200" />
          <div className="shimmer relative h-2.5 w-full overflow-hidden rounded bg-moss-100" />
        </div>
      </div>
      <div className="px-5 py-5">
        <p className="flex items-center gap-2 text-sm text-moss-700">
          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v3a5 5 0 00-5 5H4z" />
          </svg>
          Analysing the photo…
        </p>
      </div>
    </section>
  )
}

function Alternatives({ top3, topClass }) {
  if (!top3?.length) return null
  return (
    <section className="card p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-moss-700/80">
        Top matches
      </h2>
      <ul className="mt-3 space-y-3">
        {top3.map((a, i) => {
          const isTop = a.class_name === topClass
          return (
            <li key={a.class_name}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className={isTop ? 'font-semibold text-moss-900' : 'text-moss-800/80'}>
                  <span className="mr-1.5 text-xs text-moss-600/70 tabular-nums">
                    {i + 1}.
                  </span>
                  {pretty(a.class_name)}
                </span>
                <span className="tabular-nums text-moss-700">{(a.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-moss-100">
                <div
                  className={`h-full rounded-full transition-[width] duration-700 ease-out ${
                    isTop ? 'bg-moss-600' : 'bg-moss-300'
                  }`}
                  style={{ width: `${Math.max(1, a.confidence * 100)}%` }}
                />
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function Header({ meta, health }) {
  return (
    <header className="mb-6">
      <div className="flex items-center gap-3">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-berry/90 text-2xl shadow-sm">
          <span aria-hidden="true">🍓</span>
        </span>
        <div>
          <h1 className="text-xl font-bold leading-tight text-moss-900 sm:text-2xl">
            Strawberry disease detector
          </h1>
          <p className="text-sm text-moss-700/85">
            Upload a leaf, flower or fruit &mdash; get a diagnosis and what to do about it
          </p>
        </div>
      </div>

      {meta && (
        <div className="mt-4 card p-4 text-sm">
          <p className="text-moss-900">
            <span className="font-semibold">Detects:</span>{' '}
            {meta.detects.map(pretty).join(', ')}.
          </p>
          <p className="mt-1.5 text-moss-900">
            <span className="font-semibold">Does not detect:</span>{' '}
            {meta.does_not_detect.join(', ')}.
          </p>
          {meta.limitations?.length > 0 && (
            <details className="group mt-3">
              <summary className="cursor-pointer list-none text-xs font-semibold text-moss-700 marker:hidden">
                <span className="inline-flex items-center gap-1.5">
                  <svg
                    className="h-3.5 w-3.5 transition-transform group-open:rotate-90"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    aria-hidden="true"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                  Known limitations of this model
                </span>
              </summary>
              <ul className="mt-2 space-y-1.5 text-xs leading-relaxed text-moss-800/90">
                {meta.limitations.map((l, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-[0.5em] h-1 w-1 shrink-0 rounded-full bg-moss-400" />
                    {l}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}

      {health && !health.model_ready && (
        <div role="alert" className="mt-4 rounded-xl border border-berry/30 bg-berry/10 px-4 py-3 text-sm text-berry">
          The backend is up but no model weights were found, so predictions are unavailable.
        </div>
      )}
    </header>
  )
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [health, setHealth] = useState(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [previewOk, setPreviewOk] = useState(false)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const resultRef = useRef(null)
  const lastRequest = useRef(0)

  useEffect(() => {
    getClasses().then(setMeta).catch(() => setMeta(null))
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview(null)
      setPreviewOk(false)
      return
    }
    // Only offer a preview if the browser can actually decode the file. A file that passes the
    // type/size check but is not real image data would otherwise render a broken-image icon.
    const url = URL.createObjectURL(file)
    const probe = new Image()
    probe.onload = () => {
      setPreview(url)
      setPreviewOk(true)
    }
    probe.onerror = () => {
      URL.revokeObjectURL(url)
      setPreview(null)
      setPreviewOk(false)
    }
    probe.src = url
    return () => URL.revokeObjectURL(url)
  }, [file])

  const run = useCallback(async (f) => {
    const ticket = ++lastRequest.current
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await predict(f)
      if (ticket !== lastRequest.current) return
      setResult(r)
      requestAnimationFrame(() =>
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      )
    } catch (e) {
      if (ticket !== lastRequest.current) return
      setError(
        e instanceof ApiError
          ? e.message
          : 'Something went wrong while analysing the photo. Please try again.',
      )
    } finally {
      if (ticket === lastRequest.current) setBusy(false)
    }
  }, [])

  const onFile = useCallback(
    (f) => {
      setFile(f)
      run(f)
    },
    [run],
  )

  const reset = useCallback(() => {
    setFile(null)
    setResult(null)
    setError(null)
    setBusy(false)
  }, [])

  return (
    <div className="mx-auto min-h-dvh max-w-3xl px-4 py-6 sm:px-6 sm:py-10">
      <Header meta={meta} health={health} />

      <DropZone onFile={onFile} busy={busy} />

      {file && !busy && (
        <div className="mt-4">
          <Preview url={preview} name={file.name} previewOk={previewOk} onClear={reset} busy={busy} />
        </div>
      )}
      {busy && (
        <div className="mt-4">
          <Loading url={previewOk ? preview : null} />
        </div>
      )}

      {error && (
        <div role="alert" className="card mt-4 border-berry/40 bg-berry/10 p-4 text-sm text-berry">
          <p className="font-semibold">Could not analyse that photo</p>
          <p className="mt-1 whitespace-pre-line">{error}</p>
          <button type="button" onClick={reset} className="btn-ghost mt-3">
            Try another photo
          </button>
        </div>
      )}

      <div ref={resultRef} className="mt-5 space-y-5">
        {result && !busy && result.low_confidence && (
          <LowConfidence result={result} previewUrl={previewOk ? preview : null} onRetry={reset} />
        )}

        {result && !busy && !result.low_confidence && (
          <>
            <section className="card animate-fade-up p-5">
              <div className="flex flex-wrap items-center gap-2.5">
                <h2 className="text-lg font-bold text-moss-900">
                  {result.disease_info?.display_name ?? pretty(result.top_class)}
                </h2>
                {result.disease_info && <SeverityBadge severity={result.disease_info.severity} />}
                {result.low_support && (
                  <span className="chip bg-clay-500/15 text-clay-700">less reliable for this disease</span>
                )}
              </div>
              <p className="mt-1 text-sm text-moss-700/85">
                Model class: <code className="text-xs">{result.top_class}</code>
              </p>
              <div className="mt-4">
                <ConfidenceBar value={result.confidence} />
              </div>
            </section>

            <Alternatives top3={result.top3} topClass={result.top_class} />
            {result.disease_info && <DiseaseCard info={result.disease_info} />}
          </>
        )}
      </div>

      <footer className="mt-10 border-t border-moss-200/70 pt-5 text-center text-xs leading-relaxed text-moss-700/80">
        <p>
          Results are AI-assisted and are <strong>not</strong> a substitute for an agronomist or
          plant-pathology diagnosis. Confirm anything important in person before treating a crop.
        </p>
        <p className="mt-1.5">
          Trained on 9 strawberry classes. It does not detect spider mites, nutrient deficiencies,
          or non-strawberry plants.
        </p>
      </footer>
    </div>
  )
}

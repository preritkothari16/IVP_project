const TIPS = [
  'Fill the frame with a single leaf or a single fruit.',
  'Use natural light — avoid harsh flash and heavy shadow.',
  'Tap to focus and hold steady so the image is sharp.',
  'Show the whole leaf or fruit, including its edges where symptoms usually start.',
  'Avoid backgrounds full of other plants, which can throw the model off.',
]

/** Shown when top-1 confidence is under the backend's 0.6 threshold. */
export default function LowConfidence({ result, previewUrl, onRetry }) {
  const alts = (result?.top3 ?? []).filter((a) => a.class_name !== result.top_class)
  return (
    <section
      role="status"
      className="card animate-scale-in border-clay-300 bg-clay-50/80 p-5 sm:p-6"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-clay-500/20 text-clay-700">
          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />
          </svg>
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-bold text-clay-800">Not confident about this one</h2>
          <p className="mt-1.5 text-sm leading-relaxed text-clay-900/90">
            The best match was only{' '}
            <strong className="tabular-nums">
              {((result?.confidence ?? 0) * 100).toFixed(0)}%
            </strong>{' '}
            confident, which is below the 60% we need to stand behind a result. This usually means the
            photo is unclear, too far away, or not a close-up of a strawberry leaf or fruit.
          </p>

          {previewUrl && (
            <img
              src={previewUrl}
              alt="The photo you uploaded"
              className="mt-3 max-h-44 rounded-lg border border-clay-300/70 object-contain"
            />
          )}

          <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-clay-800">
            Try again with
          </h3>
          <ul className="mt-2 space-y-1.5 text-sm text-clay-900/90">
            {TIPS.map((t) => (
              <li key={t} className="flex gap-2">
                <span className="mt-[0.45em] h-1.5 w-1.5 shrink-0 rounded-full bg-clay-500/70" />
                {t}
              </li>
            ))}
          </ul>

          {alts.length > 0 && (
            <div className="mt-4 rounded-lg border border-clay-300/70 bg-white/70 px-3.5 py-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-clay-800">
                What it leaned towards
              </h3>
              <ul className="mt-2 space-y-1 text-sm text-clay-900/90">
                {alts.map((a) => (
                  <li key={a.class_name} className="flex justify-between gap-3 tabular-nums">
                    <span>{pretty(a.class_name)}</span>
                    <span className="text-clay-700">
                      {(a.confidence * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button type="button" onClick={onRetry} className="btn-primary mt-4">
            Try a different photo
          </button>
        </div>
      </div>
    </section>
  )
}

function pretty(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

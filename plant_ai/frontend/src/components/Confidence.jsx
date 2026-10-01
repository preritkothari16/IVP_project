const TONE = {
  high: { chip: 'bg-berry/15 text-berry', bar: 'bg-berry', text: 'text-berry' },
  medium: { chip: 'bg-clay-500/15 text-clay-700', bar: 'bg-clay-500', text: 'text-clay-700' },
  low: { chip: 'bg-moss-200 text-moss-800', bar: 'bg-moss-500', text: 'text-moss-800' },
  none: { chip: 'bg-moss-100 text-moss-700', bar: 'bg-moss-400', text: 'text-moss-700' },
  unknown: { chip: 'bg-soil/10 text-soil', bar: 'bg-soil/40', text: 'text-soil' },
}

const LABEL = {
  high: 'High severity',
  medium: 'Medium severity',
  low: 'Low severity',
  none: 'Not a disease',
  unknown: 'Severity unknown',
}

export function SeverityBadge({ severity }) {
  const t = TONE[severity] ?? TONE.unknown
  return (
    <span className={`chip ${t.chip}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
      {LABEL[severity] ?? LABEL.unknown}
    </span>
  )
}

/** Animated confidence bar. Width is set via inline style so it can transition smoothly. */
export function ConfidenceBar({ value, threshold = 0.6 }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const low = value < threshold
  const tone = low ? 'bg-clay-500' : 'bg-moss-600'
  const textTone = low ? 'text-clay-700' : 'text-moss-700'

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-moss-700/80">
          Confidence
        </span>
        <span className={`text-lg font-bold tabular-nums ${textTone}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div
        className="relative h-2.5 w-full overflow-hidden rounded-full bg-moss-200"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Model confidence ${(value * 100).toFixed(0)} percent`}
      >
        {/* threshold marker at 0.6 */}
        <div
          className="absolute inset-y-0 w-px bg-soil/35"
          style={{ left: `${threshold * 100}%` }}
          aria-hidden="true"
        />
        <div
          className={`h-full rounded-full ${tone} transition-[width] duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1.5 text-xs text-moss-700/70">
        {low
          ? `Below the ${(threshold * 100).toFixed(0)}% reliability line — treat this as a hint, not a diagnosis.`
          : `Above the ${(threshold * 100).toFixed(0)}% reliability line.`}
      </p>
    </div>
  )
}

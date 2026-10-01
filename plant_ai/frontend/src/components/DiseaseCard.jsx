import { useCallback, useEffect, useRef, useState } from 'react'
import { SeverityBadge } from './Confidence'

const TABS = [
  { id: 'symptoms', label: 'Symptoms' },
  { id: 'cause', label: 'Cause & spread' },
  { id: 'treatment', label: 'Treatment' },
  { id: 'prevention', label: 'Prevention' },
]

function List({ items }) {
  if (!items?.length) return null
  return (
    <ul className="prose-list">
      {items.map((t, i) => (
        <li key={i}>{t}</li>
      ))}
    </ul>
  )
}

export default function DiseaseCard({ info }) {
  const [tab, setTab] = useState('symptoms')
  const navRef = useRef(null)
  const [canScrollLeft, setCanScrollLeft] = useState(false)
  const [canScrollRight, setCanScrollRight] = useState(false)

  // show the fade only while there is actually more to scroll to
  const updateOverflow = useCallback(() => {
    const el = navRef.current
    if (!el) return
    const max = el.scrollWidth - el.clientWidth
    setCanScrollLeft(el.scrollLeft > 4)
    setCanScrollRight(el.scrollLeft < max - 4)
  }, [])

  useEffect(() => {
    updateOverflow()
    const el = navRef.current
    if (!el || typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateOverflow)
      return () => window.removeEventListener('resize', updateOverflow)
    }
    const ro = new ResizeObserver(updateOverflow)
    ro.observe(el)
    return () => ro.disconnect()
  }, [updateOverflow, info])

  if (!info) return null

  return (
    <section className="card animate-fade-up overflow-hidden">
      <header className="border-b border-moss-200/70 bg-moss-50/60 px-5 py-4">
        <div className="flex flex-wrap items-center gap-2.5">
          <h2 className="text-lg font-bold text-moss-900">{info.display_name}</h2>
          <SeverityBadge severity={info.severity} />
        </div>
        <p className="mt-2 text-sm leading-relaxed text-moss-900/90">{info.summary}</p>

        {/* powdery-mildew pair: one disease, organ is informational */}
        {info.group_note && (
          <div className="mt-3 rounded-lg border border-clay-300/60 bg-clay-50 px-3.5 py-2.5">
            <p className="text-xs font-semibold text-clay-800">
              {info.group_display_name} — detected on {info.organ}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-clay-800/90">{info.group_note}</p>
          </div>
        )}
      </header>

      {/* relative wrapper so the fade edge can sit on top of the scroll container without
          touching its scroll behaviour */}
      <div className="relative">
        <nav
          ref={navRef}
          onScroll={updateOverflow}
          className="tabs-scroll flex overflow-x-auto border-b border-moss-200/70 px-2"
          role="tablist"
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={`tab whitespace-nowrap ${tab === t.id ? 'tab-active' : ''}`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        {/* fade on the right while there is more to scroll to; disappears at the end */}
        {canScrollRight && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l
                       from-white via-white/85 to-transparent"
          />
        )}
        {/* mirrored fade on the left once scrolled */}
        {canScrollLeft && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r
                       from-white via-white/85 to-transparent"
          />
        )}
      </div>

      <div className="px-5 py-4">
        {tab === 'symptoms' && <List items={info.symptoms} />}
        {tab === 'cause' && (
          <div className="space-y-3.5 text-sm leading-relaxed text-moss-900">
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-moss-700/80">
                What causes it
              </h3>
              <p>{info.cause}</p>
            </div>
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-moss-700/80">
                How it spreads
              </h3>
              <p>{info.spread}</p>
            </div>
          </div>
        )}
        {tab === 'treatment' && <List items={info.treatment} />}
        {tab === 'prevention' && <List items={info.prevention} />}
      </div>

      {info.uncertainty && (
        <details className="group border-t border-moss-200/70 bg-clay-50/50 px-5 py-3">
          <summary className="cursor-pointer list-none text-xs font-semibold text-clay-800 marker:hidden">
            <span className="inline-flex items-center gap-1.5">
              <svg className="h-3.5 w-3.5 transition-transform group-open:rotate-90" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
              How reliable is this label?
            </span>
          </summary>
          <p className="mt-2 text-xs leading-relaxed text-clay-900/90">{info.uncertainty}</p>
        </details>
      )}

      {info.verified === false && (
        <p className="border-t border-moss-200/70 px-5 py-2.5 text-[11px] leading-relaxed text-moss-700/70">
          Disease information is AI-assisted and unverified — not yet reviewed by an agronomist.
        </p>
      )}
    </section>
  )
}

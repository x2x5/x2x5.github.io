import { useState, useEffect } from 'react'

type Lang = 'zh' | 'en'

const I18N: Record<Lang, Record<string, string>> = {
  zh: {
    title: 'x2x5 Research',
    writeTitle: 'Write',
    writeDesc: '写论文的技能',
    supervisorTitle: '骆昱宇',
    supervisorDesc: '港科广助理教授',
    findTitle: 'Find',
    findDesc: '检索顶会论文',
    blogTitle: 'Blog',
    blogDesc: '经验 & 思考',
    ccfTitle: 'CCF',
    ccfDesc: 'CCF 等级查询',
    toolsLabel: '工具',
    supervisorsLabel: '导师',
    thoughtsLabel: '思考',
    selfTitle: '卡子',
    selfDesc: '野鸡大学副教授',
  },
  en: {
    title: 'x2x5 Research',
    writeTitle: 'Write',
    writeDesc: 'skills for write paper',
    supervisorTitle: 'Yuyu Luo',
    supervisorDesc: 'Professor, HKUST(GZ)',
    findTitle: 'Find',
    findDesc: 'retrieval for top papers',
    blogTitle: 'Blog',
    blogDesc: 'experience & thoughts',
    ccfTitle: 'CCF',
    ccfDesc: 'CCF recommended level',
    toolsLabel: 'Tools',
    supervisorsLabel: 'Supervisors',
    thoughtsLabel: 'Thoughts',
    selfTitle: 'Cardz',
    selfDesc: 'Professor, WCU',
  },
}

interface CardData {
  href: string
  titleKey: string
  descKey: string
  icon: React.ReactNode
  color: string
}

const tools: CardData[] = [
  {
    href: 'https://x2x5.top/find',
    titleKey: 'findTitle',
    descKey: 'findDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.3-4.3" />
      </svg>
    ),
    color: 'blue',
  },
  {
    href: 'https://x2x5.top/write',
    titleKey: 'writeTitle',
    descKey: 'writeDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
      </svg>
    ),
    color: 'purple',
  },
  {
    href: 'https://x2x5.top/ccf',
    titleKey: 'ccfTitle',
    descKey: 'ccfDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
      </svg>
    ),
    color: 'orange',
  },
]

const thoughts: CardData[] = [
  {
    href: 'https://x2x5.top/blog/',
    titleKey: 'blogTitle',
    descKey: 'blogDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
        <polyline points="14 2 14 8 20 8" />
      </svg>
    ),
    color: 'emerald',
  },
]

const supervisors: CardData[] = [
  {
    href: 'https://x2x5.top/lyy',
    titleKey: 'supervisorTitle',
    descKey: 'supervisorDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
    color: 'rose',
  },
  {
    href: 'https://x2x5.top/cardz/',
    titleKey: 'selfTitle',
    descKey: 'selfDesc',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect width="20" height="14" x="2" y="5" rx="2" />
        <line x1="2" x2="22" y1="10" y2="10" />
      </svg>
    ),
    color: 'cyan',
  },
]

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'light'
    const saved = localStorage.getItem('theme')
    if (saved === 'dark' || saved === 'light') return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light')
      }
    }
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  return { theme, setTheme }
}

function useLanguage() {
  const [lang, setLang] = useState<Lang>(() => {
    if (typeof window === 'undefined') return 'zh'
    return (localStorage.getItem('lang') as Lang) || 'zh'
  })

  useEffect(() => {
    document.documentElement.lang = lang
    localStorage.setItem('lang', lang)
  }, [lang])

  return { lang, setLang }
}

const colorMap: Record<string, string> = {
  blue: 'bg-blue-500/10 text-blue-500',
  green: 'bg-emerald-500/10 text-emerald-500',
  purple: 'bg-purple-500/10 text-purple-500',
  orange: 'bg-orange-500/10 text-orange-500',
  rose: 'bg-rose-500/10 text-rose-500',
  cyan: 'bg-cyan-500/10 text-cyan-500',
}

function Card({ data, dict }: { data: CardData; dict: Record<string, string> }) {
  const bg = colorMap[data.color] || colorMap.blue

  return (
    <a
      href={data.href}
      className="group block rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:-translate-y-[3px] hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)] min-h-[110px] no-underline text-inherit"
    >
      <div className={`w-10 h-10 rounded-xl ${bg} flex items-center justify-center mb-3 transition-transform duration-200 group-hover:scale-110`}>
        {data.icon}
      </div>
      <h2 className="text-xl font-semibold mb-1">{dict[data.titleKey]}</h2>
      <p className="text-muted text-[0.95rem] leading-relaxed m-0">{dict[data.descKey]}</p>
    </a>
  )
}

export default function App() {
  const { theme, setTheme } = useTheme()
  const { lang, setLang } = useLanguage()
  const dict = I18N[lang]

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[960px] mx-auto">
        <header className="flex items-center justify-between gap-3 mb-2.5">
          <h1 className="text-[clamp(1.7rem,3vw,2.2rem)] font-bold m-0">
            {dict.title}
          </h1>
          <div className="flex items-center gap-2 flex-shrink-0">
            <a
              href="./readme.html"
              className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 no-underline transition-colors hover:text-text hover:border-accent"
              aria-label="README"
              title="README"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>
            </a>
            <a
              href="./commits.html"
              className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 no-underline transition-colors hover:text-text hover:border-accent"
              aria-label="Commit 历史"
              title="Commit 历史"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </a>
            <button
              type="button"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 cursor-pointer transition-colors hover:text-text hover:border-accent"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
              )}
            </button>
            <button
              type="button"
              onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')}
              className="rounded-full border px-3 py-1.5 text-sm cursor-pointer transition-all bg-accent border-accent text-white hover:opacity-90"
            >
              {lang === 'zh' ? 'EN' : '中'}
            </button>
          </div>
        </header>

        {/* 工具 */}
        <p className="text-[0.85rem] text-muted tracking-widest leading-none mt-7 mb-2.5">
          {dict.toolsLabel}
        </p>
        <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {tools.map((t) => (
            <Card key={t.titleKey} data={t} dict={dict} />
          ))}
        </section>

        {/* 导师 */}
        <p className="text-[0.85rem] text-muted tracking-widest leading-none mt-10 mb-2.5">
          {dict.supervisorsLabel}
        </p>
        <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {supervisors.map((s) => (
            <Card key={s.titleKey} data={s} dict={dict} />
          ))}
        </section>

        {/* 思考 */}
        <p className="text-[0.85rem] text-muted tracking-widest leading-none mt-10 mb-2.5">
          {dict.thoughtsLabel}
        </p>
        <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {thoughts.map((t) => (
            <Card key={t.titleKey} data={t} dict={dict} />
          ))}
        </section>
      </main>
    </div>
  )
}

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
    selfTitle: 'Cardz',
    selfDesc: 'Professor, WCU',
  },
}

interface CardData {
  href: string
  titleKey: string
  descKey: string
}

const tools: CardData[] = [
  { href: 'https://x2x5.top/find', titleKey: 'findTitle', descKey: 'findDesc' },
  { href: 'https://x2x5.top/write', titleKey: 'writeTitle', descKey: 'writeDesc' },
  { href: 'https://x2x5.top/blog/', titleKey: 'blogTitle', descKey: 'blogDesc' },
  { href: 'https://x2x5.top/ccf', titleKey: 'ccfTitle', descKey: 'ccfDesc' },
]

const supervisors: CardData[] = [
  { href: 'https://x2x5.top/lyy', titleKey: 'supervisorTitle', descKey: 'supervisorDesc' },
  { href: 'https://x2x5.top/cardz/', titleKey: 'selfTitle', descKey: 'selfDesc' },
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

function Card({ data, dict }: { data: CardData; dict: Record<string, string> }) {
  return (
    <a
      href={data.href}
      className="group block rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:-translate-y-[3px] hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)] min-h-[110px] no-underline text-inherit"
    >
      <h2 className="text-xl font-semibold mb-2">{dict[data.titleKey]}</h2>
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
              onClick={() => setLang('zh')}
              className={`rounded-full border px-3 py-1.5 text-sm cursor-pointer transition-all ${
                lang === 'zh'
                  ? 'bg-accent border-accent text-white'
                  : 'bg-card-bg border-border text-muted hover:text-text hover:border-accent'
              }`}
            >
              中文
            </button>
            <button
              type="button"
              onClick={() => setLang('en')}
              className={`rounded-full border px-3 py-1.5 text-sm cursor-pointer transition-all ${
                lang === 'en'
                  ? 'bg-accent border-accent text-white'
                  : 'bg-card-bg border-border text-muted hover:text-text hover:border-accent'
              }`}
            >
              EN
            </button>
          </div>
        </header>

        {/* Desktop: left-right layout; Mobile: stacked */}
        <div className="flex flex-col lg:flex-row gap-0 items-start">
          <div className="flex-1 min-w-0 w-full">
            <p className="text-[0.85rem] text-muted tracking-widest leading-none mt-7 mb-2.5">
              {dict.toolsLabel}
            </p>
            <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {tools.map((t) => (
                <Card key={t.titleKey} data={t} dict={dict} />
              ))}
            </section>
          </div>

          <div className="hidden lg:block w-px self-stretch bg-border mx-6 flex-shrink-0" />

          <div className="w-full lg:w-[240px] flex-shrink-0 mt-6 lg:mt-0">
            <p className="text-[0.85rem] text-muted tracking-widest leading-none mt-7 mb-2.5">
              {dict.supervisorsLabel}
            </p>
            <section className="grid gap-4 grid-cols-1">
              {supervisors.map((s) => (
                <Card key={s.titleKey} data={s} dict={dict} />
              ))}
            </section>
          </div>
        </div>
      </main>
    </div>
  )
}

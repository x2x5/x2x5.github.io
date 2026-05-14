import { useState, useEffect } from 'react'

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

const colorMap: Record<string, string> = {
  blue: 'bg-blue-500 text-white',
  purple: 'bg-purple-500 text-white',
  emerald: 'bg-emerald-500 text-white',
  orange: 'bg-orange-500 text-white',
  rose: 'bg-rose-500 text-white',
  cyan: 'bg-cyan-500 text-white',
  amber: 'bg-amber-500 text-white',
  indigo: 'bg-indigo-500 text-white',
}

interface DayCommit {
  date: string
  color: string
  items: string[]
}

const dayCommits: DayCommit[] = [
  {
    date: '2026-05-15',
    color: 'blue',
    items: [
      '1. 使用 React + Tailwind CSS + TypeScript 重写整个网站',
      '2. 添加系统主题检测和手动切换功能',
      '3. 实现响应式布局适配手机、平板和电脑',
    ],
  },
  {
    date: '2026-05-12',
    color: 'purple',
    items: [
      '1. 修复中英文环境下卡片高度不一致的问题',
      '2. 新增 CCF 等级查询卡片',
      '3. 调整导师区域为左右分栏布局',
    ],
  },
  {
    date: '2026-05-07',
    color: 'emerald',
    items: [
      '1. 将个人卡片标题改为"卡子 / Cardz"',
      '2. 更新个人卡片链接地址',
    ],
  },
  {
    date: '2026-05-05',
    color: 'orange',
    items: [
      '1. 将导师区域独立为单独区块',
      '2. 新增个人介绍卡片',
      '3. 更新各卡片描述文案',
    ],
  },
  {
    date: '2026-05-04',
    color: 'rose',
    items: [
      '1. 新增博客卡片并支持中英文切换',
    ],
  },
  {
    date: '2026-05-02',
    color: 'cyan',
    items: [
      '1. 更新导师卡片链接到 lyy 仓库',
      '2. 优化 Find 卡片的英文描述',
      '3. 移除 Think 卡片并更新导师卡片文案',
    ],
  },
  {
    date: '2026-05-01',
    color: 'amber',
    items: [
      '1. 更新主页 Think 和导师卡片',
      '2. 添加中英文双语 README',
      '3. 重新排列主页卡片顺序',
    ],
  },
  {
    date: '2026-04-30',
    color: 'indigo',
    items: [
      '1. 更新 README 以匹配当前主页内容',
      '2. 更新主页文案、布局和语言切换功能',
    ],
  },
  {
    date: '2026-04-15',
    color: 'blue',
    items: [
      '1. 使用三个论文工具卡片重新设计主页',
    ],
  },
  {
    date: '2026-04-11',
    color: 'purple',
    items: [
      '1. 创建简单的 x2x5 主页',
      '2. 配置自定义域名 CNAME',
    ],
  },
]

function DayCard({ data }: { data: DayCommit }) {
  const bg = colorMap[data.color] || colorMap.blue
  return (
    <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:-translate-y-[3px] hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)]">
      <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold mb-3 ${bg}`}>
        {data.date}
      </div>
      <div className="space-y-2">
        {data.items.map((item, i) => (
          <p key={i} className="text-sm text-muted leading-relaxed">
            {item}
          </p>
        ))}
      </div>
    </div>
  )
}

export default function CommitsPage() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[960px] mx-auto">
        {/* 顶部 */}
        <div className="flex items-center justify-between mb-8">
          <a
            href="./"
            className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 no-underline transition-colors hover:text-text hover:border-accent"
            aria-label="返回主页"
            title="返回主页"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
          </a>
          <button
            type="button"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 cursor-pointer transition-colors hover:text-text hover:border-accent"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
            )}
          </button>
        </div>

        {/* 标题 */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 text-accent mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2H2v10h10V2z" /><path d="M12 12H2v10h10V12z" /><path d="M22 2h-10v10h10V2z" /><path d="M22 12h-10v10h10V12z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold m-0">Commit 历史</h1>
          <p className="text-muted mt-2">网站迭代记录</p>
        </div>

        {/* 卡片网格 */}
        <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {dayCommits.map((d) => (
            <DayCard key={d.date} data={d} />
          ))}
        </section>
      </main>
    </div>
  )
}

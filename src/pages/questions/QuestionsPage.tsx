import { useTheme } from '../../hooks/useTheme'

const colorMap: Record<string, string> = {
  blue: 'bg-blue-500/10 text-blue-500',
  purple: 'bg-purple-500/10 text-purple-500',
  emerald: 'bg-emerald-500/10 text-emerald-500',
  orange: 'bg-orange-500/10 text-orange-500',
  rose: 'bg-rose-500/10 text-rose-500',
  cyan: 'bg-cyan-500/10 text-cyan-500',
  violet: 'bg-violet-500/10 text-violet-500',
  amber: 'bg-amber-500/10 text-amber-500',
}

interface QuestionMeta {
  id: string
  title: string
  desc: string
  time: string
  icon: React.ReactNode
  color: string
}

const questions: QuestionMeta[] = [
  {
    id: 'dev-error-behavior',
    title: '改代码时语法出错，正在跑的网页会崩吗？',
    desc: 'Vite 热更新的容错机制',
    time: '26/05/15 13:01:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" />
        <polyline points="8 6 2 12 8 18" />
      </svg>
    ),
    color: 'blue',
  },
  {
    id: 'why-html-files',
    title: '写了 React 为什么还要手写 HTML？',
    desc: '多页 SPA 的入口机制',
    time: '26/05/15 12:59:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" x2="8" y1="13" y2="13" />
        <line x1="16" x2="8" y1="17" y2="17" />
        <line x1="10" x2="8" y1="9" y2="9" />
      </svg>
    ),
    color: 'emerald',
  },
  {
    id: 'no-router',
    title: '什么叫「无路由」？',
    desc: '多页 SPA 的架构选择',
    time: '26/05/15 12:57:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
        <path d="M8 8 2 2" />
      </svg>
    ),
    color: 'rose',
  },
  {
    id: 'no-idea',
    title: '想不到提什么问题怎么办？',
    desc: '从 0 到 1 生成一个好问题',
    time: '25/05/15 05:30:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <path d="M12 17h.01" />
      </svg>
    ),
    color: 'violet',
  },
  {
    id: 'channel-collection',
    title: '怎么及时汇总和更新各种渠道？',
    desc: '搭建你的渠道雷达',
    time: '25/05/15 05:38:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4.9 16.1C1 12.2 1 5.8 4.9 1.9" />
        <path d="M7.8 13.2c-2.3-2.3-2.3-6.1 0-8.5" />
        <path d="M16.2 4.7c2.3 2.3 2.3 6.1 0 8.5" />
        <path d="M19.1 1.9c3.9 3.9 3.9 10.3 0 14.2" />
        <circle cx="12" cy="16" r="2" />
        <path d="M12 18v4" />
      </svg>
    ),
    color: 'blue',
  },
  {
    id: 'blog-vs-ai',
    title: '博客 / Markdown 笔记过时了吗？',
    desc: '从文档苦力到 AI 对话式创作',
    time: '25/05/15 05:47:00',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
    color: 'amber',
  },
]

export default function QuestionsPage() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[960px] mx-auto">
        <div className="relative flex items-center justify-center mb-8">
          <div className="absolute left-0">
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
          </div>
          <h1 className="text-2xl font-bold m-0 flex items-center gap-2">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-violet-500">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <path d="M12 17h.01" />
            </svg>
            提问
          </h1>
          <div className="absolute right-0">
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
        </div>

        <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          {questions.map((q) => {
            const bg = colorMap[q.color] || colorMap.violet
            return (
              <a
                key={q.id}
                href={`answer.html?id=${q.id}`}
                className="group flex flex-col rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:-translate-y-[3px] hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)] no-underline text-inherit"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className={`w-10 h-10 rounded-xl ${bg} flex items-center justify-center transition-transform duration-200 group-hover:scale-110`}>
                    {q.icon}
                  </div>
                  <span className="text-[0.65rem] text-muted/40 flex-shrink-0 ml-2">{q.time}</span>
                </div>
                <h2 className="text-xl font-semibold mb-1">{q.title}</h2>
                <p className="text-muted text-[0.95rem] leading-relaxed m-0">{q.desc}</p>
              </a>
            )
          })}
        </section>
      </main>
    </div>
  )
}

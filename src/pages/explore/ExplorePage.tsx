import { useTheme } from '../../hooks/useTheme'

interface ArticleMeta {
  id: string
  title: string
  summary: string
  date: string
  month: string
  emoji: string
  color: string
}

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

const articles: ArticleMeta[] = [
  {
    id: 'on-policy-distillation',
    title: 'On Policy Distillation',
    summary: '最近注意到一篇关于 on-policy distillation 的文章。简单来说，就是把 policy distillation 和 on-policy 学习结合起来——让 student 模型在探索环境的过程中实时向 teacher 学习，而不是从固定的离线数据集中学。',
    date: '05/15',
    month: '2026年5月',
    emoji: '🔄',
    color: 'blue',
  },
]

function ArticleRow({ article }: { article: ArticleMeta }) {
  const bg = colorMap[article.color] || colorMap.blue

  return (
    <a
      href={`article.html?id=${article.id}`}
      className="block rounded-2xl border border-border bg-card-bg p-4 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:-translate-y-[1px] hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)] no-underline text-inherit"
    >
      <div className="flex gap-4">
        <div className={`w-20 h-20 rounded-xl ${bg} flex items-center justify-center text-2xl flex-shrink-0`}>
          {article.emoji}
        </div>
        <div className="flex flex-col min-w-0 flex-1">
          <h3 className="font-semibold text-text">{article.title}</h3>
          <p className="text-sm text-muted leading-relaxed mt-1 line-clamp-2">{article.summary}</p>
          <div className="mt-auto pt-2 text-right">
            <span className="text-[0.65rem] text-muted/40">{article.date}</span>
          </div>
        </div>
      </div>
    </a>
  )
}

export default function ExplorePage() {
  const { theme, setTheme } = useTheme()

  const grouped = articles.reduce<Record<string, ArticleMeta[]>>((acc, a) => {
    if (!acc[a.month]) acc[a.month] = []
    acc[a.month].push(a)
    return acc
  }, {})

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[720px] mx-auto">
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
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-cyan-500">
              <circle cx="12" cy="12" r="10" />
              <path d="M16.24 7.76l-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z" />
            </svg>
            研究
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

        {Object.entries(grouped).map(([month, items]) => (
          <div key={month} className="mb-8">
            <h2 className="text-sm font-semibold text-muted mb-3 tracking-wider">{month}</h2>
            <div className="space-y-3">
              {items.map((a) => (
                <ArticleRow key={a.id} article={a} />
              ))}
            </div>
          </div>
        ))}
      </main>
    </div>
  )
}

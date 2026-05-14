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

function Terminal({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-[#1a1f2b] overflow-hidden mt-3">
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-white/10">
        <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
        <div className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
        <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
        <span className="text-xs text-white/40 ml-2 font-mono">终端</span>
      </div>
      <div className="p-4 font-mono text-sm text-white/90 leading-relaxed">
        {children}
      </div>
    </div>
  )
}

function FileTag({ name }: { name: string }) {
  return (
    <code className="inline-block bg-accent/10 text-accent px-2 py-0.5 rounded-md text-xs font-mono border border-accent/20">
      {name}
    </code>
  )
}

function ArrowDown() {
  return (
    <div className="flex justify-center my-2">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted animate-bounce">
        <path d="M12 5v14" />
        <path d="m19 12-7 7-7-7" />
      </svg>
    </div>
  )
}

const colorMap: Record<string, string> = {
  blue: 'bg-blue-500 text-white',
  purple: 'bg-purple-500 text-white',
  emerald: 'bg-emerald-500 text-white',
  orange: 'bg-orange-500 text-white',
  rose: 'bg-rose-500 text-white',
  cyan: 'bg-cyan-500 text-white',
  amber: 'bg-amber-500 text-white',
}

interface StepProps {
  num: string
  title: string
  desc: string
  color: string
  icon: React.ReactNode
  children?: React.ReactNode
}

function Step({ num, title, desc, color, icon, children }: StepProps) {
  const bg = colorMap[color] || colorMap.blue
  return (
    <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)] transition-all duration-200 hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)]">
      <div className="flex items-start gap-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-xl ${bg} flex items-center justify-center text-lg font-bold`}>
          {num}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-muted">{icon}</span>
            <h3 className="text-lg font-semibold text-text">{title}</h3>
          </div>
          <p className="text-sm text-muted leading-relaxed">{desc}</p>
          {children}
        </div>
      </div>
    </div>
  )
}

function Tip({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 p-3">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-500 flex-shrink-0 mt-0.5">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" x2="12" y1="8" y2="12" />
        <line x1="12" x2="12.01" y1="16" y2="16" />
      </svg>
      <span className="text-sm text-amber-700 dark:text-amber-300">{children}</span>
    </div>
  )
}

export default function ReadmePage() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[720px] mx-auto">
        {/* 顶部 */}
        <div className="flex items-center justify-between mb-6">
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

        {/* 标题区 */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 text-accent mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" x2="8" y1="13" y2="13" />
              <line x1="16" x2="8" y1="17" y2="17" />
              <line x1="10" x2="8" y1="9" y2="9" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold m-0">给人看的 README</h1>
          <p className="text-muted mt-2">5 分钟学会修改这个网站</p>
        </div>

        {/* 快速开始 */}
        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)] mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold">快速开始</h2>
              <p className="text-sm text-muted">先把项目跑起来，看到效果再说</p>
            </div>
          </div>
          <Terminal>
            <div className="text-white/50"># 第一次用：安装依赖</div>
            <div className="text-green-400">npm install</div>
            <div className="mt-2 text-white/50"># 启动本地预览（会自动打开浏览器）</div>
            <div className="text-green-400">npm run dev</div>
            <div className="mt-2 text-white/50"># 然后浏览器访问 http://localhost:5173/</div>
          </Terminal>
          <Tip>改代码后页面会自动刷新，不需要手动刷新浏览器。</Tip>
        </div>

        <ArrowDown />

        {/* Step 1 */}
        <Step
          num="1"
          title="改文字内容"
          desc="网站上的所有文字，都在 App.tsx 的一个字典里。找到它，改就是了。"
          color="blue"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
          }
        >
          <div className="mt-3 text-sm text-muted">
            打开 <FileTag name="src/App.tsx" />，找到下面这段代码：
          </div>
          <Terminal>
            <div><span className="text-purple-400">const</span> <span className="text-blue-400">I18N</span> = {'{'}</div>
            <div className="pl-4 text-white/50">// 中文版</div>
            <div className="pl-4">zh: {'{'}</div>
            <div className="pl-8 text-yellow-300">findTitle: <span className="text-green-300">&quot;Find&quot;</span>,</div>
            <div className="pl-8 text-yellow-300">findDesc: <span className="text-green-300">&quot;检索顶会论文&quot;</span>,</div>
            <div className="pl-4">{'}'}</div>
          </Terminal>
          <Tip>中英文都有，改 zh 是改中文页面，改 en 是改英文页面。</Tip>
        </Step>

        <ArrowDown />

        {/* Step 2 */}
        <Step
          num="2"
          title="改卡片链接"
          desc="每个卡片点进去跳转到哪，也是在这里改。"
          color="purple"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
            </svg>
          }
        >
          <div className="mt-3 text-sm text-muted">
            还在 <FileTag name="src/App.tsx" /> 里，往下找：
          </div>
          <Terminal>
            <div><span className="text-purple-400">const</span> <span className="text-blue-400">tools</span> = [</div>
            <div className="pl-4">{'{'}</div>
            <div className="pl-8 text-yellow-300">href: <span className="text-green-300">&quot;https://x2x5.top/find&quot;</span>,</div>
            <div className="pl-8 text-white/50">// 改成你想要的链接</div>
            <div className="pl-4">{'}'}</div>
          </Terminal>
        </Step>

        <ArrowDown />

        {/* Step 3 */}
        <Step
          num="3"
          title="新增卡片"
          desc="想在网站上加一个新入口？复制一段，改一改就行。"
          color="emerald"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="20" height="14" x="2" y="5" rx="2" />
              <line x1="2" x2="22" y1="10" y2="10" />
            </svg>
          }
        >
          <div className="mt-3 text-sm text-muted">
            在 <FileTag name="src/App.tsx" /> 的 <code className="text-accent font-mono text-xs">tools</code> 或 <code className="text-accent font-mono text-xs">supervisors</code> 数组里，复制一段：
          </div>
          <Terminal>
            <div className="pl-4">{'{'}</div>
            <div className="pl-8 text-yellow-300">href: <span className="text-green-300">&quot;https://你的链接&quot;</span>,</div>
            <div className="pl-8 text-yellow-300">titleKey: <span className="text-green-300">&quot;你的标题&quot;</span>,</div>
            <div className="pl-8 text-yellow-300">descKey: <span className="text-green-300">&quot;你的描述&quot;</span>,</div>
            <div className="pl-8 text-white/50">// 然后在 I18N 字典里加上对应的文字</div>
            <div className="pl-4">{'}'}</div>
          </Terminal>
          <Tip>卡片图标和颜色也可以换，蓝色 blue、绿色 emerald、紫色 purple、橙色 orange、红色 rose、青色 cyan。</Tip>
        </Step>

        <ArrowDown />

        {/* Step 4 */}
        <Step
          num="4"
          title="改颜色"
          desc="整个网站的主题色、背景色、暗黑模式，都在一个文件里。"
          color="orange"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="13.5" cy="6.5" r="2.5" />
              <path d="M13.5 9.5c-2.5 0-4.5 2-4.5 4.5v4h9v-4c0-2.5-2-4.5-4.5-4.5Z" />
              <path d="M9 18v1a3 3 0 0 0 6 0v-1" />
            </svg>
          }
        >
          <div className="mt-3 text-sm text-muted">
            打开 <FileTag name="src/index.css" />，修改这些颜色值：
          </div>
          <Terminal>
            <div className="text-white/50">/* 亮色模式 */</div>
            <div><span className="text-yellow-300">--color-bg</span>: <span className="text-green-300">#f6f7fb</span>;</div>
            <div><span className="text-yellow-300">--color-accent</span>: <span className="text-green-300">#365fcf</span>;</div>
            <div className="mt-2 text-white/50">/* 暗黑模式 */</div>
            <div>html.dark {'{'}</div>
            <div className="pl-4"><span className="text-yellow-300">--color-bg</span>: <span className="text-green-300">#11141b</span>;</div>
            <div className="pl-4">{'}'}</div>
          </Terminal>
        </Step>

        <ArrowDown />

        {/* Step 5 */}
        <Step
          num="5"
          title="构建并发布"
          desc="改完了？推到 GitHub，自动就上线了。"
          color="rose"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14" />
              <path d="m12 5 7 7-7 7" />
            </svg>
          }
        >
          <div className="mt-3 text-sm text-muted">
            三条命令，搞定：
          </div>
          <Terminal>
            <div className="text-white/50"># 1. 把改动存到 git</div>
            <div className="text-green-400">git add -A</div>
            <div className="text-green-400">git commit -m <span className="text-green-300">&quot;更新网站&quot;</span></div>
            <div className="mt-2 text-white/50"># 2. 推送到 GitHub</div>
            <div className="text-green-400">git push origin main</div>
            <div className="mt-2 text-white/50"># 3. 等 1-2 分钟，刷新 https://x2x5.github.io/</div>
          </Terminal>
          <Tip>推送后 GitHub Actions 会自动构建部署，不需要手动操作。</Tip>
        </Step>

        {/* 常见问题 */}
        <div className="mt-10 mb-10">
          <p className="text-[0.85rem] text-muted tracking-widest leading-none mb-3">常见问题</p>
          <div className="space-y-3">
            <div className="rounded-xl border border-border bg-card-bg p-4 shadow-[0_10px_30px_var(--color-card-shadow)]">
              <h4 className="font-semibold text-text mb-1">页面一片空白？</h4>
              <p className="text-sm text-muted">检查浏览器控制台（F12 → Console），如果报错 `'application/octet-stream' is not a valid JavaScript MIME type`，说明 GitHub Pages 部署的是源码而不是构建产物。去仓库 Settings → Pages → Source 确认是 GitHub Actions。</p>
            </div>
            <div className="rounded-xl border border-border bg-card-bg p-4 shadow-[0_10px_30px_var(--color-card-shadow)]">
              <h4 className="font-semibold text-text mb-1">本地预览正常，线上样式不对？</h4>
              <p className="text-sm text-muted">可能是浏览器缓存。按 Ctrl+Shift+R（Mac 是 Cmd+Shift+R）强制刷新，或者等几分钟再试。</p>
            </div>
            <div className="rounded-xl border border-border bg-card-bg p-4 shadow-[0_10px_30px_var(--color-card-shadow)]">
              <h4 className="font-semibold text-text mb-1">npm install 报错？</h4>
              <p className="text-sm text-muted">确保安装了 Node.js 18 或更高版本。命令行输入 <code className="text-accent font-mono text-xs">node -v</code> 查看版本。</p>
            </div>
          </div>
        </div>

        {/* 技术栈 */}
        <div className="mt-10 mb-10">
          <p className="text-[0.85rem] text-muted tracking-widest leading-none mb-3">技术栈</p>
          <div className="flex flex-wrap gap-2">
            {['React 19', 'TypeScript', 'Vite', 'Tailwind CSS v4', 'GitHub Actions'].map((tech) => (
              <span key={tech} className="inline-flex items-center rounded-full border border-border bg-card-bg px-3 py-1 text-sm text-muted">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}

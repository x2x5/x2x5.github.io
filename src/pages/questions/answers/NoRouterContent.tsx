import { useState, useEffect } from 'react'

export default function NoRouterContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <style>{`
        @keyframes flow-left {
          0% { transform: translateX(0); }
          100% { transform: translateX(-8px); }
        }
        .flow { animation: flow-left 1.5s ease-in-out infinite alternate; }
      `}</style>

      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-rose-500/10 text-rose-500 mb-5">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            <path d="M8 8 2 2" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          什么叫「无路由」？
        </h1>
        <p className="text-muted mt-2 text-base">多页 SPA：不装路由器，也能导航</p>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-rose-500" />
          <h2 className="text-lg font-semibold text-text m-0">🧭 先理解：路由是什么？</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] mb-4">
          <p className="text-sm text-text leading-relaxed">
            在网站开发里，<strong>路由（router）</strong>就是"根据 URL 显示不同内容"的机制。
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-rose-500/10 text-rose-500 flex items-center justify-center text-sm">🏛️</div>
              <h3 className="font-semibold text-text text-sm">传统网站</h3>
            </div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">index.html</div>
              <div className="text-center text-rose-400">⬇️ 点链接</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">about.html</div>
              <div className="text-center text-rose-400">⬇️ 点链接</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">contact.html</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-rose-500/10 text-rose-500 rounded-full px-2 py-0.5 text-xs font-medium">每个页面 = 独立的 .html 文件</span>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center text-sm">🏗️</div>
              <h3 className="font-semibold text-text text-sm">现代 SPA</h3>
            </div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">index.html（只有一个）</div>
              <div className="text-center text-emerald-400">⬇️ JS 接管</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">路由器匹配 URL → 渲染组件</div>
              <div className="text-center text-emerald-400">⬇️ 无刷新切换</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">地址栏变了，页面没重新加载</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-emerald-500/10 text-emerald-500 rounded-full px-2 py-0.5 text-xs font-medium">需要 react-router 之类的库</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔀 那这个项目呢？</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-8">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-blue-500/10 text-blue-500 flex items-center justify-center text-2xl mx-auto mb-2 flow">🏠</div>
              <div className="text-xs font-medium text-blue-500">index.html</div>
              <div className="text-[0.6rem] text-muted">主页</div>
            </div>
            <div className="text-2xl text-muted hidden sm:block">→</div>
            <div className="text-2xl text-muted sm:hidden rotate-90">→</div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-violet-500/10 text-violet-500 flex items-center justify-center text-2xl mx-auto mb-2 flow">❓</div>
              <div className="text-xs font-medium text-violet-500">questions.html</div>
              <div className="text-[0.6rem] text-muted">提问列表</div>
            </div>
            <div className="text-2xl text-muted hidden sm:block">→</div>
            <div className="text-2xl text-muted sm:hidden rotate-90">→</div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center text-2xl mx-auto mb-2 flow">🧭</div>
              <div className="text-xs font-medium text-emerald-500">explore.html</div>
              <div className="text-[0.6rem] text-muted">探索列表</div>
            </div>
          </div>
          <div className="text-center mt-6 pt-4 border-t border-border">
            <span className="text-sm text-muted">
              每个页面是<strong className="text-text">独立的 HTML 文件</strong>，用 <code className="text-accent font-mono text-xs">&lt;a href&gt;</code> 链接跳转
            </span>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">⚖️ 两种方式对比</h2>
        </div>
        <p className="text-sm text-muted mb-4">没有绝对的好坏，只有适不适合</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-sm font-bold text-emerald-500 mb-2">✅ 有路由（react-router）</div>
            <ul className="space-y-1.5 text-xs text-muted list-disc list-inside">
              <li>页面切换无刷新，体验流畅</li>
              <li>适合大型复杂应用</li>
              <li>需要额外库 + 配置</li>
              <li>对静态托管不友好（需要 fallback）</li>
            </ul>
          </div>
          <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
            <div className="text-sm font-bold text-blue-500 mb-2">✅ 无路由（多页 SPA）</div>
            <ul className="space-y-1.5 text-xs text-muted list-disc list-inside">
              <li>零依赖，零配置</li>
              <li>对 GitHub Pages 等静态托管完美</li>
              <li>每页独立加载，JS 按需分离</li>
              <li>页面切换会刷新，但对内容站不是问题</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">💡 为什么这个项目选无路由？</h2>
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-lg">📄</div>
            <div>
              <h4 className="font-semibold text-text text-sm">页面数量少且固定</h4>
              <p className="text-xs text-muted mt-0.5">总共就几个页面，不需要复杂的动态路由。每个页面单独写一个 HTML + 一个 React 组件，简单直接。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-lg">🚀</div>
            <div>
              <h4 className="font-semibold text-text text-sm">GitHub Pages 原生支持</h4>
              <p className="text-xs text-muted mt-0.5">多 HTML 文件直接部署，不需要配置 404 fallback。如果用了 react-router，还要加额外处理才能让 GH Pages 正常工作。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-lg">🎯</div>
            <div>
              <h4 className="font-semibold text-text text-sm">对 AI 更友好</h4>
              <p className="text-xs text-muted mt-0.5">每页一个独立 URL，AI 可以直接访问和引用具体页面，不需要理解路由逻辑。README 里写 `answer.html?id=xxx`，AI 就知道怎么用。</p>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-rose-500/5 to-blue-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🔑</div>
        <p className="text-text font-medium">
          「无路由」不是"没有导航"，而是"用浏览器自带的导航"。
        </p>
        <p className="text-sm text-muted mt-1">
          这个项目的每个页面你都能直接访问，不需要 JavaScript 帮你路由——回归 Web 最原始但也最可靠的方式。
        </p>
      </div>
    </>
  )
}

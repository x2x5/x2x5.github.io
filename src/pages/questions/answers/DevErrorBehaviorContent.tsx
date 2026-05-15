import { useState, useEffect } from 'react'

export default function DevErrorBehaviorContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-blue-500/10 text-blue-500 mb-5">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          改代码时语法出错，正在跑的网页会崩吗？
        </h1>
        <p className="text-muted mt-2 text-base">Vite 热更新的容错机制</p>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🏃 首先：npm run dev 在跑什么？</h2>
        </div>
        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <p className="text-sm text-text leading-relaxed">
            你在终端跑 <code className="text-accent font-mono text-xs">npm run dev</code>，实际启动的是 <strong>Vite 开发服务器</strong>。它的核心能力叫 <strong>HMR</strong>（Hot Module Replacement，热模块替换）。
          </p>
          <div className="mt-4 rounded-xl bg-blue-500/5 border border-blue-500/10 p-3">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-blue-500 font-medium">HMR 的作用：</span>
              <span className="text-muted">你改一个文件，Vite 只重新编译那一个文件，然后无缝替换到浏览器里。</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">⚠️ 如果写了一半，语法错误怎么办？</h2>
        </div>

        <div className="space-y-4">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-lg">📝</span>
              <span className="text-sm font-semibold text-text">场景</span>
            </div>
            <p className="text-xs text-muted">你正在改 <code className="text-accent font-mono text-xs">ChannelContent.tsx</code>，代码写到一半还没写完，语法不对。Vite 检测到文件变化，尝试重新编译。</p>
          </div>

          <div className="flex justify-center text-muted text-lg">↓</div>

          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-lg">🛑</span>
              <span className="text-sm font-semibold text-text">Vite 编译失败</span>
            </div>
            <p className="text-xs text-muted">Vite 发现 <code className="text-accent font-mono text-xs">ChannelContent.tsx</code> 有语法错误，编译中断。</p>
          </div>

          <div className="flex justify-center text-muted text-lg">↓</div>

          <div className="rounded-xl border border-border bg-card-bg p-4">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-lg">🖥️</span>
              <span className="text-sm font-semibold text-text">浏览器出现错误遮罩</span>
            </div>
            <p className="text-xs text-muted">Vite 在浏览器窗口显示一个半透明的错误提示层（Error Overlay），告诉你哪个文件、哪一行出了什么错。但<strong>其他页面完全不受影响</strong>。</p>
          </div>

          <div className="flex justify-center text-muted text-lg">↓</div>

          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-lg">🔧</span>
              <span className="text-sm font-semibold text-text">改好 → 自动恢复</span>
            </div>
            <p className="text-xs text-muted">你继续改完代码，保存。Vite 自动重新编译，成功了就自动替换，错误遮罩消失。不需要手动刷新浏览器。</p>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🧩 为什么其他页面不受影响？</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
              <div className="text-sm font-bold text-rose-500 mb-3">❌ 你的担心</div>
              <div className="flex flex-col items-center text-xs text-muted">
                <div className="rounded-lg bg-rose-500/10 p-2 w-full text-center">一个文件写错</div>
                <span className="text-rose-400">↓</span>
                <div className="rounded-lg bg-rose-500/10 p-2 w-full text-center">整个网站崩溃</div>
                <span className="text-rose-400">↓</span>
                <div className="rounded-lg bg-rose-500/10 p-2 w-full text-center">所有页面都打不开</div>
              </div>
            </div>
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
              <div className="text-sm font-bold text-emerald-500 mb-3">✅ 实际情况</div>
              <div className="flex flex-col items-center text-xs text-muted">
                <div className="rounded-lg bg-emerald-500/10 p-2 w-full text-center">一个文件写错</div>
                <span className="text-emerald-400">↓</span>
                <div className="rounded-lg bg-emerald-500/10 p-2 w-full text-center">仅这个模块编译失败</div>
                <span className="text-emerald-400">↓</span>
                <div className="rounded-lg bg-emerald-500/10 p-2 w-full text-center">其他页面照常运行</div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-xs text-muted text-center">
              Vite 的 HMR 是<strong>模块级别的</strong>。每个 <code className="text-accent font-mono text-xs">.tsx</code> 文件是一个独立模块。<br />
              一个模块编译失败，不影响其他模块。你甚至可以打开<strong>其他页面正常浏览</strong>。
            </p>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">📋 具体场景举例</h2>
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-lg">1</div>
            <div>
              <h4 className="font-semibold text-text text-sm">你在改提问页的回答</h4>
              <p className="text-xs text-muted mt-0.5">修改 <code className="text-accent">answers/ChannelContent.tsx</code> 时写错语法。</p>
              <p className="text-xs text-muted mt-0.5"><span className="text-emerald-500 font-medium">结果：</span>只有 <code className="text-accent">answer.html?id=channel-collection</code> 这个页面出现错误遮罩。首页、提问列表、其他回答页面全都可以正常访问。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-lg">2</div>
            <div>
              <h4 className="font-semibold text-text text-sm">你在改首页 App.tsx</h4>
              <p className="text-xs text-muted mt-0.5">修改卡片配置时写错了。</p>
              <p className="text-xs text-muted mt-0.5"><span className="text-emerald-500 font-medium">结果：</span>只有首页（<code className="text-accent">index.html</code>）报错。其他如 <code className="text-accent">questions.html</code>、<code className="text-accent">explore.html</code> 都正常。因为每个 HTML 是独立入口。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-lg">3</div>
            <div>
              <h4 className="font-semibold text-text text-sm">全局文件出错了</h4>
              <p className="text-xs text-muted mt-0.5">比如改了 <code className="text-accent">hooks/useTheme.ts</code> 或 <code className="text-accent">index.css</code>。</p>
              <p className="text-xs text-muted mt-0.5"><span className="text-emerald-500 font-medium">结果：</span>所有引用了这个文件的页面都会报错。但这种情况很少发生——这类全局文件一般不需要频繁修改。</p>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🛡️</div>
        <p className="text-text font-medium">
          一个文件写错了，<strong>只影响那一个文件</strong>。<br />其他页面该干嘛干嘛，浏览器也不用刷新。
        </p>
        <p className="text-sm text-muted mt-1">
          这就是 HMR 的好处——改代码像换零件，不是拆房子。
        </p>
      </div>
    </>
  )
}

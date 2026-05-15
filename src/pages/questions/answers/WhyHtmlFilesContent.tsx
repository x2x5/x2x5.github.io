import { useState, useEffect } from 'react'

export default function WhyHtmlFilesContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-emerald-500/10 text-emerald-500 mb-5">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" x2="8" y1="13" y2="13" />
            <line x1="16" x2="8" y1="17" y2="17" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          写了 React 为什么还要手写 HTML？
        </h1>
        <p className="text-muted mt-2 text-base">每个 .html 文件就是一扇门</p>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🤔 你的直觉是对的</h2>
        </div>
        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <p className="text-sm text-text leading-relaxed">
            在一个<strong>标准 React SPA</strong> 里，你确实只需要一个 <code className="text-accent font-mono text-xs">index.html</code>——里面就一个 <code className="text-accent font-mono text-xs">&lt;div id="root"&gt;&lt;/div&gt;</code> 加一个 <code className="text-accent font-mono text-xs">&lt;script&gt;</code>，剩下的全是 <code className="text-accent font-mono text-xs">.tsx</code>。
          </p>
          <p className="text-sm text-text leading-relaxed mt-3">
            Vite 编译时会自动把所有 JSX 打包成 JS，注入到这个 HTML 里。你<strong>不需要手动管理 HTML</strong>。
          </p>
          <div className="mt-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 p-3">
            <div className="text-xs font-medium text-emerald-500 mb-1">✅ 标准 SPA 只需要一个 index.html</div>
            <div className="text-xs text-muted font-mono">React 组件都在 .tsx 里，HTML 只是个空壳</div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔄 但这里不一样：多页 SPA</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <p className="text-sm text-text leading-relaxed">
            这个项目<strong>没有路由库</strong>，每个页面是独立的 HTML 入口。所以每加一个页面，就需要<strong>手写一个 .html 文件</strong>作为 Vite 的入口。
          </p>
          <p className="text-sm text-text leading-relaxed mt-2">
            但别担心——这些 HTML 很轻，每个就十几行，结构完全一样：
          </p>
          <div className="mt-4 rounded-xl bg-[#1a1f2b] p-4 font-mono text-xs text-white/90 leading-relaxed overflow-x-auto">
            <div className="text-white/40">&lt;!doctype html&gt;</div>
            <div className="text-white/40">&lt;html lang="zh"&gt;</div>
            <div className="text-white/40">  &lt;head&gt;</div>
            <div className="pl-4 text-white/40">... 主题检测脚本 ...</div>
            <div className="text-white/40">  &lt;/head&gt;</div>
            <div className="text-white/40">  &lt;body&gt;</div>
            <div className="text-emerald-400">    &lt;div id="root"&gt;&lt;/div&gt;</div>
            <div className="text-emerald-400">    &lt;script src="/src/page.tsx"&gt;&lt;/script&gt;</div>
            <div className="text-white/40">  &lt;/body&gt;</div>
            <div className="text-white/40">&lt;/html&gt;</div>
          </div>
          <p className="text-xs text-muted mt-3">高亮的两行是唯一有信息量的——其他都是模板</p>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔗 一条完整链路</h2>
        </div>
        <p className="text-sm text-muted mb-4">从手写 HTML 到屏幕上渲染，Vite 做了什么</p>

        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-lg font-bold">1</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-text">手写 .html</span>
                <span className="text-xs text-muted font-mono">questions.html</span>
              </div>
              <p className="text-xs text-muted mt-0.5">十几行的架子，只有 <code className="text-accent">&lt;div id="root"&gt;</code> 和 <code className="text-accent">&lt;script&gt;</code> 是关键</p>
            </div>
          </div>
          <div className="flex justify-center text-muted text-lg">↓</div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-lg font-bold">2</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-text">Vite 编译</span>
                <span className="text-xs text-muted font-mono">.tsx → .js</span>
              </div>
              <p className="text-xs text-muted mt-0.5">Vite 把 TSX 编译成 JS，把 <code className="text-accent">questions.tsx</code> 打包，注入到 <code className="text-accent">questions.html</code></p>
            </div>
          </div>
          <div className="flex justify-center text-muted text-lg">↓</div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-lg font-bold">3</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-text">输出到 dist/</span>
                <span className="text-xs text-muted font-mono">questions.html + JS + CSS</span>
              </div>
              <p className="text-xs text-muted mt-0.5">Vite 生成最终的 HTML（自动注入 <code className="text-accent">&lt;link&gt;</code> 和 <code className="text-accent">&lt;script&gt;</code>），你手写的 HTML 是它的起点</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">📊 一句话总结</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="text-sm font-bold text-rose-500 mb-2">❌ 你以为的</div>
            <div className="flex flex-col items-center gap-2 text-xs text-muted">
              <div className="rounded-lg bg-rose-500/10 p-2 w-full text-center">写 React → HTML 自动生成</div>
              <span className="text-rose-400">↓</span>
              <div className="rounded-lg bg-rose-500/10 p-2 w-full text-center">不需要手动碰 HTML</div>
            </div>
            <p className="text-xs text-muted mt-3 text-center">对标准 SPA 成立，但这里不是标准 SPA</p>
          </div>
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-sm font-bold text-emerald-500 mb-2">✅ 实际的</div>
            <div className="flex flex-col items-center gap-2 text-xs text-muted">
              <div className="rounded-lg bg-emerald-500/10 p-2 w-full text-center">手写入口 HTML（每页一个）</div>
              <span className="text-emerald-400">↓</span>
              <div className="rounded-lg bg-emerald-500/10 p-2 w-full text-center">Vite 编译 + 注入 JS/CSS</div>
            </div>
            <p className="text-xs text-muted mt-3 text-center">HTML 是"门"，React 组件是"房间里面的东西"</p>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-purple-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔀 那适不适合改成有路由的？</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] mb-4">
          <p className="text-sm text-text leading-relaxed">
            <strong>技术上可以</strong>——装个 react-router，所有页面都跑在 <code className="text-accent font-mono text-xs">index.html</code> 里。<br />
            但问题是：<strong>GitHub Pages 不认识路由</strong>。
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="text-sm font-bold text-rose-500 mb-3">⚠️ 如果用路由</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">用户直接访问 <code>explore.html</code></div>
              <div className="text-center text-rose-400">↓</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">GitHub Pages 找不到这个文件</div>
              <div className="text-center text-rose-400">↓</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">返回 404</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-rose-500/10 text-rose-500 rounded-full px-2 py-0.5 text-xs font-medium">需要额外 hack 才能解决</span>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-sm font-bold text-emerald-500 mb-3">✅ 现在这样（多页 SPA）</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">用户直接访问 <code>explore.html</code></div>
              <div className="text-center text-emerald-400">↓</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">GitHub Pages 找到真实文件</div>
              <div className="text-center text-emerald-400">↓</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">正常返回页面 ✅</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-emerald-500/10 text-emerald-500 rounded-full px-2 py-0.5 text-xs font-medium">零配置，原生支持</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-xl bg-purple-500/5 border border-purple-500/10 p-4">
          <div className="flex items-start gap-2">
            <span className="text-purple-500 flex-shrink-0 text-lg">💡</span>
            <div className="text-xs text-muted">
              <strong className="text-text">结论：</strong>改成路由不仅没有明显好处，反而会让部署变复杂。目前的方案对 GitHub Pages 最友好，也最简单。不值得改。
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-cyan-500" />
          <h2 className="text-lg font-semibold text-text m-0">📌 什么叫"新增一个页面"？</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] mb-4">
          <p className="text-sm text-text leading-relaxed">
            在"多页 SPA"的语境下，<strong>"新增一个页面"</strong> = 新增一个可以直接在浏览器地址栏里输入的 URL。比如你加一个 <code className="text-accent font-mono text-xs">gallery.html</code>，别人可以直接访问 <code className="text-accent font-mono text-xs">你的网站/gallery.html</code>。
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-sm font-bold text-emerald-500 mb-3">✅ 这不算新增页面</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">在提问列表加一张新卡片</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">链接到 <code>answer.html?id=new</code></div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">在探索列表加一篇新文章</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">链接到 <code>article.html?id=new</code></div>
              <div className="text-center mt-2">
                <span className="inline-block bg-emerald-500/10 text-emerald-500 rounded-full px-2 py-0.5 text-xs font-medium">都是在已有页面上加内容</span>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
            <div className="text-sm font-bold text-amber-500 mb-3">⚠️ 这算新增页面</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-amber-500/10 p-2.5 text-center">加一个 <code>about.html</code></div>
              <div className="rounded-lg bg-amber-500/10 p-2.5 text-center">加一个 <code>projects.html</code></div>
              <div className="rounded-lg bg-amber-500/10 p-2.5 text-center">加一个 <code>gallery.html</code></div>
              <div className="text-muted text-center text-xs mt-1">（需要手写 .html + 注册 vite.config.ts）</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-amber-500/10 text-amber-500 rounded-full px-2 py-0.5 text-xs font-medium">全新的独立入口</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-xl bg-cyan-500/5 border border-cyan-500/10 p-4">
          <div className="flex items-start gap-2">
            <span className="text-cyan-500 flex-shrink-0 text-lg">🔑</span>
            <div className="text-xs text-muted">
              <strong className="text-text">简单判断：</strong>如果新内容能塞进已有的 URL 模式（比如 <code className="text-accent font-mono text-xs">answer.html?id=xxx</code>），就不是新页面，只是新内容。只有当你想要一个<strong>全新的 URL 前缀</strong>时，才需要新 .html 文件。
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🧭 用「探索」举个例子</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">1</div>
              <div className="text-sm text-text">
                你探索了 <strong>On Policy Distillation</strong>
              </div>
            </div>
            <div className="ml-11 space-y-2 text-xs text-muted">
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">→</span>
                探索列表多了一行：<code className="text-accent font-mono text-xs">ExplorePage.tsx</code> 的 <code className="text-accent font-mono text-xs">articles</code> 数组加一条
              </div>
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">→</span>
                点进去看到文章内容：<code className="text-accent font-mono text-xs">articles/OnPolicyDistillationContent.tsx</code> 写内容
              </div>
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">→</span>
                URL 是 <code className="text-accent font-mono text-xs">article.html?id=on-policy-distillation</code>
              </div>
              <div className="mt-2 rounded-lg bg-emerald-500/5 border border-emerald-500/10 p-2.5">
                <span className="font-medium">不需要新 .html 文件。</span>因为 <code className="text-accent font-mono text-xs">article.html</code> 早就存在了。
              </div>
            </div>

            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">2</div>
                <div className="text-sm text-text">
                  你又探索了 <strong>某新东西 B</strong>
                </div>
              </div>
              <div className="ml-11 mt-2 space-y-2 text-xs text-muted">
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">→</span>
                  完全一样：加数组 + 加内容文件
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">→</span>
                  URL 是 <code className="text-accent font-mono text-xs">article.html?id=b</code>
                </div>
                <div className="mt-2 rounded-lg bg-amber-500/5 border border-amber-500/10 p-2.5">
                  <span className="font-medium">同样不需要新 .html 文件。</span>因为 <code className="text-accent font-mono text-xs">article.html</code> 可以处理任意 <code className="text-accent font-mono text-xs">?id=</code>。
                </div>
              </div>
            </div>

            <div className="border-t border-border pt-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">3</div>
                <div className="text-sm text-text">
                  你新增了一个叫 <strong>「作品集」</strong> 的顶级栏目
                </div>
              </div>
              <div className="ml-11 mt-2 space-y-2 text-xs text-muted">
                <div className="flex items-center gap-2">
                  <span className="text-rose-500">→</span>
                  URL 是 <code className="text-accent font-mono text-xs">gallery.html</code> ——全新的前缀
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-rose-500">→</span>
                  这才是需要 <strong>手写新 .html</strong> 的情况
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10 p-3">
          <div className="flex items-start gap-2">
            <span className="text-emerald-500 flex-shrink-0 text-lg">✅</span>
            <div className="text-xs text-muted">
              所以回到你的问题：探索 A、探索 B、探索 C……<strong>都不需要新 .html</strong>。<br />
              你只需要加内容文件、加数组条目、注册路由——这三个步骤，已经全部在 <code className="text-accent font-mono text-xs">src/pages/explore/</code> 目录里完成。
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-orange-500" />
          <h2 className="text-lg font-semibold text-text m-0">🤯 但等等——"两个不都是新网页吗？"</h2>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] mb-4">
          <p className="text-sm text-text leading-relaxed">
            你说得对。从<strong>用户视角</strong>看，<strong>列表页和详情页都是新网页</strong>。你点击一个链接，浏览器跳转到一个新 URL，看到新内容——谁说这不是新网页？
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="text-sm font-bold text-rose-500 mb-2">🧑 你的视角</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">主页卡片「随便」</div>
              <div className="text-center text-rose-400">↓ 点击</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">suibian.html ← 新网页 ✅</div>
              <div className="text-center text-rose-400">↓ 再点一个卡片</div>
              <div className="rounded-lg bg-rose-500/10 p-2.5 text-center">suibian-detail.html?id=xxx ← 也是新网页 ✅</div>
              <div className="text-center mt-2">
                <span className="inline-block bg-rose-500/10 text-rose-500 rounded-full px-2 py-0.5 text-xs font-medium">用户看来，两个都是新页面</span>
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-sm font-bold text-emerald-500 mb-2">⚙️ 架构视角</div>
            <div className="space-y-2 text-xs text-muted">
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">HTML 是入口文件</div>
              <div className="text-center text-emerald-400">↓</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">一个问题：「有没有现成的入口？」</div>
              <div className="text-center text-emerald-400">↓</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">有 → 加个 ?id= 就行 ✅</div>
              <div className="rounded-lg bg-emerald-500/10 p-2.5 text-center">没有 → 需要写新 HTML 🏗️</div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <p className="text-sm font-semibold text-text mb-3">用「随便」举个例子</p>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-orange-500/10 text-orange-500 flex items-center justify-center flex-shrink-0 text-sm">1</div>
              <div className="text-sm text-text">
                主页加卡片「随便」，想链接到一个列表页
              </div>
            </div>
            <div className="ml-11 text-xs text-muted">
              这个列表页的 URL 是什么？如果叫 <code className="text-accent">suibian.html</code>——<strong>这个 URL 前缀不存在</strong>，所以需要新建 <code className="text-accent">suibian.html</code> + <code className="text-accent">suibian.tsx</code> + 注册 <code className="text-accent">vite.config.ts</code>。
            </div>

            <div className="border-t border-border pt-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-sm">2</div>
                <div className="text-sm text-text">
                  列表页里有个卡片，点进去看详情
                </div>
              </div>
              <div className="ml-11 mt-2 text-xs text-muted">
                这个详情页的 URL 可以设计成 <code className="text-accent">article.html?id=suibian-xxx</code>。<br />
                <code className="text-accent">article.html</code> 已经存在 → 不需要新 HTML，加内容文件就行。
              </div>
            </div>

            <div className="border-t border-border pt-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-sm">3</div>
                <div className="text-sm text-text">
                  但你也可以选择新建一个详情页入口
                </div>
              </div>
              <div className="ml-11 mt-2 text-xs text-muted">
                如果你想让 URL 变成 <code className="text-accent">suibian-detail.html?id=xxx</code> 而不是 <code className="text-accent">article.html?id=xxx</code>——那也需要新 HTML。这是<strong>设计选择</strong>，不是强制要求。
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-xl bg-orange-500/5 border border-orange-500/10 p-4">
          <div className="flex items-start gap-2">
            <span className="text-orange-500 flex-shrink-0 text-lg">📞</span>
            <div className="text-xs text-muted">
              <strong className="text-text">一个比喻：</strong>.html 文件就像<strong>电话号码前缀</strong>（比如 010），<code className="text-accent font-mono text-xs">?id=</code> 就像<strong>分机号</strong>（比如 1234）。<br />
              你要开一条新线路 → 需要新前缀（新 .html）。<br />
              你只是在现有线路上加个分机 → 不需要新前缀（不需要新 .html）。<br />
              但 <code className="text-accent font-mono text-xs">010-1234</code> 和 <code className="text-accent font-mono text-xs">021-5678</code> 打出去都是电话——用户不在乎前缀还是分机，反正打通了就是新电话。
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-emerald-500/5 to-blue-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🚪</div>
        <p className="text-text font-medium">
          <strong>HTML = 门，TSX = 房间里的家具。</strong><br />多页 SPA 需要多扇门，所以多写了几个 HTML。
        </p>
        <p className="text-sm text-muted mt-1">
          但每扇门就几行，写一次后面都是复制粘贴改个名字的事。
        </p>
      </div>
    </>
  )
}

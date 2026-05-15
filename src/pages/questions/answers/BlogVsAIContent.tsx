import { useState, useEffect } from 'react'

function PainCard({ emoji, title, desc }: { emoji: string; title: string; desc: string }) {
  return (
    <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 flex items-start gap-3">
      <div className="w-10 h-10 rounded-xl bg-rose-500/10 text-rose-500 flex items-center justify-center flex-shrink-0 text-xl">
        {emoji}
      </div>
      <div className="min-w-0">
        <h4 className="font-semibold text-text text-sm">{title}</h4>
        <p className="text-xs text-muted mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

function FlowStep({ emoji, label, sub, last = false }: { emoji: string; label: string; sub: string; last?: boolean }) {
  return (
    <div className="flex flex-col items-center">
      <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center text-2xl">
        {emoji}
      </div>
      <span className="text-sm font-semibold text-text mt-2">{label}</span>
      <span className="text-xs text-muted text-center mt-0.5">{sub}</span>
      {!last && <div className="text-muted text-lg mt-1">↓</div>}
    </div>
  )
}

function CompareCard({ dim, oldWay, ai, color }: { dim: string; oldWay: string; ai: string; color: string }) {
  const [open, setOpen] = useState(false)
  const dot = `bg-${color}-500`

  return (
    <div
      className={`rounded-xl border ${open ? `border-${color}-500/30` : 'border-border'} bg-card-bg p-4 cursor-pointer transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)] select-none`}
      onClick={() => setOpen(!open)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(!open) } }}
    >
      <div className="flex items-center gap-3 mb-2">
        <div className={`w-5 h-5 rounded-md ${dot} flex items-center justify-center text-white text-xs font-bold transition-transform duration-200 ${open ? 'rotate-45' : ''}`}>
          +
        </div>
        <h4 className="font-semibold text-text text-sm">{dim}</h4>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-lg bg-rose-500/5 border border-rose-500/10 p-2.5">
          <span className="text-rose-500 font-bold block mb-0.5">📝 Markdown</span>
          <span className="text-muted">{oldWay}</span>
        </div>
        <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/10 p-2.5">
          <span className="text-emerald-500 font-bold block mb-0.5">✨ AI + HTML</span>
          <span className="text-muted">{ai}</span>
        </div>
      </div>
      <div className={`overflow-hidden transition-all duration-300 ease-in-out ${open ? 'max-h-40 opacity-100 mt-3' : 'max-h-0 opacity-0'}`}>
        <div className="pt-3 border-t border-border text-xs text-muted leading-relaxed">
          {color === 'rose' && '写一篇博客：构思 30% + 排版 30% + 实际内容 40%。一半时间花在"怎么呈现"而不是"呈现什么"。'}
          {color === 'emerald' && 'Markdown 只有黑白文字和代码块。AI 生成的 HTML 可以带布局、颜色、动画、交互——信息密度和传达效率完全不是一个级别。'}
          {color === 'blue' && '改一个 Markdown 段落可能导致整个文档结构崩塌。AI 生成的新页面独立、自洽，改哪句直接重说，不需要动其他地方。'}
          {color === 'amber' && 'Markdown 要求你同时想"写什么"和"怎么排版"。AI 对话只需要你想"问什么"，剩下的 AI 搞定。认知负担降低 80%。'}
        </div>
      </div>
    </div>
  )
}

export default function BlogVsAIContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <style>{`
        @keyframes slide-in-left {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slide-in-right {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .slide-left { animation: slide-in-left 0.6s ease-out forwards; }
        .slide-right { animation: slide-in-right 0.6s ease-out forwards; }
      `}</style>

      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-amber-500/10 text-amber-500 mb-5">
          <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          博客 / Markdown 笔记过时了吗？
        </h1>
        <p className="text-muted mt-2 text-base">从文档苦力到 AI 对话式创作</p>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">⚡ 一目了然：两种创作方式</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className={`rounded-2xl border border-rose-500/20 bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)] ${mounted ? 'slide-left' : 'opacity-0'}`}>
            <div className="w-12 h-12 rounded-xl bg-rose-500/10 text-rose-500 flex items-center justify-center mb-3 text-2xl">📝</div>
            <h3 className="font-bold text-text mb-2">旧方式：Markdown 苦力</h3>
            <div className="space-y-2 text-sm text-muted">
              <div className="flex items-center gap-2"><span className="text-rose-500">✗</span>写一篇文章 = 内容 + 排版 + 格式</div>
              <div className="flex items-center gap-2"><span className="text-rose-500">✗</span>想改结构？重写大半篇</div>
              <div className="flex items-center gap-2"><span className="text-rose-500">✗</span>视觉单调，信息密度低</div>
              <div className="flex items-center gap-2"><span className="text-rose-500">✗</span>维护成本随着篇数线性增长</div>
            </div>
            <div className="mt-4 pt-3 border-t border-rose-500/10 text-xs text-rose-500 font-medium">
              你花 80% 的时间在伺候格式，20% 在传递思想
            </div>
          </div>

          <div className={`rounded-2xl border border-emerald-500/20 bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)] ${mounted ? 'slide-right' : 'opacity-0'}`}>
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center mb-3 text-2xl">✨</div>
            <h3 className="font-bold text-text mb-2">新方式：AI 对话式创作</h3>
            <div className="space-y-2 text-sm text-muted">
              <div className="flex items-center gap-2"><span className="text-emerald-500">✓</span>说话即可创作，AI 生成 HTML</div>
              <div className="flex items-center gap-2"><span className="text-emerald-500">✓</span>不满意就重说，AI 生成新页面</div>
              <div className="flex items-center gap-2"><span className="text-emerald-500">✓</span>每个页面独立、自洽、视觉丰富</div>
              <div className="flex items-center gap-2"><span className="text-emerald-500">✓</span>知识自然生长成网络，而非线性文档</div>
            </div>
            <div className="mt-4 pt-3 border-t border-emerald-500/10 text-xs text-emerald-500 font-medium">
              你花 100% 的时间在思考，剩下的 AI 搞定
            </div>
          </div>
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-rose-500" />
          <h2 className="text-lg font-semibold text-text m-0">😫 旧时代的痛</h2>
        </div>
        <p className="text-sm text-muted mb-4">你之所以头疼，不是因为笨，是因为工具不对</p>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-3">
          <PainCard emoji="📝" title="格式地狱" desc="Markdown 的缩进、转义、渲染差异——花一半时间调格式，而不是想内容。" />
          <PainCard emoji="🔄" title="反复修改" desc="想调整一个段落的结构？可能导致整篇文章崩塌。改着改着就放弃了。" />
          <PainCard emoji="🧠" title="认知超载" desc="同时想内容、组织结构、控制排版——大脑缓存溢出，写作变成受刑。" />
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">✨ 新时代的爽</h2>
        </div>
        <p className="text-sm text-muted mb-4">你只需要做一件事：提问。剩下的交给 AI</p>

        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-0">
            <FlowStep emoji="🤔" label="有个想法" sub="甚至只是一个模糊的念头" />
            <FlowStep emoji="💬" label="跟 AI 说" sub="用自然语言描述需求" />
            <FlowStep emoji="🎨" label="生成 HTML" sub="AI 输出精美信息页" />
            <FlowStep emoji="🔄" label="不满意？重说" sub="针对不懂的部分再提问" />
            <FlowStep emoji="🌳" label="知识生长" sub="新页面不断衍生，形成网络" last />
          </div>
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">📊 四个维度对比</h2>
        </div>
        <p className="text-sm text-muted mb-4">点击卡片展开详细说明</p>
        <div className="space-y-3">
          <CompareCard dim="⏱ 创作速度" oldWay="写完一篇以小时计" ai="生成一页以秒计" color="rose" />
          <CompareCard dim="🎨 视觉表现" oldWay="黑白文字 + 代码块" ai="布局、颜色、动画全自定义" color="emerald" />
          <CompareCard dim="🔧 修改成本" oldWay="牵一发而动全身" ai="重说一句，全新一页" color="blue" />
          <CompareCard dim="🧠 认知负担" oldWay="内容 + 结构 + 排版同时想" ai="只管提问，只管思考" color="amber" />
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-amber-500/5 to-emerald-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🔑</div>
        <p className="text-text font-medium">
          笔记的本质不是记录，是理解。而理解的最佳路径不是写，是对话。
        </p>
        <p className="text-sm text-muted mt-1">
          你眼前的这个页面，就是用你说的方式生成的——你不是在读一篇博客，你正在见证这个新范式本身。
        </p>
      </div>
    </>
  )
}

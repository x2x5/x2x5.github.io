import { useState, useEffect } from 'react'

function StarterCard({ title, sub, emoji, example, color }: {
  title: string; sub: string; emoji: string; example: string; color: string
}) {
  const [open, setOpen] = useState(false)
  const dot = `bg-${color}-500`
  const border = open ? `border-${color}-500/30` : 'border-border'

  return (
    <div
      className={`rounded-2xl border ${border} bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] cursor-pointer transition-all duration-200 hover:shadow-[0_16px_40px_var(--color-card-shadow-hover)] select-none`}
      onClick={() => setOpen(!open)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(!open) } }}
    >
      <div className="flex items-center gap-3">
        <div className="text-2xl">{emoji}</div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-text text-base">{title}</h3>
          <p className="text-sm text-muted">{sub}</p>
        </div>
        <div className={`w-6 h-6 rounded-full ${dot} flex items-center justify-center text-white text-xs font-bold transition-transform duration-200 ${open ? 'rotate-45' : ''}`}>
          +
        </div>
      </div>
      <div className={`overflow-hidden transition-all duration-300 ease-in-out ${open ? 'max-h-60 opacity-100 mt-4' : 'max-h-0 opacity-0'}`}>
        <div className="pt-4 border-t border-border">
          <p className="text-sm text-text leading-relaxed">
            <span className="text-muted font-medium">举个例子：</span>
            {example}
          </p>
        </div>
      </div>
    </div>
  )
}

function TreeBranch({ label, icon, children, depth = 0 }: {
  label: string; icon: string; children?: string[]; depth?: number
}) {
  const pad = depth * 6

  return (
    <div>
      <div className="flex items-center gap-3" style={{ paddingLeft: `${pad}px` }}>
        <div className="relative flex items-center">
          {depth > 0 && (
            <div className="absolute -left-6 top-1/2 w-6 h-px bg-border" />
          )}
          <div className="w-10 h-10 rounded-xl bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-lg">
            {icon}
          </div>
        </div>
        <span className="text-text font-medium">{label}</span>
      </div>
      {children && (
        <div className="relative ml-5 pl-5 border-l-2 border-border mt-3 space-y-3">
          {children.map((c, i) => (
            <div key={i} className="relative flex items-center gap-3">
              <div className="absolute -left-[21px] top-1/2 w-[18px] h-px bg-border" />
              <div className="w-7 h-7 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-xs">
                ?
              </div>
              <span className="text-sm text-muted">{c}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const templates = [
  '"传统的 ___ 方法有什么局限？"',
  '"如果换一种方式做 ___，会怎样？"',
  '"___ 和 ___ 之间有什么关系？"',
  '"能不能用 ___ 的思路来解决 ___ 的问题？"',
  '"现有的 ___ 为什么在 ___ 情况下不管用？"',
  '"怎么评价 ___ 这个方法到底好不好？"',
  '"用户在使用 ___ 时，真正的痛点是什么？"',
  '"为什么 ___ 的结果和预期不一样？"',
]

function Generator() {
  const [current, setCurrent] = useState('点击按钮，随机获得一个问题模板 🎲')
  const [spinning, setSpinning] = useState(false)

  const spin = () => {
    if (spinning) return
    setSpinning(true)
    setCurrent('🎰 摇一摇...')
    const duration = 800
    const start = Date.now()

    const interval = setInterval(() => {
      const elapsed = Date.now() - start
      if (elapsed >= duration) {
        clearInterval(interval)
        const pick = templates[Math.floor(Math.random() * templates.length)]
        setCurrent(pick)
        setSpinning(false)
      } else {
        const rand = templates[Math.floor(Math.random() * templates.length)]
        setCurrent(rand)
      }
    }, 100)
  }

  return (
    <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)] text-center">
      <div className="text-4xl mb-3">🎲</div>
      <h3 className="text-lg font-semibold text-text mb-1">问题生成器</h3>
      <p className="text-sm text-muted mb-4">点一下，随机获得一个问题灵感</p>
      <button
        onClick={spin}
        disabled={spinning}
        className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-white transition-all duration-200 cursor-pointer ${spinning ? 'bg-muted cursor-not-allowed' : 'bg-violet-500 hover:bg-violet-600 active:scale-95'}`}
      >
        <span className="text-lg">{spinning ? '🌀' : '🎲'}</span>
        {spinning ? '生成中...' : '随机生成'}
      </button>
      <div className={`mt-5 p-4 rounded-xl border transition-all duration-300 ${current.startsWith('🎰') || current.startsWith('点击') ? 'border-border text-muted' : 'border-violet-500/20 bg-violet-500/5 text-text font-medium'}`}>
        {current}
      </div>
    </div>
  )
}

export default function NoIdeaContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-12px); }
        }
        @keyframes pulse-ring {
          0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.3); }
          70% { box-shadow: 0 0 0 20px rgba(139, 92, 246, 0); }
          100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
        }
        .float-anim { animation: float 3s ease-in-out infinite; }
        .pulse-ring { animation: pulse-ring 2s ease-out infinite; }
      `}</style>

      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-violet-500/10 text-violet-500 mb-5 pulse-ring">
          <span className="text-5xl float-anim inline-block">?</span>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          想不到提什么问题怎么办？
        </h1>
        <p className="text-muted mt-2 text-base">从 0 到 1 生成一个好问题</p>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">🧰 四个万能提问角度</h2>
        </div>
        <p className="text-sm text-muted mb-4">点击卡片展开具体例子</p>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">
          <StarterCard title="假设一下" sub="What if…?" emoji="🔄" color="violet"
            example='"如果把现有的方法反过来做，会怎样？" 比如大家都在用深度学习做分类，你试试用规则+少量样本做分类，就是新问题。' />
          <StarterCard title="追问原因" sub="Why…?" emoji="🔍" color="blue"
            example='"为什么这个现象在 A 场景出现，在 B 场景却不出现？" 找到差异，就找到了问题的切入口。' />
          <StarterCard title="探索方法" sub="How…?" emoji="⚙️" color="emerald"
            example='"怎么用更少的数据达到同样的效果？" 效率、成本、可解释性——这些都是问题矿藏。' />
          <StarterCard title="跨界联想" sub="What about…?" emoji="🔗" color="orange"
            example='"隔壁领域的那个方法，能不能用到我这来？" 比如把 NLP 的 attention 机制用到可视化里，就是新方向。' />
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">🌳 从宽到窄：问题分解树</h2>
        </div>
        <p className="text-sm text-muted mb-4">不知道怎么问？把你的研究领域一步步拆解</p>

        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="space-y-5">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-lg">🎯</div>
              <span className="text-text font-medium">你的研究方向</span>
            </div>
            <div className="relative ml-5 pl-5 border-l-2 border-violet-300 dark:border-violet-700 space-y-4">
              <TreeBranch label="有什么问题还没解决？" icon="❓" depth={1} />
              <TreeBranch label="现有方法有什么局限？" icon="⚠️" depth={1} />
              <div className="relative">
                <div className="flex items-center gap-3" style={{ paddingLeft: '6px' }}>
                  <div className="relative flex items-center">
                    <div className="absolute -left-6 top-1/2 w-6 h-px bg-border" />
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-lg">🔧</div>
                  </div>
                  <span className="text-text font-medium">能不能换个思路？</span>
                </div>
                <div className="relative ml-5 pl-5 border-l-2 border-amber-300 dark:border-amber-700 mt-3 space-y-3">
                  <SubQuestion text='"跨领域的方法能不能借用？"' />
                  <SubQuestion text='"过去被淘汰的方法，现在有新硬件了能不能复活？"' />
                  <SubQuestion text='"用户真实场景中，哪些需求被忽略了？"' />
                </div>
              </div>
            </div>
            <div className="text-center mt-4 pt-4 border-t border-border">
              <span className="inline-flex items-center gap-2 text-sm text-violet-500 font-medium">
                <span>👇</span> 从大到小，从宽到窄，问题自然浮现
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🎲 随手摇个问题</h2>
        </div>
        <Generator />
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-violet-500/5 to-amber-500/5 p-6 text-center">
        <div className="text-2xl mb-2">💡</div>
        <p className="text-text font-medium">好问题不是想出来的，是从「随便问问」里长出来的。</p>
        <p className="text-sm text-muted mt-1">先问 100 个烂问题，第 101 个可能就是好问题。</p>
      </div>
    </>
  )
}

function SubQuestion({ text }: { text: string }) {
  return (
    <div className="relative flex items-center gap-3">
      <div className="absolute -left-[21px] top-1/2 w-[18px] h-px bg-border" />
      <div className="w-7 h-7 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-xs">?</div>
      <span className="text-sm text-muted">{text}</span>
    </div>
  )
}

import { useState, useEffect, useRef } from 'react'

function useInView(threshold = 0.15) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect() } }, { threshold })
    obs.observe(el)
    return () => obs.disconnect()
  }, [threshold])
  return { ref, visible }
}

function Section({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  const { ref, visible } = useInView()
  return (
    <div ref={ref} className={`mb-10 transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'} ${className}`}>
      {children}
    </div>
  )
}

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)] ${className}`}>
      {children}
    </div>
  )
}

/* ─── RL Scenario ─── */
function RLScenario() {
  const [step, setStep] = useState(0)
  const steps = [
    { role: '🧒 Student', action: '解方程：2x + 3 = 7', detail: '2x + 3 = 7\n2x = 7 + 3\n2x = 10\nx = 5', correct: false },
    { role: '🌍 环境', action: '检查答案', detail: '正确答案是 x = 2\nStudent 答 x = 5\n→ ❌ 错了！', correct: false },
    { role: '🧒 Student', action: '收到反馈', detail: '只知道"错了"\n但不知道哪步错了……\n是 7+3 算错了？还是最后除法错了？', correct: false },
  ]
  useEffect(() => {
    setStep(0)
    const timers = steps.map((_, i) => setTimeout(() => setStep(i), 400 + i * 1000))
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <Card className="border-amber-500/20">
      <div className="text-sm font-semibold text-amber-500 mb-3">RL（强化学习）场景</div>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <div key={i} className={`transition-all duration-500 ${i <= step ? 'opacity-100' : 'opacity-20'}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm">{s.role}</span>
              <span className="text-xs text-muted">{s.action}</span>
            </div>
            <div className={`rounded-lg p-3 text-xs font-mono whitespace-pre-line ${i === 2 && s.correct === false ? 'bg-amber-500/10 text-amber-400' : 'bg-card-bg text-muted'}`}>
              {s.detail}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs text-muted text-center">
        环境只能告诉 Student <strong className="text-amber-500">"对"或"错"</strong>，无法指出具体哪步错了
      </div>
    </Card>
  )
}

/* ─── Off-Policy Scenario ─── */
function OffPolicyScenario() {
  const [step, setStep] = useState(0)
  const steps = [
    { role: '🧠 Teacher', action: '解同一道题', detail: '2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2 ✅', correct: true },
    { role: '📦 数据集', action: '存入 Teacher 的解法', detail: '记录：题目 → 正确步骤 → 答案\n（这是 Teacher 走过的路）', correct: true },
    { role: '🧒 Student', action: '模仿学习', detail: '记住：遇到 "2x+3=7" 就按这个步骤做\n但……如果遇到 "2x+3=8" 呢？没学过。', correct: true },
  ]
  useEffect(() => {
    setStep(0)
    const timers = steps.map((_, i) => setTimeout(() => setStep(i), 400 + i * 1000))
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <Card className="border-rose-500/20">
      <div className="text-sm font-semibold text-rose-500 mb-3">Off-Policy Distillation（SFT）场景</div>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <div key={i} className={`transition-all duration-500 ${i <= step ? 'opacity-100' : 'opacity-20'}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm">{s.role}</span>
              <span className="text-xs text-muted">{s.action}</span>
            </div>
            <div className={`rounded-lg p-3 text-xs font-mono whitespace-pre-line ${s.correct ? 'bg-emerald-500/5 text-text' : 'bg-card-bg text-muted'}`}>
              {s.detail}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs text-muted text-center">
        Student 只学过 Teacher 走过的路，<strong className="text-rose-500">自己犯错时的状态从未见过</strong>
      </div>
    </Card>
  )
}

/* ─── On-Policy Distillation Scenario ─── */
function OnPolicyDistillScenario() {
  const [step, setStep] = useState(0)
  const studentTokens = ['2', 'x', ' ', '=', ' ', '7', ' ', '+', ' ', '3', '\n', '2', 'x', ' ', '=', ' ', '7', ' ', '+', ' ', '3', '\n', '2', 'x', ' ', '=', ' ', '1', '0', '\n', 'x', ' ', '=', ' ', '5']

  useEffect(() => {
    setStep(0)
    const timers = Array.from({ length: studentTokens.length + 1 }, (_, i) =>
      setTimeout(() => setStep(i), 300 + i * 200)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <Card className="border-emerald-500/20">
      <div className="text-sm font-semibold text-emerald-500 mb-3">On-Policy Distillation 场景</div>

      <div className="space-y-3 mb-3">
        <div>
          <div className="text-xs text-muted mb-1">🧒 Student 自己写的步骤：</div>
          <div className="rounded-lg p-3 text-xs font-mono whitespace-pre-line bg-card-bg">
            {studentTokens.map((t, i) => {
              const isWrong = i >= 16 && i <= 18 // "7 + 3" 应该是 "7 - 3"
              const isAnswer = i >= 28 // "x = 5"
              const seen = i < step
              return (
                <span key={i} className={`transition-colors duration-200 ${!seen ? 'text-muted/20' : isWrong ? 'text-rose-500 font-bold bg-rose-500/10' : isAnswer ? 'text-amber-500' : 'text-text'}`}>
                  {t}
                </span>
              )
            })}
          </div>
        </div>

        {step > 16 && (
          <div className="transition-all duration-500">
            <div className="text-xs text-muted mb-1">🧠 Teacher 的评分（每个 token）：</div>
            <div className="rounded-lg p-3 text-xs font-mono whitespace-pre-line bg-card-bg">
              {studentTokens.map((t, i) => {
                const isWrong = i >= 16 && i <= 18
                const isAnswer = i >= 28
                const seen = i < step
                if (!seen) return <span key={i} className="text-muted/10">{t}</span>
                if (isWrong) return <span key={i} className="text-rose-500 font-bold">[高 KL！应该是 "-"]</span>
                if (isAnswer) return <span key={i} className="text-amber-500">[可预期，不惩罚]</span>
                return <span key={i} className="text-emerald-500">✓</span>
              })}
            </div>
          </div>
        )}
      </div>

      <div className="mt-3 text-xs text-muted text-center">
        Teacher 对 Student <strong className="text-text">自己写的每一步</strong>都给出反馈 → Student 知道 <strong className="text-emerald-500">"7+3 那步错了，应该是 7-3"</strong>
      </div>
    </Card>
  )
}

/* ─── Environment Explained ─── */
function EnvironmentExplained() {
  const [tab, setTab] = useState<'math' | 'chess' | 'llm'>('math')

  const envs = {
    math: {
      name: '数学题',
      env: '一个判题程序',
      can: '检查最终答案是否等于 2',
      cannot: '不知道 Student 是 "7+3" 算错还是 "10÷2" 算错',
      emoji: '🔢',
    },
    chess: {
      name: '下棋',
  env: '棋局规则 + 胜负判定',
      can: '知道谁赢了',
      cannot: '不知道哪步棋是败着、哪步棋是好棋',
      emoji: '♟️',
    },
    llm: {
      name: '对话/写作',
      env: '人类评价 或 另一个 LLM',
      can: '给整体质量打分',
      cannot: '很难逐 token 指出"这个词用错了"',
      emoji: '💬',
    },
  }
  const e = envs[tab]

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">🌍 "环境"到底是什么？</div>
      <div className="flex gap-2 mb-3">
        {(['math', 'chess', 'llm'] as const).map(key => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-medium transition-all ${tab === key ? 'bg-blue-500 text-white' : 'bg-card-bg text-muted border border-border'}`}
          >
            {envs[key].emoji} {envs[key].name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-3">
          <div className="text-xs font-medium text-emerald-500 mb-1">✅ 环境能做的</div>
          <div className="text-xs text-muted">{e.can}</div>
        </div>
        <div className="rounded-lg bg-rose-500/5 border border-rose-500/20 p-3">
          <div className="text-xs font-medium text-rose-500 mb-1">❌ 环境做不到的</div>
          <div className="text-xs text-muted">{e.cannot}</div>
        </div>
      </div>
      <div className="mt-3 text-xs text-muted">
        环境通常 <strong className="text-text">不是 LLM</strong>，而是一个简单的规则程序或打分函数。它很"笨"，只能看结果。
      </div>
    </Card>
  )
}

/* ─── Teacher vs Environment ─── */
function TeacherVsEnv() {
  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">🧠 Teacher vs 🌍 环境</div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 px-2 text-muted"></th>
              <th className="text-center py-2 px-2 text-amber-500">🌍 环境</th>
              <th className="text-center py-2 px-2 text-blue-500">🧠 Teacher (大模型)</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border">
              <td className="py-2 px-2 text-text">是什么？</td>
              <td className="py-2 px-2 text-center text-muted">判题程序 / 规则</td>
              <td className="py-2 px-2 text-center text-muted">更大的 LLM</td>
            </tr>
            <tr className="border-b border-border">
              <td className="py-2 px-2 text-text">能看什么？</td>
              <td className="py-2 px-2 text-center text-muted">最终答案</td>
              <td className="py-2 px-2 text-center text-muted">每一步的 token</td>
            </tr>
            <tr className="border-b border-border">
              <td className="py-2 px-2 text-text">反馈粒度</td>
              <td className="py-2 px-2 text-center text-amber-500">整题：对/错</td>
              <td className="py-2 px-2 text-center text-blue-500">逐 token：这步好不好</td>
            </tr>
            <tr className="border-b border-border">
              <td className="py-2 px-2 text-text">能指出哪步错？</td>
              <td className="py-2 px-2 text-center text-rose-500">❌ 不能</td>
              <td className="py-2 px-2 text-center text-emerald-500">✅ 能（通过 KL 散度）</td>
            </tr>
            <tr>
              <td className="py-2 px-2 text-text">怎么教 Student？</td>
              <td className="py-2 px-2 text-center text-muted">"你错了，改"</td>
              <td className="py-2 px-2 text-center text-muted">"如果是我，这步会这么写"</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ─── Main ─── */
export default function WhatIsEnvironmentContent() {
  return (
    <>
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.3); }
          50% { box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
        }
        .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
      `}</style>

      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-500/10 text-blue-500 mb-3 pulse-glow">
          <span className="text-2xl">🌍</span>
        </div>
        <h1 className="text-[clamp(1.3rem,3vw,1.8rem)] font-bold text-text m-0">"环境"到底是什么？</h1>
        <p className="text-muted mt-2 text-sm">为什么环境不能指出错误，而 Teacher 能？</p>
      </div>

      {/* 1. Your understanding is correct */}
      <Section>
        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="text-sm font-semibold text-emerald-500 mb-2">✅ 你的理解完全正确</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-xl bg-card-bg border border-border p-3">
              <div className="text-sm font-medium text-text mb-1">🧒 RL = 自己做题</div>
              <p className="text-xs text-muted">小模型自己写计算过程 → 环境告诉它对/错 → 根据对错调整</p>
            </div>
            <div className="rounded-xl bg-card-bg border border-border p-3">
              <div className="text-sm font-medium text-text mb-1">📖 Off-Policy = 看学霸笔记</div>
              <p className="text-xs text-muted">大模型写出过程 → 小模型照着模仿学习</p>
            </div>
          </div>
        </div>
      </Section>

      {/* 2. RL in action */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">🎯 RL 里环境是怎么打分的？</h2>
        </div>
        <RLScenario />
      </Section>

      {/* 3. Off-Policy in action */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-rose-500" />
          <h2 className="text-lg font-semibold text-text m-0">📖 Off-Policy 是怎么学的？</h2>
        </div>
        <OffPolicyScenario />
      </Section>

      {/* 4. What is the environment? */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">🌍 那"环境"到底是什么？</h2>
        </div>
        <EnvironmentExplained />
      </Section>

      {/* 5. Teacher vs Environment */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🧠 Teacher vs 🌍 环境</h2>
        </div>
        <TeacherVsEnv />
      </Section>

      {/* 6. On-Policy Distillation */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🚀 On-Policy Distillation 怎么结合两者？</h2>
        </div>
        <OnPolicyDistillScenario />
      </Section>

      {/* 7. Answer the core question */}
      <Section>
        <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6">
          <div className="text-sm font-semibold text-text mb-3">🎯 回答你的核心问题</div>
          <div className="space-y-3 text-sm text-muted leading-relaxed">
            <p><strong className="text-text">"没有高手的答案，环境不也能挑吗？"</strong></p>
            <p><strong className="text-rose-500">不能。</strong>环境只能看最终答案，没有能力分析过程。它就像一个只会判对错的机器。</p>
            <p><strong className="text-text">"环境本身没有能力挑，你得通过有高手的答案才能挑？"</strong></p>
            <p><strong className="text-emerald-500">对。</strong>Teacher（大模型）有能力理解每一步，知道"如果是我，这步会怎么写"。它不是打分，而是<strong className="text-text">示范</strong>——告诉 Student "在你当前这个状态下，正确的下一步应该是什么"。</p>
            <p>所以 On-Policy Distillation 不是"两个分数选一个学"，而是：<strong className="text-text">Student 自己走，Teacher 每一步告诉它"这步好不好、应该怎么走"</strong>。</p>
          </div>
        </div>
      </Section>
    </>
  )
}

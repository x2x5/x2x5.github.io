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
    <div ref={ref} className={`mb-12 transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'} ${className}`}>
      {children}
    </div>
  )
}

function SectionTitle({ emoji, text }: { emoji: string; text: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <div className="w-1 h-5 rounded-full bg-blue-500" />
      <h2 className="text-lg font-semibold text-text m-0">{emoji} {text}</h2>
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

/* ─── Animated Pipeline ─── */
function PipelineDemo({ mode }: { mode: 'off-policy' | 'rl' | 'on-policy' }) {
  const [step, setStep] = useState(0)
  const steps = {
    'off-policy': [
      { label: 'Teacher 生成答案', icon: '🧠', desc: '大模型写出完整推理过程' },
      { label: '存入数据集', icon: '📦', desc: '收集成固定数据集' },
      { label: 'Student 离线模仿', icon: '📖', desc: '小模型从数据集中学习' },
      { label: '训练结束', icon: '✅', desc: '不再与 teacher 交互' },
    ],
    'rl': [
      { label: 'Student 自己作答', icon: '✍️', desc: '小模型独立生成完整答案' },
      { label: '环境打分', icon: '🎯', desc: '只对最终答案给对/错' },
      { label: '更新策略', icon: '🔄', desc: '根据对错调整参数' },
      { label: '继续尝试', icon: '🔁', desc: '重复采样-打分-更新' },
    ],
    'on-policy': [
      { label: 'Student 自己作答', icon: '✍️', desc: '小模型独立生成答案' },
      { label: 'Teacher 逐 token 评分', icon: '🔍', desc: '对每一步给出详细反馈' },
      { label: 'Student 逐 token 学习', icon: '📚', desc: '知道哪步错了、为什么错' },
      { label: '更新后继续', icon: '🚀', desc: '带着改进继续生成 → 循环' },
    ],
  }
  const s = steps[mode]
  useEffect(() => {
    setStep(0)
    const timers = s.map((_, i) => setTimeout(() => setStep(i), 600 + i * 800))
    return () => timers.forEach(clearTimeout)
  }, [mode])

  const colors = {
    'off-policy': { border: 'border-rose-500/30', bg: 'bg-rose-500/5', accent: 'text-rose-500', dot: 'bg-rose-500' },
    'rl': { border: 'border-amber-500/30', bg: 'bg-amber-500/5', accent: 'text-amber-500', dot: 'bg-amber-500' },
    'on-policy': { border: 'border-emerald-500/30', bg: 'bg-emerald-500/5', accent: 'text-emerald-500', dot: 'bg-emerald-500' },
  }
  const c = colors[mode]

  return (
    <div className={`rounded-xl border ${c.border} ${c.bg} p-4`}>
      <div className="flex flex-col items-center gap-2">
        {s.map((s, i) => (
          <div key={i} className="flex flex-col items-center w-full">
            <div className={`w-full rounded-lg p-3 text-center transition-all duration-500 ${i <= step ? 'opacity-100 scale-100' : 'opacity-20 scale-95'} ${i === step ? `ring-2 ${c.border.replace('/30', '/50')}` : ''}`}>
              <div className="text-xl">{s.icon}</div>
              <div className={`text-sm font-medium mt-1 ${i <= step ? 'text-text' : 'text-muted'}`}>{s.label}</div>
              {i <= step && <div className="text-xs text-muted mt-0.5">{s.desc}</div>}
            </div>
            {i < 3 && <div className={`text-lg transition-colors duration-300 ${i < step ? c.accent : 'text-muted/30'}`}>↓</div>}
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-center gap-1.5">
        {s.map((_, i) => (
          <div key={i} className={`w-2 h-2 rounded-full transition-all duration-300 ${i <= step ? c.dot : 'bg-muted/20'}`} />
        ))}
      </div>
    </div>
  )
}

/* ─── Chess Analogy ─── */
function ChessAnalogy() {
  const [moveIdx, setMoveIdx] = useState(0)
  const moves = [
    { from: 'e2', to: 'e4', grade: 'brilliant', color: 'bg-blue-500', label: '好棋！控制中心' },
    { from: 'e7', to: 'e5', grade: 'best', color: 'bg-blue-400', label: '最佳应对' },
    { from: 'g1', to: 'f3', grade: 'brilliant', color: 'bg-blue-500', label: '好棋！发展棋子' },
    { from: 'b8', to: 'c6', grade: 'best', color: 'bg-blue-400', label: '正常出子' },
    { from: 'f1', to: 'c4', grade: 'brilliant', color: 'bg-blue-500', label: '好棋！瞄准 f7' },
    { from: 'g8', to: 'f6', grade: 'mistake', color: 'bg-amber-500', label: '失误！忽略了威胁' },
    { from: 'f3', to: 'g5', grade: 'brilliant', color: 'bg-blue-500', label: '好棋！利用弱点' },
    { from: 'd7', to: 'd5', grade: 'blunder', color: 'bg-rose-500', label: '大错！丢子了' },
  ]
  useEffect(() => {
    if (moveIdx >= moves.length) return
    const t = setTimeout(() => setMoveIdx(i => i + 1), 1200)
    return () => clearTimeout(t)
  }, [moveIdx])

  return (
    <Card>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <div className="text-sm font-medium text-text mb-2">🏁 像 chess.com 的走棋分析</div>
          <div className="grid grid-cols-8 gap-0.5 mb-3">
            {Array.from({ length: 64 }).map((_, i) => {
              const row = Math.floor(i / 8)
              const col = i % 8
              const isLight = (row + col) % 2 === 0
              const move = moves.find(m => {
                const fromCol = m.from.charCodeAt(0) - 97
                const fromRow = 8 - parseInt(m.from[1])
                const toCol = m.to.charCodeAt(0) - 97
                const toRow = 8 - parseInt(m.to[1])
                return i === fromRow * 8 + fromCol || i === toRow * 8 + toCol
              })
              const moveOrder = moves.indexOf(move!)
              const isActive = move && moveOrder < moveIdx
              return (
                <div
                  key={i}
                  className={`aspect-square rounded-sm transition-all duration-500 ${isLight ? 'bg-amber-100 dark:bg-amber-900/30' : 'bg-amber-800 dark:bg-amber-700/50'} ${isActive ? `${move!.color} opacity-80` : ''}`}
                />
              )
            })}
          </div>
          <div className="flex gap-2 text-xs text-muted flex-wrap">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-blue-500" />好棋</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" />失误</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-rose-500" />大错</span>
          </div>
        </div>
        <div className="space-y-1.5 max-h-52 overflow-y-auto">
          {moves.slice(0, moveIdx).map((m, i) => (
            <div key={i} className={`flex items-center gap-2 text-xs p-1.5 rounded transition-all duration-300 ${m.grade === 'blunder' ? 'bg-rose-500/10' : m.grade === 'mistake' ? 'bg-amber-500/10' : 'bg-blue-500/5'}`}>
              <span className={`w-2 h-2 rounded-full ${m.color}`} />
              <span className="font-mono text-text">{m.from}→{m.to}</span>
              <span className="text-muted">{m.label}</span>
            </div>
          ))}
          {moveIdx >= moves.length && (
            <div className="text-xs text-emerald-500 font-medium mt-2">✅ 每步棋都有详细反馈 → 这就是 on-policy distillation 的思路</div>
          )}
        </div>
      </div>
    </Card>
  )
}

/* ─── KL Divergence Visual ─── */
function KLVisual() {
  const [progress, setProgress] = useState(0)
  const { ref, visible } = useInView()
  useEffect(() => {
    if (!visible) return
    let frame: number
    const start = performance.now()
    const animate = (now: number) => {
      const p = Math.min((now - start) / 3000, 1)
      setProgress(p)
      if (p < 1) frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [visible])

  const teacherPeak = 50
  const studentPeak = 50 + (1 - progress) * 30
  const teacherSpread = 8
  const studentSpread = 8 + (1 - progress) * 15

  const generateCurve = (peak: number, spread: number) => {
    const pts: string[] = []
    for (let x = 0; x <= 100; x++) {
      const y = Math.exp(-0.5 * ((x - peak) / spread) ** 2)
      pts.push(`${x * 2.4},${80 - y * 70}`)
    }
    return `M${pts.join(' L')}`
  }

  return (
    <div ref={ref}>
      <Card>
        <div className="text-sm font-medium text-text mb-2">Reverse KL：让 student 分布逼近 teacher</div>
        <svg viewBox="0 0 240 90" className="w-full h-auto">
          <defs>
            <linearGradient id="teacherGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgb(59,130,246)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="rgb(59,130,246)" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="studentGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgb(16,185,129)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="rgb(16,185,129)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={generateCurve(teacherPeak, teacherSpread)} fill="url(#teacherGrad)" stroke="rgb(59,130,246)" strokeWidth="2" />
          <path d={generateCurve(studentPeak, studentSpread)} fill="url(#studentGrad)" stroke="rgb(16,185,129)" strokeWidth="2" />
          <text x="120" y="88" textAnchor="middle" className="fill-muted" fontSize="8">token 空间</text>
        </svg>
        <div className="flex justify-center gap-4 text-xs mt-1">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500" />Teacher 分布</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500" />Student 分布</span>
        </div>
        <div className="mt-3 text-center">
          <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium transition-all duration-500 ${progress > 0.8 ? 'bg-emerald-500/10 text-emerald-500' : 'bg-amber-500/10 text-amber-500'}`}>
            {progress > 0.8 ? '✅ KL ≈ 0 — Student 已学会 Teacher 的行为' : `训练中... KL 正在减小 (${Math.round(progress * 100)}%)`}
          </span>
        </div>
      </Card>
    </div>
  )
}

/* ─── Comparison Table ─── */
function MethodTable() {
  const rows = [
    { method: 'Supervised Fine-Tuning', sampling: 'Off-policy', reward: 'Dense', icon: '📖' },
    { method: 'Reinforcement Learning', sampling: 'On-policy', reward: 'Sparse', icon: '🎯' },
    { method: 'On-Policy Distillation', sampling: 'On-policy', reward: 'Dense', icon: '🚀' },
  ]
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-3 text-muted font-medium">方法</th>
            <th className="text-center py-2 px-3 text-muted font-medium">采样方式</th>
            <th className="text-center py-2 px-3 text-muted font-medium">反馈密度</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={`border-b border-border ${i === 2 ? 'bg-emerald-500/5' : ''}`}>
              <td className="py-2.5 px-3 text-text font-medium">{r.icon} {r.method}</td>
              <td className="py-2.5 px-3 text-center text-muted">{r.sampling}</td>
              <td className="py-2.5 px-3 text-center text-muted">{r.reward}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Experiment Bar Chart ─── */
function BarChart() {
  const { ref, visible } = useInView()
  const data = [
    { label: 'SFT 400K', value: 60, color: 'bg-rose-500' },
    { label: 'SFT 2M (预估)', value: 70, color: 'bg-rose-400' },
    { label: 'RL', value: 68, color: 'bg-amber-500' },
    { label: 'On-Policy Distill', value: 74.4, color: 'bg-emerald-500' },
  ]
  return (
    <div ref={ref}>
      <Card>
        <div className="text-sm font-medium text-text mb-3">📊 AIME'24 数学竞赛成绩对比</div>
        <div className="space-y-3">
          {data.map((d, i) => (
            <div key={i}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-text">{d.label}</span>
                <span className="text-muted font-mono">{d.value}%</span>
              </div>
              <div className="h-6 rounded-lg bg-card-bg overflow-hidden">
                <div
                  className={`h-full ${d.color} rounded-lg transition-all duration-1000 ease-out flex items-center justify-end pr-2`}
                  style={{ width: visible ? `${(d.value / 80) * 100}%` : '0%' }}
                >
                  <span className="text-[0.6rem] text-white font-medium">{d.value}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-muted">On-policy distillation 用 1/10 的算力超过了 RL 的效果</div>
      </Card>
    </div>
  )
}

/* ─── Personalization Flow ─── */
function PersonalizationFlow() {
  const { ref, visible } = useInView()
  const stages = [
    { name: 'Qwen3-8B', qa: 18, ifEval: 85, desc: '原始模型：指令跟随强，但不了解公司内部文档' },
    { name: '+ 内部文档微调', qa: 43, ifEval: 45, desc: '学到了知识，但指令跟随能力严重退化' },
    { name: '+ On-Policy Distill', qa: 41, ifEval: 83, desc: '用 teacher 恢复了指令跟随，知识也保留了' },
  ]
  return (
    <div ref={ref}>
      <Card>
        <div className="text-sm font-medium text-text mb-3">🧑‍💼 个性化助手：先学知识，再恢复行为</div>
        <div className="space-y-4">
          {stages.map((s, i) => (
            <div key={i} className="flex flex-col sm:flex-row sm:items-center gap-3">
              <div className="sm:w-36 flex-shrink-0">
                <div className="text-sm font-medium text-text">{s.name}</div>
                <div className="text-xs text-muted">{s.desc}</div>
              </div>
              <div className="flex-1 grid grid-cols-2 gap-2">
                <div>
                  <div className="text-xs text-muted mb-1">知识 (Internal QA)</div>
                  <div className="h-5 rounded bg-card-bg overflow-hidden">
                    <div className={`h-full rounded transition-all duration-1000 ${s.qa > 35 ? 'bg-emerald-500' : 'bg-rose-500'}`} style={{ width: visible ? `${s.qa}%` : '0%' }} />
                  </div>
                  <div className="text-xs text-muted font-mono mt-0.5">{s.qa}%</div>
                </div>
                <div>
                  <div className="text-xs text-muted mb-1">指令跟随 (IF-eval)</div>
                  <div className="h-5 rounded bg-card-bg overflow-hidden">
                    <div className={`h-full rounded transition-all duration-1000 ${s.ifEval > 70 ? 'bg-emerald-500' : 'bg-rose-500'}`} style={{ width: visible ? `${s.ifEval}%` : '0%' }} />
                  </div>
                  <div className="text-xs text-muted font-mono mt-0.5">{s.ifEval}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* ─── Main Component ─── */
export default function OnPolicyDistillationContent() {
  const [mode, setMode] = useState<'off-policy' | 'rl' | 'on-policy'>('off-policy')

  return (
    <>
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.3); }
          50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        }
        .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
      `}</style>

      {/* Hero */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-blue-500/10 text-blue-500 mb-4 pulse-glow">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.5rem,3.5vw,2rem)] font-bold text-text m-0">On-Policy Distillation</h1>
        <p className="text-muted mt-2 text-sm">让 student 在自己生成的轨迹上，接受 teacher 的逐 token 指导</p>
        <span className="inline-block mt-2 text-[0.65rem] text-muted/40">来源: Thinking Machines Lab · Oct 2025</span>
      </div>

      {/* 1. The Big Picture */}
      <Section>
        <SectionTitle emoji="🗺️" text="训练的三个stage" />
        <Card>
          <div className="flex flex-col sm:flex-row items-stretch gap-3">
            {[
              { emoji: '📚', title: 'Pre-training', desc: '学语言、常识、世界知识', color: 'from-blue-500/10 to-blue-500/5', border: 'border-blue-500/20' },
              { emoji: '🔧', title: 'Mid-training', desc: '注入领域知识（代码、医学、公司文档）', color: 'from-amber-500/10 to-amber-500/5', border: 'border-amber-500/20' },
              { emoji: '🎯', title: 'Post-training', desc: '激发特定行为（指令跟随、推理、对话）', color: 'from-emerald-500/10 to-emerald-500/5', border: 'border-emerald-500/20' },
            ].map((s, i) => (
              <div key={i} className={`flex-1 rounded-xl bg-gradient-to-br ${s.color} border ${s.border} p-4 text-center`}>
                <div className="text-2xl mb-1">{s.emoji}</div>
                <div className="text-sm font-semibold text-text">{s.title}</div>
                <div className="text-xs text-muted mt-1">{s.desc}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted mt-3 text-center">本文关注的是 <strong className="text-text">post-training</strong> 阶段——怎么让小模型获得 expert 级别的行为</p>
        </Card>
      </Section>

      {/* 2. The Core Problem */}
      <Section>
        <SectionTitle emoji="⚡" text="两种 post-training 方式的问题" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
          <Card className="border-rose-500/20">
            <div className="text-sm font-semibold text-rose-500 mb-2">Off-Policy（SFT / Distillation）</div>
            <p className="text-xs text-muted">Student 从 teacher 生成的<strong>固定数据集</strong>中学习。问题是：student 学到的都是 teacher 常遇到的状态，自己犯错时的状态从未见过 → <strong className="text-text">误差会累积放大</strong>。</p>
          </Card>
          <Card className="border-amber-500/20">
            <div className="text-sm font-semibold text-amber-500 mb-2">On-Policy（RL）</div>
            <p className="text-xs text-muted">Student 自己生成答案，环境只给<strong>最终对/错</strong>。问题是：反馈太稀疏——做了一道数学题，只知道答案错了，但<strong className="text-text">不知道哪步错了</strong>。</p>
          </Card>
        </div>

        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="text-sm font-semibold text-emerald-500 mb-2">💡 On-Policy Distillation = 两者结合</div>
          <p className="text-xs text-muted">Student <strong className="text-text">自己生成</strong>轨迹（on-policy），但 teacher 对<strong className="text-text">每个 token</strong>都给出评分（dense reward）。就像 chess.com 的走棋分析——每步棋都标注"好棋""失误""大错"。</p>
        </div>
      </Section>

      {/* 3. Interactive Pipeline Comparison */}
      <Section>
        <SectionTitle emoji="🔄" text="三种方法流程对比" />
        <div className="flex gap-2 mb-4">
          {[
            { key: 'off-policy' as const, label: 'Off-Policy SFT', color: 'bg-rose-500' },
            { key: 'rl' as const, label: 'RL', color: 'bg-amber-500' },
            { key: 'on-policy' as const, label: 'On-Policy Distill', color: 'bg-emerald-500' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setMode(tab.key)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${mode === tab.key ? `${tab.color} text-white` : 'bg-card-bg text-muted border border-border'}`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <PipelineDemo mode={mode} />
      </Section>

      {/* 4. Chess Analogy */}
      <Section>
        <SectionTitle emoji="♟️" text="Chess.com 类比" />
        <p className="text-xs text-muted mb-3">原文用 chess.com 的走棋分析来类比：RL 只知道输赢，off-policy 只看高手下棋，on-policy distillation 则是每步棋都有引擎评分。</p>
        <ChessAnalogy />
      </Section>

      {/* 5. Method Comparison Table */}
      <Section>
        <SectionTitle emoji="📋" text="方法对比" />
        <Card>
          <MethodTable />
        </Card>
      </Section>

      {/* 6. How it works - Reverse KL */}
      <Section>
        <SectionTitle emoji="📐" text="核心：Reverse KL 损失函数" />
        <p className="text-xs text-muted mb-3">On-policy distillation 用 reverse KL 散度来衡量 student 和 teacher 的差异。Student 每生成一个 token，teacher 就给出这个 token 的 log probability，student 据此调整自己。</p>
        <KLVisual />
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
          {[
            { icon: '🎯', title: 'Mode Seeking', desc: '专注学习 teacher 的一种行为，而不是分散到多个次优选项' },
            { icon: '🛡️', title: 'Unhackable', desc: '低 KL 一定对应 teacher 认可的高概率行为，不会被"钻空子"' },
            { icon: '⚡', title: '减少 Exposure Bias', desc: 'Student 在自己遇到的状态中学习，而不是只在 teacher 的状态中' },
          ].map((c, i) => (
            <div key={i} className="rounded-xl border border-border bg-card-bg p-3">
              <div className="text-lg mb-1">{c.icon}</div>
              <div className="text-sm font-medium text-text">{c.title}</div>
              <div className="text-xs text-muted mt-0.5">{c.desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* 7. Pseudocode */}
      <Section>
        <SectionTitle emoji="💻" text="实现只需要 4 步" />
        <Card>
          <div className="space-y-3">
            {[
              { step: '1', title: '初始化 Teacher 客户端', code: 'teacher_client = create_sampling_client(...)', desc: '用 Tinker API 连接 teacher 模型' },
              { step: '2', title: 'Student 采样轨迹', code: 'trajectories = do_group_rollout(student_client, ...)', desc: '和 RL 一样，从 student 采样 rollout' },
              { step: '3', title: 'Teacher 计算 logprobs', code: 'teacher_logprobs = teacher_client.compute_logprobs(trajectories)', desc: '对 student 生成的每个 token 求 teacher 的 log probability' },
              { step: '4', title: 'RL 训练更新', code: 'advantages = -(sampled_logprobs - teacher_logprobs)', desc: '用负的 reverse KL 作为 advantage，调用 RL 的 loss 函数更新' },
            ].map((s, i) => (
              <div key={i} className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-500/10 text-blue-500 flex items-center justify-center text-sm font-bold flex-shrink-0">{s.step}</div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-text">{s.title}</div>
                  <code className="text-xs text-accent font-mono bg-card-bg px-2 py-0.5 rounded mt-0.5 inline-block">{s.code}</code>
                  <div className="text-xs text-muted mt-1">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <div className="mt-4 text-center">
          <a
            href="./article.html?id=how-on-policy-distill-works"
            className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/5 text-blue-500 px-4 py-2 text-sm no-underline transition-colors hover:bg-blue-500/10"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
            深入：每一步的 tensor shape 和 compute_logprobs 内部机制
          </a>
        </div>
      </Section>

      {/* 8. Reasoning Experiments */}
      <Section>
        <SectionTitle emoji="🧮" text="实验 1：数学推理" />
        <p className="text-xs text-muted mb-3">用 Qwen3-8B-Base 作为 student，Qwen3-32B 作为 teacher，在数学推理任务上对比三种方法。</p>
        <BarChart />
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Card>
            <div className="text-sm font-semibold text-text mb-2">💰 算力对比</div>
            <div className="space-y-2">
              {[
                { label: 'SFT 2M (预估)', factor: '1×', color: 'text-rose-500' },
                { label: 'RL', factor: '≈1×', color: 'text-amber-500' },
                { label: 'On-Policy Distill', factor: '9-30× 更省', color: 'text-emerald-500' },
              ].map((r, i) => (
                <div key={i} className="flex justify-between items-center text-sm">
                  <span className="text-text">{r.label}</span>
                  <span className={`font-mono font-medium ${r.color}`}>{r.factor}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <div className="text-sm font-semibold text-text mb-2">🔑 关键发现</div>
            <ul className="text-xs text-muted space-y-1.5">
              <li>• On-policy distillation 用 <strong className="text-text">1/10 的 GPU 小时</strong>超过 RL</li>
              <li>• 如果算上生成 SFT 数据的成本，节省 <strong className="text-text">30 倍</strong></li>
              <li>• LoRA 模型在 on-policy distill 下表现更好，差距从 13% 缩小到 6%</li>
            </ul>
          </Card>
        </div>
      </Section>

      {/* 9. Personalization */}
      <Section>
        <SectionTitle emoji="🧑‍💼" text="实验 2：个性化助手" />
        <p className="text-xs text-muted mb-3">训练一个了解公司内部文档的助手。问题：学新知识会破坏原有的指令跟随能力。On-policy distillation 可以恢复。</p>
        <PersonalizationFlow />
        <div className="mt-3 rounded-xl border border-border bg-card-bg p-4">
          <div className="text-sm font-medium text-text mb-2">🔄 持续学习的应用场景</div>
          <p className="text-xs text-muted">可以交替进行：<strong className="text-text">学新知识（mid-train）</strong>→<strong className="text-text">恢复行为（distill）</strong>→ 再学新知识 → 再恢复。模型可以持续更新而不退化。</p>
        </div>
      </Section>

      {/* 10. Single Prompt Experiment */}
      <Section>
        <SectionTitle emoji="🎯" text="只用 1 道题就能学？" />
        <Card>
          <p className="text-sm text-text mb-3">原文做了一个惊人的实验：<strong>只用 1 道数学题</strong>，反复采样 5120 次，on-policy distillation 就能让 student 达到 teacher 的 AIME 水平。</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-xl bg-card-bg border border-border p-3 text-center">
              <div className="text-2xl font-bold text-text">1</div>
              <div className="text-xs text-muted">训练题目数量</div>
            </div>
            <div className="rounded-xl bg-card-bg border border-border p-3 text-center">
              <div className="text-2xl font-bold text-text">5120</div>
              <div className="text-xs text-muted">采样序列总数</div>
            </div>
            <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/20 p-3 text-center">
              <div className="text-2xl font-bold text-emerald-500">≈Teacher</div>
              <div className="text-xs text-muted">最终 AIME 成绩</div>
            </div>
          </div>
          <p className="text-xs text-muted mt-3">原因：reverse KL 学的是 teacher 的<strong className="text-text">完整分布</strong>，不是死记一个答案。同一道题可以反复采样出不同的推理路径，每条路径都提供新的学习信号。</p>
        </Card>
      </Section>

      {/* 11. Key Insights */}
      <Section>
        <SectionTitle emoji="💡" text="核心洞察" />
        <div className="space-y-3">
          {[
            {
              icon: '🔬',
              title: 'RL 是在搜索"语义策略空间"',
              desc: 'Pre-training 是在高维参数空间里探索，而 RL 是在"策略空间"里摸索。一旦找到好策略，distillation 就是捷径——直接学最终策略，不用重走中间过程。就像科学研究花了很多时间找答案，但教给别人只需要用自然语言表达出来。',
            },
            {
              icon: '📊',
              title: 'Dense supervision 大幅提升效率',
              desc: 'RL 每轮只教 O(1) bits（对/错），distillation 每轮教 O(N) bits（每个 token 都有反馈）。实验显示 distillation 用 7-10 步就达到 RL 用 70 步的效果，综合算力节省 50-100 倍。',
            },
            {
              icon: '🔄',
              title: 'On-policy 学习是持续学习的工具',
              desc: 'SFT 会在自己的样本上退化（有限 batch 导致分布偏移），但 on-policy distillation 因为 teacher 固定，student 始终收敛到 teacher 的行为，不会退化。这使它成为持续学习的有力工具。',
            },
          ].map((insight, i) => (
            <Card key={i}>
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-lg">{insight.icon}</div>
                <div>
                  <h4 className="font-semibold text-text text-sm">{insight.title}</h4>
                  <p className="text-xs text-muted mt-1 leading-relaxed">{insight.desc}</p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Section>

      {/* Summary */}
      <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🎯</div>
        <p className="text-text font-medium">
          On-Policy Distillation 一句话总结：<br />
          Student 自己走，Teacher 每步评，学得又快又准。
        </p>
        <p className="text-sm text-muted mt-2">结合了 on-policy 的相关性和 dense reward 的效率，是 post-training 的"两全其美"方案。</p>
        <a href="https://thinkingmachines.ai/blog/on-policy-distillation/" target="_blank" rel="noopener noreferrer" className="inline-block mt-3 text-xs text-blue-500 hover:underline">
          阅读原文 → thinkingmachines.ai/blog/on-policy-distillation
        </a>
      </div>
    </>
  )
}

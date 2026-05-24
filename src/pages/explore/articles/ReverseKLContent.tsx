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

/* ─── Distribution Visualizer ─── */
function DistributionViz({
  teacher,
  student,
  showKL,
  label,
}: {
  teacher: number[]
  student: number[]
  showKL: boolean
  label: string
}) {
  const tokens = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'in']
  const maxVal = Math.max(...teacher, ...student)

  return (
    <div>
      <div className="text-sm font-medium text-text mb-2">{label}</div>
      <div className="flex items-end gap-1 h-32 mb-2">
        {tokens.map((t, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-0.5 h-full justify-end">
            <div className="w-full flex flex-col items-center gap-0.5" style={{ height: '100%' }}>
              <div className="flex flex-col items-center justify-end w-full" style={{ height: '100%' }}>
                {showKL && (
                  <div
                    className="w-full rounded-t transition-all duration-500 bg-rose-500/30 border border-rose-500/50"
                    style={{ height: `${Math.abs(teacher[i] - student[i]) / maxVal * 50}%`, minHeight: Math.abs(teacher[i] - student[i]) > 0.01 ? '4px' : '0' }}
                  />
                )}
                <div
                  className="w-full rounded-t transition-all duration-500 bg-blue-500/60"
                  style={{ height: `${teacher[i] / maxVal * 60}%` }}
                />
                <div
                  className="w-full rounded-t transition-all duration-500 bg-emerald-500/60 -mt-1"
                  style={{ height: `${student[i] / maxVal * 60}%` }}
                />
              </div>
            </div>
            <span className="text-[0.55rem] text-muted">{t}</span>
          </div>
        ))}
      </div>
      <div className="flex gap-3 text-xs">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-blue-500/60" />Teacher</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/60" />Student</span>
        {showKL && <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-rose-500/30" />KL 差异</span>}
      </div>
    </div>
  )
}

/* ─── Interactive KL Calculator ─── */
function KLInteractive() {
  const [mode, setMode] = useState<'forward' | 'reverse'>('reverse')
  const [progress, setProgress] = useState(0)
  const { ref, visible } = useInView()

  useEffect(() => {
    if (!visible) return
    setProgress(0)
    let frame: number
    const start = performance.now()
    const animate = (now: number) => {
      const p = Math.min((now - start) / 2000, 1)
      setProgress(p)
      if (p < 1) frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [visible, mode])

  const teacher = [0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02]
  const studentBase = [0.2, 0.1, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
  const student = studentBase.map((s, i) => s + (teacher[i] - s) * progress)

  const forwardKL = teacher.reduce((sum, t, i) => sum + t * Math.log(t / Math.max(student[i], 1e-10)), 0)
  const reverseKL = student.reduce((sum, s, i) => sum + s * Math.log(s / Math.max(teacher[i], 1e-10)), 0)

  return (
    <div ref={ref}>
      <Card>
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setMode('reverse')}
            className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${mode === 'reverse' ? 'bg-emerald-500 text-white' : 'bg-card-bg text-muted border border-border'}`}
          >
            Reverse KL（On-Policy 用的）
          </button>
          <button
            onClick={() => setMode('forward')}
            className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${mode === 'forward' ? 'bg-amber-500 text-white' : 'bg-card-bg text-muted border border-border'}`}
          >
            Forward KL（SFT 用的）
          </button>
        </div>

        <DistributionViz
          teacher={teacher}
          student={student}
          showKL={true}
          label={mode === 'reverse' ? 'Reverse KL: E_{Student}[log Student - log Teacher]' : 'Forward KL: E_{Teacher}[log Teacher - log Student]'}
        />

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className={`rounded-lg p-3 transition-all ${mode === 'reverse' ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-card-bg border border-border'}`}>
            <div className="text-xs text-muted mb-1">Reverse KL 值</div>
            <div className={`text-lg font-mono font-bold ${mode === 'reverse' ? 'text-emerald-500' : 'text-muted'}`}>
              {reverseKL.toFixed(3)}
            </div>
          </div>
          <div className={`rounded-lg p-3 transition-all ${mode === 'forward' ? 'bg-amber-500/10 border border-amber-500/30' : 'bg-card-bg border border-border'}`}>
            <div className="text-xs text-muted mb-1">Forward KL 值</div>
            <div className={`text-lg font-mono font-bold ${mode === 'forward' ? 'text-amber-500' : 'text-muted'}`}>
              {forwardKL.toFixed(3)}
            </div>
          </div>
        </div>

        <div className="mt-3 text-xs text-muted">
          {mode === 'reverse' ? (
            <span>Reverse KL 是 <strong className="text-text">mode seeking</strong>：Student 会聚焦到 Teacher 概率最高的那个 token 上（"cat"），忽略其他可能性。</span>
          ) : (
            <span>Forward KL 是 <strong className="text-text">mode covering</strong>：Student 会尝试覆盖 Teacher 的所有可能性，导致分布更分散。</span>
          )}
        </div>
      </Card>
    </div>
  )
}

/* ─── Mode Seeking vs Mode Covering ─── */
function ModeComparison() {
  const [step, setStep] = useState(0)
  const { ref, visible } = useInView()

  useEffect(() => {
    if (!visible) return
    setStep(0)
    const timers = [0, 1, 2].map(i => setTimeout(() => setStep(i), 400 + i * 1000))
    return () => timers.forEach(clearTimeout)
  }, [visible])

  const scenarios = [
    {
      title: 'Teacher 的分布',
      desc: 'Teacher 认为 "cat" 概率最高（70%），其他词概率很低',
      teacher: [0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02],
      student: [0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02],
    },
    {
      title: 'Forward KL（SFT）的结果',
      desc: 'Student 尝试覆盖 Teacher 的所有可能性，分布更分散，不够确定',
      teacher: [0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02],
      student: [0.1, 0.4, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05],
    },
    {
      title: 'Reverse KL（On-Policy）的结果',
      desc: 'Student 聚焦到 Teacher 最可能的选择上，分布更集中',
      teacher: [0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02],
      student: [0.03, 0.85, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01],
    },
  ]

  return (
    <div ref={ref}>
      <Card>
        <div className="text-sm font-semibold text-text mb-3">🎯 Mode Seeking vs Mode Covering</div>
        <div className="space-y-4">
          {scenarios.map((s, i) => (
            <div key={i} className={`transition-all duration-500 ${i <= step ? 'opacity-100' : 'opacity-20'}`}>
              <div className="text-sm font-medium text-text mb-1">{s.title}</div>
              <p className="text-xs text-muted mb-2">{s.desc}</p>
              <DistributionViz teacher={s.teacher} student={s.student} showKL={false} label="" />
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* ─── Why Reverse KL for On-Policy ─── */
function WhyReverseKL() {
  const reasons = [
    {
      icon: '🎯',
      title: 'Mode Seeking',
      desc: 'Reverse KL 让 Student 学习 Teacher 最可能的行为，而不是尝试覆盖所有可能性。这避免了 Student 产生 Teacher 不会产生的低质量输出。',
    },
    {
      icon: '🛡️',
      title: 'Unhackable',
      desc: '低 Reverse KL 一定对应 Teacher 认可的高概率行为。Student 不能"钻空子"——它必须真正模仿 Teacher 的分布。',
    },
    {
      icon: '📉',
      title: '减少 Exposure Bias',
      desc: 'Student 在自己生成的轨迹上学习，而不是在 Teacher 的轨迹上学习。这避免了训练和推理时的分布不匹配问题。',
    },
  ]

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">🤔 为什么 On-Policy Distillation 用 Reverse KL？</div>
      <div className="space-y-3">
        {reasons.map((r, i) => (
          <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-card-bg border border-border">
            <div className="w-9 h-9 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-lg">{r.icon}</div>
            <div>
              <h4 className="text-sm font-semibold text-text">{r.title}</h4>
              <p className="text-xs text-muted mt-1 leading-relaxed">{r.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

/* ─── Formula Breakdown ─── */
function FormulaBreakdown() {
  const [expanded, setExpanded] = useState<number | null>(null)

  const parts = [
    { symbol: 'KL(π_θ || π_teacher)', name: 'Reverse KL 散度', desc: '衡量 Student 分布 π_θ 和 Teacher 分布 π_teacher 之间的差异。越小越好。' },
    { symbol: 'E_{x ~ π_θ}', name: '期望（对 Student 采样）', desc: '从 Student 的分布中采样 token x。这就是"on-policy"——数据来自 Student 自己。' },
    { symbol: 'log π_θ(x)', name: 'Student 的 log probability', desc: 'Student 生成 token x 时，自己认为这个 token 的 log 概率。' },
    { symbol: 'log π_teacher(x)', name: 'Teacher 的 log probability', desc: 'Teacher 对同一个 token x 的 log 概率。通过 compute_logprobs 获得。' },
    { symbol: 'log π_θ - log π_teacher', name: '逐 token 的 KL', desc: '如果 Student 和 Teacher 在这个 token 上一致，差值为 0。如果不一致，差值越大，惩罚越大。' },
  ]

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">📐 公式拆解</div>
      <div className="rounded-lg bg-card-bg border border-border p-4 mb-4 text-center">
        <div className="text-sm font-mono text-text">
          KL(π_θ || π_teacher) = E<sub>x ~ π_θ</sub>[ log π_θ(x) - log π_teacher(x) ]
        </div>
      </div>
      <div className="space-y-2">
        {parts.map((p, i) => (
          <div key={i} className="rounded-lg border border-border bg-card-bg overflow-hidden">
            <button
              onClick={() => setExpanded(expanded === i ? null : i)}
              className="w-full flex items-center justify-between p-3 text-left"
            >
              <code className="text-xs text-blue-500 font-mono">{p.symbol}</code>
              <span className="text-muted text-xs">{expanded === i ? '▲' : '▼'}</span>
            </button>
            {expanded === i && (
              <div className="px-3 pb-3 border-t border-border pt-2">
                <p className="text-xs font-medium text-text">{p.name}</p>
                <p className="text-xs text-muted mt-1">{p.desc}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  )
}

/* ─── Main ─── */
export default function ReverseKLContent() {
  return (
    <>
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.3); }
          50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        }
        .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
      `}</style>

      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-500/10 text-emerald-500 mb-3 pulse-glow">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20V10" />
            <path d="M18 20V4" />
            <path d="M6 20v-4" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.3rem,3vw,1.8rem)] font-bold text-text m-0">Reverse KL 到底是什么？</h1>
        <p className="text-muted mt-2 text-sm">用可视化理解 On-Policy Distillation 的核心损失函数</p>
      </div>

      {/* 1. Intuition */}
      <Section>
        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="text-sm font-semibold text-emerald-500 mb-2">💡 一句话理解</div>
          <p className="text-sm text-muted leading-relaxed">
            Reverse KL 就是衡量 <strong className="text-text">Student 的分布</strong> 和 <strong className="text-text">Teacher 的分布</strong> 有多像。
            不像就惩罚，像就奖励。目标是让 Student 的分布尽可能接近 Teacher 的分布。
          </p>
        </div>
      </Section>

      {/* 2. What is a distribution? */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">📊 什么是"概率分布"？</h2>
        </div>
        <Card>
          <p className="text-xs text-muted mb-3">
            当模型要生成下一个 token 时，它不是直接选一个词，而是对所有可能的词给出一个概率。这就是"分布"。
          </p>
          <DistributionViz
            teacher={[0.05, 0.7, 0.1, 0.05, 0.03, 0.02, 0.03, 0.02]}
            student={[0.1, 0.4, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05]}
            showKL={true}
            label="Teacher vs Student 的下一个 token 概率分布"
          />
          <p className="text-xs text-muted mt-3">
            Teacher 认为 "cat" 概率最高（70%），Student 也觉得 "cat" 最高但只有 40%，还分了很多概率给其他词。
            这两个分布的"差距"就是 KL 散度要衡量的东西。
          </p>
        </Card>
      </Section>

      {/* 3. Interactive KL */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔄 交互式 KL 演示</h2>
        </div>
        <KLInteractive />
      </Section>

      {/* 4. Formula Breakdown */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">📐 公式拆解</h2>
        </div>
        <FormulaBreakdown />
      </Section>

      {/* 5. Mode Seeking vs Covering */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">🎯 Mode Seeking vs Mode Covering</h2>
        </div>
        <ModeComparison />
      </Section>

      {/* 6. Why Reverse KL */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🤔 为什么 On-Policy 用 Reverse KL？</h2>
        </div>
        <WhyReverseKL />
      </Section>

      {/* 7. Summary */}
      <div className="rounded-2xl border border-border bg-gradient-to-br from-emerald-500/5 to-blue-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🎯</div>
        <p className="text-text font-medium">
          Reverse KL 一句话总结：<br />
          让 Student 的分布聚焦到 Teacher 最可能的选择上，而不是尝试覆盖所有可能性。
        </p>
        <p className="text-sm text-muted mt-2">这就是为什么 On-Policy Distillation 能学到 Teacher 的精确行为，而不是模糊模仿。</p>
      </div>
    </>
  )
}

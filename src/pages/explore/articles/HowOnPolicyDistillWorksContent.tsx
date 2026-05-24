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

function ShapeBadge({ shape, label }: { shape: string; label: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 rounded-md bg-card-bg border border-border px-2 py-1">
      <span className="text-[0.6rem] text-muted">{label}</span>
      <code className="text-[0.65rem] text-blue-500 font-mono font-bold">{shape}</code>
    </div>
  )
}

/* ─── Step-by-step pipeline ─── */
function TokenLevelViz() {
  const [step, setStep] = useState(0)
  const tokens = [
    { text: '2', student_lp: -0.1, teacher_lp: -0.1, kl: 0.0 },
    { text: 'x', student_lp: -0.05, teacher_lp: -0.05, kl: 0.0 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '=', student_lp: -0.1, teacher_lp: -0.1, kl: 0.0 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '7', student_lp: -0.3, teacher_lp: -0.3, kl: 0.0 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '+', student_lp: -0.8, teacher_lp: -0.05, kl: 0.75 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '3', student_lp: -0.3, teacher_lp: -0.1, kl: 0.2 },
    { text: '\n', student_lp: -0.01, teacher_lp: -0.01, kl: 0.0 },
    { text: '2', student_lp: -0.1, teacher_lp: -0.1, kl: 0.0 },
    { text: 'x', student_lp: -0.05, teacher_lp: -0.05, kl: 0.0 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '=', student_lp: -0.1, teacher_lp: -0.1, kl: 0.0 },
    { text: ' ', student_lp: -0.02, teacher_lp: -0.02, kl: 0.0 },
    { text: '1', student_lp: -0.5, teacher_lp: -0.05, kl: 0.45 },
    { text: '0', student_lp: -0.5, teacher_lp: -0.05, kl: 0.45 },
  ]

  useEffect(() => {
    setStep(0)
    const timers = tokens.map((_, i) => setTimeout(() => setStep(i), 300 + i * 250))
    return () => timers.forEach(clearTimeout)
  }, [])

  const klColor = (kl: number) => {
    if (kl === 0) return 'text-emerald-500 bg-emerald-500/5'
    if (kl < 0.3) return 'text-amber-500 bg-amber-500/10'
    return 'text-rose-500 bg-rose-500/10 font-bold'
  }

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">逐 token 的 reverse KL 计算</div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-1.5 px-2 text-muted">Step</th>
              <th className="text-left py-1.5 px-2 text-muted">Token</th>
              <th className="text-right py-1.5 px-2 text-blue-500">log π_student</th>
              <th className="text-right py-1.5 px-2 text-purple-500">log π_teacher</th>
              <th className="text-right py-1.5 px-2 text-text">Reverse KL</th>
              <th className="text-right py-1.5 px-2 text-muted">Advantage</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((t, i) => (
              <tr key={i} className={`transition-all duration-300 ${i < step ? 'opacity-100' : 'opacity-10'}`}>
                <td className="py-1 px-2 text-muted">{i}</td>
                <td className="py-1 px-2 text-text font-bold">{t.text === '\n' ? '\\n' : t.text}</td>
                <td className="py-1 px-2 text-right text-blue-500">{t.student_lp.toFixed(2)}</td>
                <td className="py-1 px-2 text-right text-purple-500">{t.teacher_lp.toFixed(2)}</td>
                <td className={`py-1 px-2 text-right rounded ${klColor(t.kl)}`}>{t.kl.toFixed(2)}</td>
                <td className={`py-1 px-2 text-right ${t.kl > 0 ? 'text-rose-500' : 'text-muted'}`}>{t.kl > 0 ? `-${t.kl.toFixed(2)}` : '0.00'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-xs text-muted">
        Reverse KL = log π_student - log π_teacher。KL 越大 → Student 越偏离 Teacher → 惩罚越大（advantage 越负）。
      </div>
    </Card>
  )
}

/* ─── Tensor shapes ─── */
function TensorShapes() {
  const [expanded, setExpanded] = useState<number | null>(null)
  const tensors = [
    { name: 'trajectories (tokens)', shape: '[batch=256, seq_len=512]', desc: 'Student 采样生成的 token IDs', detail: '每个元素是一个 token ID（整数），范围 [0, vocab_size)。这是 Student 自己生成的轨迹，不是 Teacher 的。' },
    { name: 'sampled_logprobs', shape: '[batch=256, seq_len=512]', desc: 'Student 对自己生成的每个 token 的 log probability', detail: '采样时 Student 模型已经计算过这些值。log π_student(x_t | x_<t)。' },
    { name: 'teacher_logprobs', shape: '[batch=256, seq_len=512]', desc: 'Teacher 对 Student 生成的每个 token 的 log probability', detail: '用 teacher_client.compute_logprobs(trajectories) 计算。把 Student 的轨迹喂给 Teacher，让 Teacher 输出每个位置的条件概率取 log。' },
    { name: 'reverse_kl', shape: '[batch=256, seq_len=512]', desc: '逐 token 的 reverse KL 散度', detail: 'reverse_kl = sampled_logprobs - teacher_logprobs。element-wise 减法。KL=0 表示 Student 和 Teacher 在这一步完全一致。' },
    { name: 'advantages', shape: '[batch=256, seq_len=512]', desc: 'RL 训练用的 advantage', detail: 'advantages = -reverse_kl。取负号是因为我们要最大化 advantage（最小化 KL）。KL 大的位置 advantage 很负，告诉 Student "这步要远离"。' },
    { name: 'loss', shape: '[] (scalar)', desc: 'importance_sampling loss', detail: '用 advantages 和 sampled_logprobs 计算 policy gradient loss，反向传播更新 Student 参数。' },
  ]

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">📐 每一步的 Tensor 形状</div>
      <div className="space-y-2">
        {tensors.map((t, i) => (
          <div key={i} className="rounded-lg border border-border bg-card-bg overflow-hidden">
            <button
              onClick={() => setExpanded(expanded === i ? null : i)}
              className="w-full flex items-center justify-between p-3 text-left"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-medium text-text">{t.name}</span>
                <ShapeBadge shape={t.shape} label="" />
              </div>
              <span className="text-muted text-xs">{expanded === i ? '▲' : '▼'}</span>
            </button>
            {expanded === i && (
              <div className="px-3 pb-3 border-t border-border pt-2">
                <p className="text-xs text-muted">{t.desc}</p>
                <p className="text-xs text-text mt-1">{t.detail}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  )
}

/* ─── Code walkthrough ─── */
function CodeWalkthrough() {
  const [activeLine, setActiveLine] = useState(0)
  const lines = [
    { code: '# 1. 初始化 Teacher 客户端', comment: 'Teacher 是一个独立的模型服务，不需要梯度', color: 'text-muted' },
    { code: 'teacher_client = create_sampling_client(teacher_config)', comment: '', color: 'text-accent' },
    { code: '', comment: '', color: 'text-muted' },
    { code: '# 2. Student 采样轨迹（和 RL 完全一样）', comment: '', color: 'text-muted' },
    { code: 'trajectories = do_group_rollout(student_client, env_group_builder)', comment: 'Student 自己生成答案，不依赖 Teacher', color: 'text-accent' },
    { code: 'sampled_logprobs = trajectories.loss_fn_inputs["logprobs"]', comment: '采样时自动记录了 Student 的 logprobs', color: 'text-accent' },
    { code: '', comment: '', color: 'text-muted' },
    { code: '# 3. Teacher 对 Student 的轨迹打分', comment: '关键步骤：Teacher 看 Student 写的东西', color: 'text-muted' },
    { code: 'teacher_logprobs = teacher_client.compute_logprobs(trajectories)', comment: 'Teacher 对每个 token 输出 log probability', color: 'text-accent' },
    { code: 'reverse_kl = sampled_logprobs - teacher_logprobs', comment: '逐 token 的 KL 散度', color: 'text-accent' },
    { code: 'trajectories["advantages"] = -reverse_kl', comment: '取负号作为 RL 的 advantage', color: 'text-accent' },
    { code: '', comment: '', color: 'text-muted' },
    { code: '# 4. 用 RL 的方式更新 Student', comment: '和标准 RL 训练完全一样的接口', color: 'text-muted' },
    { code: 'training_client.forward_backward(trajectories, loss_fn="importance_sampling")', comment: 'policy gradient 更新', color: 'text-accent' },
  ]

  useEffect(() => {
    setActiveLine(0)
    const timers = lines.map((_, i) => setTimeout(() => setActiveLine(i), 200 + i * 400))
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">💻 完整代码流程（Tinker 实现）</div>
      <pre className="rounded-lg bg-card-bg border border-border p-4 text-xs font-mono overflow-x-auto">
        {lines.map((l, i) => (
          <div key={i} className={`transition-all duration-300 ${i <= activeLine ? 'opacity-100' : 'opacity-20'} ${i === activeLine ? 'bg-blue-500/5 -mx-4 px-4 py-0.5 rounded' : ''}`}>
            <span className="text-muted/30 mr-3 select-none">{String(i + 1).padStart(2, ' ')}</span>
            <span className={l.color}>{l.code}</span>
            {l.comment && <span className="text-muted/60 ml-2"># {l.comment}</span>}
          </div>
        ))}
      </pre>
      <div className="mt-3 text-xs text-muted">
        注意：第 3 步 <code className="text-accent">compute_logprobs</code> 是核心。Teacher 不需要生成任何东西，只需要对已有的 token 序列计算条件概率。这是一次 forward pass，不需要反向传播。
      </div>
    </Card>
  )
}

/* ─── compute_logprobs deep dive ─── */
function ComputeLogprobsDeepDive() {
  const [step, setStep] = useState(0)
  const steps = [
    {
      title: '输入：Student 生成的轨迹',
      desc: '把 Student 生成的 token IDs 作为输入',
      shape: '[256, 512] int64',
      detail: '这些 token 是 Student 自己采样出来的，不是 Teacher 生成的。Teacher 的任务是：如果是我，在每个位置会输出什么概率分布？',
    },
    {
      title: 'Teacher 做一次 forward pass',
      desc: 'Teacher 模型接收整个序列，输出每个位置的 logits',
      shape: '[256, 512, vocab_size=152000]',
      detail: 'Teacher 对每个位置输出一个大小为 vocab_size 的向量，表示在这个位置每个 token 的 logit（未归一化的分数）。',
    },
    {
      title: '取 log softmax',
      desc: '对 logits 做 log softmax，得到 log probability 分布',
      shape: '[256, 512, vocab_size=152000]',
      detail: 'log_softmax(logits) = logits - log(sum(exp(logits)))。这样每个位置的概率分布之和为 1。',
    },
    {
      title: 'Gather：只取 Student 实际生成的 token',
      desc: '从 Teacher 的分布中，取出 Student 实际生成的那个 token 的 log probability',
      shape: '[256, 512]',
      detail: 'Teacher 的分布有 152000 个值，但我们只关心 Student 实际生成的那个 token 对应的 log probability。用 gather 操作提取。',
    },
  ]

  useEffect(() => {
    setStep(0)
    const timers = steps.map((_, i) => setTimeout(() => setStep(i), 400 + i * 1200))
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <Card>
      <div className="text-sm font-semibold text-text mb-3">🔬 compute_logprobs 内部发生了什么？</div>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <div key={i} className={`transition-all duration-500 ${i <= step ? 'opacity-100' : 'opacity-20'}`}>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-5 h-5 rounded-full bg-blue-500 text-white flex items-center justify-center text-[0.6rem] font-bold">{i + 1}</div>
              <span className="text-sm font-medium text-text">{s.title}</span>
            </div>
            <div className="ml-7">
              <p className="text-xs text-muted">{s.desc}</p>
              <ShapeBadge shape={s.shape} label="shape" />
              <p className="text-xs text-muted mt-1">{s.detail}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 rounded-lg bg-blue-500/5 border border-blue-500/20 p-3 text-xs text-muted">
        <strong className="text-text">关键点：</strong>Teacher 不需要生成任何新 token。它只是对已有的序列做一次 forward pass，计算条件概率。这比采样便宜得多（不需要自回归循环）。
      </div>
    </Card>
  )
}

/* ─── Main ─── */
export default function HowOnPolicyDistillWorksContent() {
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
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.3rem,3vw,1.8rem)] font-bold text-text m-0">On-Policy Distillation 每一步在做什么？</h1>
        <p className="text-muted mt-2 text-sm">数据维度、tensor shape、代码流程——没有比喻，只有实现</p>
      </div>

      {/* 1. The core question */}
      <Section>
        <div className="rounded-2xl border border-blue-500/30 bg-blue-500/5 p-5">
          <div className="text-sm font-semibold text-text mb-2">🎯 核心问题：Teacher 怎么"手把手教"？</div>
          <p className="text-sm text-muted leading-relaxed">
            Teacher 不是给一个分数，而是对 Student 生成的<strong className="text-text">每一个 token</strong>，计算"如果是我，我会给这个 token 多高的概率"。
            这个计算通过 <code className="text-accent">compute_logprobs</code> 实现——Teacher 对 Student 的轨迹做一次 forward pass，输出每个位置的条件 log probability。
          </p>
        </div>
      </Section>

      {/* 2. Tensor shapes */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">📐 数据维度（Tensor Shapes）</h2>
        </div>
        <TensorShapes />
      </Section>

      {/* 3. compute_logprobs deep dive */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-purple-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔬 compute_logprobs 内部机制</h2>
        </div>
        <ComputeLogprobsDeepDive />
      </Section>

      {/* 4. Token-level KL */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔢 逐 token 的 Reverse KL</h2>
        </div>
        <TokenLevelViz />
        <div className="mt-4 text-center">
          <a
            href="./article.html?id=reverse-kl"
            className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/5 text-emerald-500 px-4 py-2 text-sm no-underline transition-colors hover:bg-emerald-500/10"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20V10" />
              <path d="M18 20V4" />
              <path d="M6 20v-4" />
            </svg>
            深入：Reverse KL 到底是什么？（交互式可视化）
          </a>
        </div>
      </Section>

      {/* 5. Code walkthrough */}
      <Section>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">💻 完整代码流程</h2>
        </div>
        <CodeWalkthrough />
      </Section>

      {/* 6. Why this works */}
      <Section>
        <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6">
          <div className="text-sm font-semibold text-text mb-3">🎯 总结：为什么这个设计可行？</div>
          <div className="space-y-3 text-sm text-muted leading-relaxed">
            <p><strong className="text-text">1. Teacher 不需要生成新内容</strong> — 它只是对已有的序列做 forward pass，计算条件概率。这比采样快得多。</p>
            <p><strong className="text-text">2. Student 的轨迹是 on-policy 的</strong> — 数据来自 Student 自己，所以学到的东西直接适用于 Student 实际会遇到的状态。</p>
            <p><strong className="text-text">3. 反馈是 dense 的</strong> — 每个 token 都有一个 KL 值，不是只有一个最终答案的对/错。512 个 token = 512 个学习信号。</p>
            <p><strong className="text-text">4. 代码改动极小</strong> — 在标准 RL 脚本中，只需要把 reward model 换成 teacher 的 compute_logprobs，其他完全一样。</p>
          </div>
        </div>
      </Section>
    </>
  )
}

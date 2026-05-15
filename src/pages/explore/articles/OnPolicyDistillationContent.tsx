import { useState, useEffect } from 'react'

export default function OnPolicyDistillationContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <style>{`
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin-slow { animation: spin-slow 6s linear infinite; }
      `}</style>

      <div className={`text-center mb-10 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-blue-500/10 text-blue-500 mb-4 spin-slow">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
          </svg>
        </div>
        <h1 className="text-[clamp(1.5rem,3.5vw,2rem)] font-bold text-text m-0">On Policy Distillation</h1>
        <p className="text-muted mt-2 text-sm">让 student 在探索中实时向 teacher 学习</p>
        <span className="inline-block mt-2 text-[0.65rem] text-muted/40">25/05/15</span>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🤔 这是什么？</h2>
        </div>
        <div className="rounded-2xl border border-border bg-card-bg p-5 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <p className="text-sm text-text leading-relaxed">
            在强化学习中，<strong>distillation（蒸馏）</strong>就是把一个大的 teacher 模型的知识压缩到一个小 student 模型里。
          </p>
          <p className="text-sm text-text leading-relaxed mt-3">
            <strong>On-policy distillation</strong> 的特殊之处在于：student 不是从别人收集好的数据集里学，而是自己在环境中探索、遇到状态、然后问 teacher "这一步该怎么走？"——边探索边学。
          </p>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">📊 Off-Policy vs On-Policy</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
            <div className="text-lg font-bold text-rose-500 mb-2">Off-Policy</div>
            <div className="flex flex-col items-center gap-2 text-sm text-muted">
              <div className="w-full rounded-lg bg-rose-500/10 p-3 text-center">📦 固定数据集</div>
              <span className="text-rose-400">↓</span>
              <div className="w-full rounded-lg bg-rose-500/10 p-3 text-center">🎓 Student 离线学习</div>
              <span className="text-rose-400">↓</span>
              <div className="w-full rounded-lg bg-rose-500/10 p-3 text-center">✅ 一次训练，不再交互</div>
            </div>
          </div>
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="text-lg font-bold text-emerald-500 mb-2">On-Policy</div>
            <div className="flex flex-col items-center gap-2 text-sm text-muted">
              <div className="w-full rounded-lg bg-emerald-500/10 p-3 text-center">🌍 Student 探索环境</div>
              <span className="text-emerald-400">↓</span>
              <div className="w-full rounded-lg bg-emerald-500/10 p-3 text-center">💬 遇到问题问 Teacher</div>
              <span className="text-emerald-400">↓</span>
              <div className="w-full rounded-lg bg-emerald-500/10 p-3 text-center">🔄 学到后继续探索</div>
            </div>
            <div className="mt-3 text-center">
              <span className="inline-block rounded-full bg-emerald-500/10 text-emerald-500 text-xs px-2 py-0.5 font-medium">循环，持续改进</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔥 为什么它更好？</h2>
        </div>
        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-lg">🎯</div>
            <div>
              <h4 className="font-semibold text-text text-sm">所见即所学</h4>
              <p className="text-xs text-muted mt-0.5">Student 从自己实际遇到的状态中学习，而不是从 teacher 的"老数据"里学。学的东西更贴合自己的实际能力。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-lg">🔄</div>
            <div>
              <h4 className="font-semibold text-text text-sm">自适应的学习循环</h4>
              <p className="text-xs text-muted mt-0.5">Student 变强了 → 探索到更难的区域 → 从 teacher 学到更多 → 继续变强。形成正向飞轮。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-lg">📈</div>
            <div>
              <h4 className="font-semibold text-text text-sm">训练更稳定</h4>
              <p className="text-xs text-muted mt-0.5">不需要维护一个巨大的 replay buffer，也不会因为数据分布偏移导致训练崩溃。训练过程更平滑。</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔄 完整流程</h2>
        </div>
        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="flex flex-col items-center gap-3">
            <div className="w-full rounded-xl bg-blue-500/10 p-3 text-center">
              <span className="text-lg">🌍</span>
              <p className="text-sm font-medium text-text mt-1">Student 与环境交互</p>
              <p className="text-xs text-muted">在当前 policy 下采样轨迹数据</p>
            </div>
            <span className="text-muted text-lg">↓</span>
            <div className="w-full rounded-xl bg-emerald-500/10 p-3 text-center">
              <span className="text-lg">💬</span>
              <p className="text-sm font-medium text-text mt-1">Teacher 给出指导</p>
              <p className="text-xs text-muted">Teacher 对每个 state 输出 action 分布</p>
            </div>
            <span className="text-muted text-lg">↓</span>
            <div className="w-full rounded-xl bg-amber-500/10 p-3 text-center">
              <span className="text-lg">📚</span>
              <p className="text-sm font-medium text-text mt-1">Student 模仿学习</p>
              <p className="text-xs text-muted">最小化 student 和 teacher 输出之间的 KL 散度</p>
            </div>
            <span className="text-muted text-lg">↓</span>
            <div className="w-full rounded-xl bg-violet-500/10 p-3 text-center">
              <span className="text-lg">🔄</span>
              <p className="text-sm font-medium text-text mt-1">更新 policy，继续探索</p>
              <p className="text-xs text-muted">Student 带着新知识继续与环境交互 → 循环</p>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6 text-center">
        <div className="text-2xl mb-2">💡</div>
        <p className="text-text font-medium">
          On-policy distillation 的核心洞察：<br />让 student 在"做中学"，而不是"看中学"。
        </p>
        <p className="text-sm text-muted mt-1">这在 LLM 对齐、机器人学习等领域正变得越来越重要。</p>
      </div>
    </>
  )
}

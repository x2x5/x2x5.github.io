import { useState, useEffect } from 'react'

const channelTypes = [
  { emoji: '🔌', title: 'API 中转站', desc: '聚合转发大模型接口，按量计费，免翻墙' },
  { emoji: '🆓', title: '免费额度', desc: '各平台注册送额度、白嫖 API 调用' },
  { emoji: '📦', title: '开源平替', desc: '可自部署的开源模型、工具，替代收费服务' },
  { emoji: '💳', title: '限时优惠', desc: '闪购、折扣码、学生优惠、团购' },
  { emoji: '🤖', title: 'AI 工具箱', desc: '各类 AI 工具合集网站、导航站' },
  { emoji: '📡', title: '镜像/代理', desc: '国内可直接访问的镜像站、反向代理' },
]

const pipelineStages = [
  { emoji: '📡', title: '信息源', desc: 'Telegram 频道、GitHub、Twitter、论坛、共享文档' },
  { emoji: '🗂️', title: '采集工具', desc: 'RSS 订阅、GitHub Watch、TG Bot、爬虫、IFTTT' },
  { emoji: '🔍', title: '筛选整理', desc: '标签分类、可用性测试、比价、去重' },
  { emoji: '📤', title: '输出分发', desc: '汇总成日报/周报、自己的频道、博客、表格' },
]

const tools = [
  { emoji: '✈️', title: 'Telegram 频道', desc: '关注聚合类频道，第一时间获取渠道推送' },
  { emoji: '⭐', title: 'GitHub Watch', desc: 'Watch 关键词仓库，开启 Release 通知' },
  { emoji: '📡', title: 'RSS 订阅', desc: '用 RSSHub 将各类网页转为可订阅源' },
  { emoji: '🐦', title: 'X/Twitter List', desc: '创建 List 关注行业爆料号，开通知铃' },
  { emoji: '📄', title: '共享文档', desc: '飞书/Notion 多人协作表格，集体维护' },
  { emoji: '🤖', title: '自建爬虫', desc: 'Python + GitHub Actions 定时巡检，变更告警' },
]

function ChannelCard({ emoji, title, desc }: { emoji: string; title: string; desc: string }) {
  return (
    <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3 transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
      <div className="w-10 h-10 rounded-xl bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-xl">
        {emoji}
      </div>
      <div className="min-w-0">
        <h4 className="font-semibold text-text text-sm">{title}</h4>
        <p className="text-xs text-muted mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

function PipelineCard({ emoji, title, desc, index }: { emoji: string; title: string; desc: string; index: number }) {
  return (
    <div className="flex flex-col items-center text-center">
      <div className="w-14 h-14 rounded-2xl bg-blue-500/10 text-blue-500 flex items-center justify-center text-2xl mb-2">
        {emoji}
      </div>
      <span className="text-xs font-bold text-blue-500 mb-0.5">STEP {index + 1}</span>
      <h4 className="font-semibold text-text text-sm">{title}</h4>
      <p className="text-xs text-muted mt-1 leading-relaxed">{desc}</p>
    </div>
  )
}

export default function ChannelContent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <>
      <style>{`
        @keyframes sonar-ring {
          0% { transform: scale(0.85); opacity: 0.8; }
          100% { transform: scale(1.6); opacity: 0; }
        }
        .sonar-ring { animation: sonar-ring 2s ease-out infinite; }
      `}</style>

      <div className={`text-center mb-12 transition-all duration-700 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="relative w-24 h-24 mx-auto mb-5">
          <div className="absolute inset-0 rounded-full border-2 border-blue-500/30 sonar-ring" />
          <div className="absolute inset-0 rounded-full border-2 border-blue-500/20 sonar-ring" style={{ animationDelay: '0.7s' }} />
          <div className="absolute inset-0 rounded-full border-2 border-blue-500/10 sonar-ring" style={{ animationDelay: '1.4s' }} />
          <div className="relative w-full h-full rounded-full bg-blue-500/10 text-blue-500 flex items-center justify-center text-3xl">
            📡
          </div>
        </div>
        <h1 className="text-[clamp(1.6rem,4vw,2.2rem)] font-bold text-text m-0 leading-tight">
          怎么及时汇总和更新各种渠道？
        </h1>
        <p className="text-muted mt-2 text-base">从信息差到信息流，搭建你的渠道雷达</p>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-blue-500" />
          <h2 className="text-lg font-semibold text-text m-0">🗺️ 先看清：有哪些渠道类型？</h2>
        </div>
        <p className="text-sm text-muted mb-4">知己知彼，先搞清楚你在搜集什么</p>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          {channelTypes.map((c) => (
            <ChannelCard key={c.title} {...c} />
          ))}
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-emerald-500" />
          <h2 className="text-lg font-semibold text-text m-0">🔁 再理解：采集流水线</h2>
        </div>
        <p className="text-sm text-muted mb-4">从发现到发布，一条完整的渠道处理链路</p>

        <div className="rounded-2xl border border-border bg-card-bg p-6 shadow-[0_10px_30px_var(--color-card-shadow)]">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-6">
            {pipelineStages.map((s, i) => (
              <>
                {i > 0 && <div className="hidden sm:flex items-center justify-center text-2xl text-muted">→</div>}
                <PipelineCard key={s.title} {...s} index={i} />
              </>
            ))}
          </div>
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-amber-500" />
          <h2 className="text-lg font-semibold text-text m-0">🛠️ 实操：推荐工具组合</h2>
        </div>
        <p className="text-sm text-muted mb-4">每个工具解决一个环节，组合起来就是你的渠道雷达</p>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          {tools.map((t) => (
            <div key={t.title} className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-3 transition-all duration-200 hover:-translate-y-[2px] hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-xl">
                {t.emoji}
              </div>
              <div className="min-w-0">
                <h4 className="font-semibold text-text text-sm">{t.title}</h4>
                <p className="text-xs text-muted mt-0.5 leading-relaxed">{t.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-full bg-violet-500" />
          <h2 className="text-lg font-semibold text-text m-0">⚡ 自动化方案：从手动到自动</h2>
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-4 transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">1</div>
            <div className="min-w-0">
              <h4 className="font-semibold text-text text-sm">锁定 3-5 个核心信息源</h4>
              <p className="text-xs text-muted mt-0.5">贪多嚼不烂。先找到最活跃的 TG 频道、GitHub 仓库、Twitter 账号，天天刷就够了。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-4 transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">2</div>
            <div className="min-w-0">
              <h4 className="font-semibold text-text text-sm">配置 RSS + 通知</h4>
              <p className="text-xs text-muted mt-0.5">RSSHub 把 TG channel 转 RSS，Feedly 聚合阅读；GitHub 开 Watch + Release notification；Twitter 开通知。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-4 transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">3</div>
            <div className="min-w-0">
              <h4 className="font-semibold text-text text-sm">建立分类 + 验证机制</h4>
              <p className="text-xs text-muted mt-0.5">按类型/可信度打标签。新渠道先小规模测试，确认可用再收录，避免浪费时间。</p>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card-bg p-4 flex items-start gap-4 transition-all duration-200 hover:shadow-[0_10px_30px_var(--color-card-shadow-hover)]">
            <div className="w-8 h-8 rounded-lg bg-violet-500/10 text-violet-500 flex items-center justify-center flex-shrink-0 text-sm font-bold">4</div>
            <div className="min-w-0">
              <h4 className="font-semibold text-text text-sm">定期输出 + 反哺社区</h4>
              <p className="text-xs text-muted mt-0.5">每周整理成汇总文档发出去。越分享，别人越会给你投喂新渠道，形成正向循环。</p>
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-gradient-to-br from-blue-500/5 to-emerald-500/5 p-6 text-center">
        <div className="text-2xl mb-2">🌐</div>
        <p className="text-text font-medium">
          信息差 = 别人不知道你知道。而最好的破除方式，是让信息流动起来。
        </p>
        <p className="text-sm text-muted mt-1">先建一个最小闭环：1 个信息源 → 1 个工具 → 每日扫一遍。跑通了再加。</p>
      </div>
    </>
  )
}

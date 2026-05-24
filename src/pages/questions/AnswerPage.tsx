import { useState } from 'react'
import { useTheme } from '../../hooks/useTheme'
import NoIdeaContent from './answers/NoIdeaContent'
import ChannelContent from './answers/ChannelContent'
import BlogVsAIContent from './answers/BlogVsAIContent'
import NoRouterContent from './answers/NoRouterContent'
import WhyHtmlFilesContent from './answers/WhyHtmlFilesContent'
import DevErrorBehaviorContent from './answers/DevErrorBehaviorContent'
import WhatIsEnvironmentContent from './answers/WhatIsEnvironmentContent'

const questionTimes: Record<string, string> = {
  'no-idea': '25/05/15 05:30:00',
  'channel-collection': '25/05/15 05:38:00',
  'blog-vs-ai': '25/05/15 05:47:00',
  'no-router': '26/05/15 12:57:00',
  'why-html-files': '26/05/15 12:59:00',
  'dev-error-behavior': '26/05/15 13:01:00',
  'what-is-environment': '26/05/15 13:05:00',
}

export default function AnswerPage() {
  const { theme, setTheme } = useTheme()
  const [params] = useState(() => new URLSearchParams(window.location.search))
  const id = params.get('id') || 'no-idea'

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[800px] mx-auto">
        <div className="flex items-center justify-between mb-6">
          <a
            href="./questions.html"
            className="inline-flex items-center gap-2 rounded-full border border-border bg-card-bg text-muted px-4 py-2 no-underline transition-colors hover:text-text hover:border-accent text-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5" />
              <path d="m12 19-7-7 7-7" />
            </svg>
            返回提问
          </a>
          <button
            type="button"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="inline-flex items-center justify-center rounded-full border border-border bg-card-bg text-muted w-9 h-9 cursor-pointer transition-colors hover:text-text hover:border-accent"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
            )}
          </button>
        </div>

        <div className="text-center mb-8">
          <span className="text-[0.65rem] text-muted/40">提问于 {questionTimes[id] || '--'}</span>
        </div>

        {id === 'dev-error-behavior' && <DevErrorBehaviorContent />}
        {id === 'why-html-files' && <WhyHtmlFilesContent />}
        {id === 'no-router' && <NoRouterContent />}
        {id === 'no-idea' && <NoIdeaContent />}
        {id === 'channel-collection' && <ChannelContent />}
        {id === 'blog-vs-ai' && <BlogVsAIContent />}
        {id === 'what-is-environment' && <WhatIsEnvironmentContent />}

        <div className="text-center mt-10 mb-6">
          <a
            href="./questions.html"
            className="inline-flex items-center gap-2 text-sm text-muted hover:text-text transition-colors no-underline"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5" />
              <path d="m12 19-7-7 7-7" />
            </svg>
            返回问题列表
          </a>
        </div>
      </main>
    </div>
  )
}

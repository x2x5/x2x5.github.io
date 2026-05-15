import { useState } from 'react'
import { useTheme } from '../../hooks/useTheme'
import OnPolicyDistillationContent from './articles/OnPolicyDistillationContent'

export default function ArticlePage() {
  const { theme, setTheme } = useTheme()
  const [params] = useState(() => new URLSearchParams(window.location.search))
  const id = params.get('id') || 'on-policy-distillation'

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <main className="w-full max-w-[720px] mx-auto">
        <div className="flex items-center justify-between mb-6">
          <a
            href="./explore.html"
            className="inline-flex items-center gap-2 rounded-full border border-border bg-card-bg text-muted px-4 py-2 no-underline transition-colors hover:text-text hover:border-accent text-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5" />
              <path d="m12 19-7-7 7-7" />
            </svg>
            返回探索
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

        {id === 'on-policy-distillation' && <OnPolicyDistillationContent />}

        <div className="text-center mt-10 mb-6">
          <a
            href="./explore.html"
            className="inline-flex items-center gap-2 text-sm text-muted hover:text-text transition-colors no-underline"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5" />
              <path d="m12 19-7-7 7-7" />
            </svg>
            返回探索列表
          </a>
        </div>
      </main>
    </div>
  )
}

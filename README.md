

# README for AI | [README for Human](https://x2x5.github.io/readme.html)

Personal homepage for x2x5. React 19 + TypeScript + Vite 8 + Tailwind CSS v4. Multi-page SPA (no router). Auto-deployed to GitHub Pages from `main`.

| | |
|---|---|
| Live | https://x2x5.github.io |
| Deploy | `.github/workflows/deploy.yml` → GH Pages |

## Quick Start

```bash
npm install
npm run dev        # dev server → http://localhost:5173
npm run build      # production → dist/
npm run preview    # serve dist/ locally
```

## Architecture

Every HTML entry in the project root is an independent page. No `react-router`. Navigation uses plain `<a href>`.

```
[page].html → src/[page].tsx (entry) → src/pages/... (component)
```

**Page mapping:**

| HTML | Entry TSX | Component |
|---|---|---|
| `index.html` | `src/main.tsx` | `src/App.tsx` |
| `readme.html` | `src/readme.tsx` | `src/ReadmePage.tsx` |
| `commits.html` | `src/commits.tsx` | `src/CommitsPage.tsx` |
| `questions.html` | `src/questions.tsx` | `src/pages/questions/QuestionsPage.tsx` |
| `answer.html` | `src/answer.tsx` | `src/pages/questions/AnswerPage.tsx` |
| `explore.html` | `src/explore.tsx` | `src/pages/explore/ExplorePage.tsx` |
| `article.html` | `src/article.tsx` | `src/pages/explore/ArticlePage.tsx` |

## Directory Structure

```
src/
  hooks/useTheme.ts            Shared theming hook (import, don't inline)
  pages/
    questions/                 Hub + detail pattern
      QuestionsPage.tsx        Question card grid
      AnswerPage.tsx           Routes to answer by ?id=
      answers/                 One file per question answer
    explore/                   Same pattern
      ExplorePage.tsx          Article list, grouped by month
      ArticlePage.tsx          Routes to article by ?id=
      articles/                One file per article
  App.tsx                      Homepage (cards, i18n dict, theme/language state)
  ReadmePage.tsx               Visual human-facing README
  CommitsPage.tsx              Commit history
  index.css                    Tailwind entry + CSS custom properties + dark mode
  [page].tsx                   Entry points (one per HTML)
vite.config.ts                 Multi-page build config
tsconfig.json / tsconfig.*.json
package.json
```

## Adding a New Page

```ts
// 1. create page.html (copy pattern from existing)
// 2. create src/page.tsx (entry, renders component)
// 3. create src/pages/.../PageComponent.tsx
// 4. register in vite.config.ts → rollupOptions.input
// 5. keep cssCodeSplit: false (required for multi-page CSS)
```

## Common Modification Patterns

| Task | Location |
|---|---|
| Change homepage cards | `src/App.tsx` → `tools` / `supervisors` / `thoughts` arrays |
| Add i18n text | `src/App.tsx` → `I18N` dictionary (zh + en) |
| Add a question | `src/pages/questions/QuestionsPage.tsx` (card) + `src/pages/questions/answers/` (answer content) + `AnswerPage.tsx` (route) |
| Add an explore article | `src/pages/explore/ExplorePage.tsx` (list entry) + `src/pages/explore/articles/` (content) + `ArticlePage.tsx` (route) |
| Change theme colors | `src/index.css` → `@theme` block + `html.dark` overrides |
| Change accent/background | Edit CSS variables in `src/index.css` |
| Theme hook | Already extracted to `src/hooks/useTheme.ts` — do not inline in new pages |
| Card color palette | `colorMap` in each page (must match Tailwind color names) |

## Build & Deploy

- `npm run build` outputs to `dist/`. **Do not edit dist/ directly** — it is regenerated.
- Push to `main` → GitHub Actions runs `npm ci && npm run build` and deploys `dist/` to Pages.
- GitHub repo Settings → Pages → Source must be set to **GitHub Actions**.

## Constraints & Gotchas

- **cssCodeSplit: false** is required. Without it, shared CSS is only injected into the first entry, leaving other pages unstyled.
- **No react-router**. All navigation is `<a href>`. Use relative paths like `./page.html`.
- **macOS filesystem is case-insensitive**. Do not create files that differ only by case (e.g. `Readme.tsx` + `readme.tsx`). Use distinct names.
- **Tailwind v4** uses `@import "tailwindcss"` + `@theme {}`. No `tailwind.config.js`. No `@apply` in config files.
- **Dark mode**: `html.dark` class toggle via `useTheme()`. Inline `<script>` in each HTML file sets it before React hydrates (prevents FOUC).
- **No test framework** configured. Add Vitest + React Testing Library if needed.
- **`dist/`** is gitignored.

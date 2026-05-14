<p align="center">
  <b>README for AI</b>
  <span style="color:#999;margin:0 8px;">|</span>
  <a href="https://x2x5.github.io/readme.html">给人看的 README</a>
</p>

<h1 align="center">x2x5.github.io</h1>

<p align="center">
  <a href="https://x2x5.github.io/" style="text-decoration:none;">
    <span style="display:inline-block;padding:4px 12px;border-radius:999px;background:#365fcf;color:#fff;font-size:12px;">Live Site</span>
  </a>
  <span style="display:inline-block;padding:4px 12px;border-radius:999px;background:#10b981;color:#fff;font-size:12px;margin-left:6px;">React 19</span>
  <span style="display:inline-block;padding:4px 12px;border-radius:999px;background:#0ea5e9;color:#fff;font-size:12px;margin-left:6px;">TypeScript</span>
  <span style="display:inline-block;padding:4px 12px;border-radius:999px;background:#8b5cf6;color:#fff;font-size:12px;margin-left:6px;">Tailwind v4</span>
</p>

<p align="center" style="color:#666;font-size:14px;">
  Personal homepage for x2x5 Research · Vite-based multi-page React app · Auto-deployed to GitHub Pages
</p>

---

## Quick Reference

| | |
|:---|:---|
| **Live URL** | https://x2x5.github.io/ |
| **Custom Domain** | x2x5.top (via CNAME) |
| **Framework** | React 19 + TypeScript |
| **Build Tool** | Vite 8 |
| **Styling** | Tailwind CSS v4 |
| **Deploy** | GitHub Actions → GitHub Pages |
| **Default Branch** | `main` |

---

## Project Structure

```
x2x5.github.io/
├── .github/workflows/deploy.yml    # CI/CD: build + deploy
├── src/
│   ├── App.tsx                     # Homepage (cards, i18n, theme)
│   ├── ReadmePage.tsx              # README page (visual tutorial)
│   ├── index.css                   # Tailwind entry + theme vars + dark mode
│   ├── main.tsx                    # Entry: index.html
│   ├── readme.tsx                  # Entry: readme.html
│   └── vite-env.d.ts              # Vite client types
├── index.html                      # Vite entry: homepage
├── readme.html                     # Vite entry: README page
├── vite.config.ts                  # Multi-page config
├── tsconfig.json / tsconfig.*.json # TypeScript configs
├── package.json                    # Dependencies
├── .gitignore                      # node_modules/, dist/
└── CNAME                           # x2x5.top
```

---

## Architecture Decisions

### Multi-page without a router

Two independent HTML entry points (`index.html` + `readme.html`), each with its own React root. Navigation uses plain `<a href>` links. No `react-router` needed.

```ts
// vite.config.ts
build: {
  cssCodeSplit: false,  // shared CSS injected into both entries
  rollupOptions: {
    input: {
      main: 'index.html',
      readme: 'readme.html',
    },
  },
}
```

> **Why `cssCodeSplit: false`?**  
> Without this, Vite extracts shared CSS into one file but only injects the `<link>` into the "primary" entry (`index.html`), leaving `readme.html` unstyled.

### Theming via CSS custom properties

Light/dark mode uses CSS variables (`--color-bg`, `--color-text`, etc.) scoped under `html.dark`. The `dark` class is toggled on `<html>` via a `useTheme()` hook. An inline script in each HTML entry sets the class before React hydrates to prevent FOUC.

### i18n (hardcoded dictionaries)

Two languages: `zh` (default) and `en`. Translation dictionaries live in `src/App.tsx`. Language preference persists to `localStorage`.

---

## Available Scripts

```bash
npm install       # first time only
npm run dev       # dev server → http://localhost:5173/
npm run build     # production build → dist/
npm run preview   # preview dist/ locally
```

---

## Deployment Pipeline

Push to `main` triggers `.github/workflows/deploy.yml`:

1. `actions/checkout@v4`
2. `actions/setup-node@v4` (Node 22)
3. `npm ci`
4. `npm run build`
5. `actions/upload-pages-artifact@v3` (path: `./dist`)
6. `actions/deploy-pages@v4`

Total time ~30s. Page source in repo Settings → Pages must be set to **GitHub Actions**.

---

## Adding a New Page

1. Create a new HTML entry (e.g. `about.html`) + TSX entry (e.g. `src/about.tsx`).
2. Add to `vite.config.ts`:
   ```ts
   input: {
     main: 'index.html',
     readme: 'readme.html',
     about: 'about.html',
   }
   ```
3. Keep `cssCodeSplit: false` so shared styles are injected.

---

## Notes for Future AI Assistants

- **Do not modify `dist/` directly** — it is regenerated on every build.
- **Page content**: edit `src/App.tsx` (homepage) or `src/ReadmePage.tsx` (README page).
- **Styling**: edit `src/index.css`.
- **New cards**: add entries to `tools` or `supervisors` arrays in `src/App.tsx`.
- **Tailwind v4 syntax**: uses `@import "tailwindcss"` + `@theme { --color-*: ... }` — no `tailwind.config.js`.
- **Dark mode**: manual `html.dark` class toggle, not `darkMode: 'class'` in a config file.
- **macOS gotcha**: the filesystem is case-insensitive. Do not create both `readme.tsx` and `Readme.tsx` — they are the same file. Use distinct names like `ReadmePage.tsx`.
- **No test framework** configured. Add Vitest + React Testing Library if needed.

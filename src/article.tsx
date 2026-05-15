import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import ArticlePage from './pages/explore/ArticlePage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ArticlePage />
  </StrictMode>,
)

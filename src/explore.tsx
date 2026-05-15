import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import ExplorePage from './pages/explore/ExplorePage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ExplorePage />
  </StrictMode>,
)

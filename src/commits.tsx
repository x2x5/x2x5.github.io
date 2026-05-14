import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import CommitsPage from './CommitsPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <CommitsPage />
  </StrictMode>,
)

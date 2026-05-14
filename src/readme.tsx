import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import ReadmePage from './ReadmePage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ReadmePage />
  </StrictMode>,
)

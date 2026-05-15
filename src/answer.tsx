import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AnswerPage from './pages/questions/AnswerPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AnswerPage />
  </StrictMode>,
)

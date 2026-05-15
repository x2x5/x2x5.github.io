import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import QuestionsPage from './pages/questions/QuestionsPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QuestionsPage />
  </StrictMode>,
)

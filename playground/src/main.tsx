import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const rootEl: HTMLElement = document.getElementById('root') as HTMLElement;
if (!rootEl) throw new Error('element #root not found');

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

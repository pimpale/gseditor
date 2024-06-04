import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

import 'bootstrap/dist/css/bootstrap.min.css';
import GaussianEditor from './components/gaussian_renderer.tsx';


ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <GaussianEditor width={400} height={400} />
    </React.StrictMode>,
)

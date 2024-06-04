import React from 'react'
import ReactDOM from 'react-dom/client'
import GaussianEditor from './components/gaussian_renderer.tsx';

import 'bootstrap/dist/css/bootstrap.min.css';


ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <GaussianEditor width={400} height={400} />
    </React.StrictMode>,
)

import React, { lazy, Suspense } from 'react';
import ReactDOM from 'react-dom/client';

// Public builds open the self-contained historical demo. The original API
// dashboard remains available with `npm run dev:live` / `build:live`.
const App = lazy(() => import.meta.env.VITE_DEMO_MODE === 'false'
  ? import('./App')
  : import('./demo/DemoApp'));

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Suspense fallback={<div role="status" style={{ padding: 32 }}>Opening finAlyse…</div>}>
      <App />
    </Suspense>
  </React.StrictMode>,
);

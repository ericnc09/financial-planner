import React, { useEffect, useState } from 'react';
import { api } from './api/client';
import { Dashboard } from './components/Dashboard';
import type { DashboardData } from './types';

const EMPTY_DATA: DashboardData = {
  signals: [],
  macro: null,
};

const App: React.FC = () => {
  const [data, setData] = useState<DashboardData>(EMPTY_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getDashboard();
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch data');
      console.error('Dashboard fetch error:', e);
    } finally {
      setLoading(false);
    }
  };

  const runPipeline = async () => {
    try {
      await api.triggerPipeline();
      setTimeout(fetchData, 2000);
    } catch (e) {
      console.error('Pipeline trigger error:', e);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (error && data === EMPTY_DATA) {
    return (
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#e1e4e8', marginBottom: 16 }}>Smart Money Follows</h1>
        <div style={{
          background: '#161b22', borderRadius: 12, padding: 40, border: '1px solid #30363d', textAlign: 'center',
        }}>
          <p style={{ color: '#f59e0b', marginBottom: 12 }}>Unable to connect to API</p>
          <p style={{ color: '#8b949e', fontSize: 13, marginBottom: 16 }}>{error}</p>
          <p style={{ color: '#8b949e', fontSize: 12 }}>
            Make sure the API server is running: <code style={{ color: '#c9d1d9' }}>uvicorn src.api.api:app --port 8000</code>
          </p>
          <button onClick={fetchData} style={{
            marginTop: 16, padding: '8px 20px', borderRadius: 8, border: 'none',
            background: '#6366f1', color: '#fff', cursor: 'pointer', fontSize: 13,
          }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return <Dashboard data={data} onRefresh={fetchData} onRunPipeline={runPipeline} loading={loading} />;
};

export default App;

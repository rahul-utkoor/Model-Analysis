import { BarChart3, CheckCircle2, GitBranch, Layers } from 'lucide-react';
import type { ReactNode } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { CoverageResponse } from '../types';
import { StatusBadge } from './StatusBadge';

export function CoverageDashboard({ coverage }: { coverage?: CoverageResponse }) {
  const rows = coverage?.table ?? [];
  const complete = rows.filter((row) => row.status === 'complete').length;
  const totalValid = rows.reduce((sum, row) => sum + Number(row.valid_plans || 0), 0);
  const totalLayers = rows.reduce((sum, row) => sum + Number(row.layers || 0), 0);
  const totalSubgraphs = rows.reduce((sum, row) => sum + Number(row.subgraphs || 0), 0);

  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Coverage Dashboard</h2>
        <p>Cross-model static support and validated symbolic MLP/FFN plans.</p>
      </div>
      <div className="metric-grid">
        <Metric icon={<CheckCircle2 />} label="Complete models" value={complete} />
        <Metric icon={<GitBranch />} label="Valid plans" value={totalValid} />
        <Metric icon={<Layers />} label="Layers / blocks" value={totalLayers} />
        <Metric icon={<BarChart3 />} label="Subgraphs" value={totalSubgraphs} />
      </div>
      <div className="chart-row">
        <div className="chart-card">
          <h3>Valid plans by model</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" hide />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="valid_plans" fill="#2f7d68" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>Subgraphs by model</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" hide />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="subgraphs" fill="#4666a5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Status</th>
            <th>Layers</th>
            <th>Subgraphs</th>
            <th>Safe</th>
            <th>Plans</th>
            <th>Valid</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.model}>
              <td>{row.model}</td>
              <td><StatusBadge value={row.status} tone={row.status} /></td>
              <td>{row.layers}</td>
              <td>{row.subgraphs}</td>
              <td>{row.safe}</td>
              <td>{row.plans}</td>
              <td>{row.valid_plans}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

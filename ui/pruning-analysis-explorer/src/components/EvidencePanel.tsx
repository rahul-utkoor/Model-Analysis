export function EvidencePanel({ title, rows, columns }: { title: string; rows: any[]; columns: string[] }) {
  return (
    <div className="evidence-panel">
      <h3>{title}</h3>
      {rows?.length ? (
        <table className="data-table dense">
          <thead>
            <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>
                {columns.map((column) => <td key={column}>{formatValue(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">No evidence in this section.</p>
      )}
    </div>
  );
}

function formatValue(value: unknown) {
  if (Array.isArray(value)) return value.length ? value.map((item) => typeof item === 'object' ? JSON.stringify(item) : String(item)).join(', ') : '';
  if (value && typeof value === 'object') return JSON.stringify(value);
  return String(value ?? '');
}

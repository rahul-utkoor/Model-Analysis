import type { LayerSummary } from '../types';
import { StatusBadge } from './StatusBadge';

export function LayerNavigator({ layers, selectedLayer, onSelect }: { layers: LayerSummary[]; selectedLayer?: number; onSelect: (layer: number) => void }) {
  return (
    <section className="panel layer-panel">
      <div className="section-heading tight">
        <h2>Layers / Blocks</h2>
        <p>{layers.length} reported units</p>
      </div>
      <div className="layer-list">
        {layers.map((layer) => (
          <button key={layer.layer_index} className={`layer-row ${selectedLayer === layer.layer_index ? 'active' : ''}`} onClick={() => onSelect(layer.layer_index)}>
            <strong>Layer {layer.layer_index}</strong>
            <span>{layer.total_subgraphs} subgraphs</span>
            <StatusBadge value={`${layer.valid_plan_subgraphs} valid`} tone={layer.valid_plan_subgraphs ? 'valid' : 'unknown'} />
          </button>
        ))}
      </div>
    </section>
  );
}

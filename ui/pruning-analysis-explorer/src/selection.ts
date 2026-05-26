import type { SubgraphSummary } from './types';

export type PreviousSubgraphIntent = {
  slug?: string;
  displayName?: string;
  semanticCategory?: string;
};

function norm(value: unknown): string {
  return String(value ?? '').toLowerCase();
}

function textFor(item: SubgraphSummary): string {
  return `${item.node_slug} ${item.display_name} ${item.semantic_category} ${item.pruning_class} ${item.plan_status} ${item.validation_status}`.toLowerCase();
}

function isMlpLike(text: string): boolean {
  const t = text.toLowerCase();
  return t.includes('feed forward') || t.includes('feed_forward') || t.includes('ffn') || t.includes('mlp');
}

function isAttentionLike(text: string): boolean {
  return text.toLowerCase().includes('attention');
}

function isResidualLike(text: string): boolean {
  return text.toLowerCase().includes('residual');
}

function isLayerNormLike(text: string): boolean {
  const t = text.toLowerCase();
  return t.includes('layernorm') || t.includes('layer_norm') || t.includes('layer norm');
}

function isProjectionLike(text: string): boolean {
  const t = text.toLowerCase();
  return t.includes('projection') || t.includes('linear') || t.includes('fc1') || t.includes('fc2') || t.includes('c_fc') || t.includes('c_proj');
}

function scoreByIntent(item: SubgraphSummary, previous?: PreviousSubgraphIntent): number {
  if (!previous) return 0;

  const itemText = textFor(item);
  const prevText = `${previous.slug ?? ''} ${previous.displayName ?? ''} ${previous.semanticCategory ?? ''}`.toLowerCase();
  let score = 0;

  if (previous.semanticCategory && norm(item.semantic_category) === norm(previous.semanticCategory)) score += 100;
  if (isMlpLike(prevText) && isMlpLike(itemText)) score += 90;
  if (isAttentionLike(prevText) && isAttentionLike(itemText)) score += 45;
  if (isResidualLike(prevText) && isResidualLike(itemText)) score += 40;
  if (isLayerNormLike(prevText) && isLayerNormLike(itemText)) score += 40;
  if (isProjectionLike(prevText) && isProjectionLike(itemText)) score += 20;

  if (isMlpLike(prevText)) {
    if (norm(item.validation_status).includes('valid')) score += 30;
    if (norm(item.plan_status).includes('valid') || norm(item.plan_status).includes('ready')) score += 20;
    if (norm(item.pruning_class).includes('safe')) score += 10;
  }

  return score;
}

export function makePreviousSubgraphIntent(
  subgraphs: SubgraphSummary[],
  selectedSlug?: string
): PreviousSubgraphIntent | undefined {
  if (!selectedSlug) return undefined;
  const current = subgraphs.find((item) => item.node_slug === selectedSlug);
  if (!current) return { slug: selectedSlug };
  return {
    slug: current.node_slug,
    displayName: current.display_name,
    semanticCategory: current.semantic_category,
  };
}

export function chooseDefaultSubgraph(
  subgraphs: SubgraphSummary[],
  previous?: PreviousSubgraphIntent,
  exactPreferredSlug?: string
): string | undefined {
  if (!subgraphs.length) return undefined;

  if (exactPreferredSlug) {
    const exact = subgraphs.find((item) => item.node_slug === exactPreferredSlug);
    if (exact) return exact.node_slug;
  }

  // Exact slug preservation is allowed only if the slug is present in the newly loaded list.
  if (previous?.slug) {
    const exact = subgraphs.find((item) => item.node_slug === previous.slug);
    if (exact) return exact.node_slug;
  }

  let best: SubgraphSummary | undefined;
  let bestScore = 0;
  for (const item of subgraphs) {
    const score = scoreByIntent(item, previous);
    if (score > bestScore) {
      best = item;
      bestScore = score;
    }
  }
  if (best) return best.node_slug;

  const validMlp = subgraphs.find((item) => {
    const haystack = textFor(item);
    return isMlpLike(haystack) && (norm(item.validation_status).includes('valid') || norm(item.plan_status).includes('valid') || norm(item.plan_status).includes('ready'));
  });
  if (validMlp) return validMlp.node_slug;

  const anyMlp = subgraphs.find((item) => isMlpLike(textFor(item)));
  if (anyMlp) return anyMlp.node_slug;

  return subgraphs[0].node_slug;
}

export function subgraphBelongsToLoadedLayer(subgraphs: SubgraphSummary[], selectedSlug?: string): boolean {
  if (!selectedSlug) return false;
  return subgraphs.some((item) => item.node_slug === selectedSlug);
}

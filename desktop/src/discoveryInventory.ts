import type { DatasetInventoryEntry } from "./apiContracts";

export const DISCOVERY_PAGE_SIZE = 20;

/** Presentation only. Never resolves, replaces, or edits an exact selection. */
export function discoveryInventoryPage(
  entries: readonly DatasetInventoryEntry[],
  query: string,
  timeframe: string,
  requestedPage: number,
) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = entries.filter((entry) => {
    if (timeframe && entry.timeframe !== timeframe) return false;
    const searchable = [entry.symbol, entry.timeframe, entry.sourceKind,
      entry.datasetIdentity, entry.generation, entry.manifestBindingSha256]
      .join(" ").toLowerCase();
    return terms.every((term) => searchable.includes(term));
  });
  const pageCount = Math.max(1, Math.ceil(filtered.length / DISCOVERY_PAGE_SIZE));
  const page = Math.max(0, Math.min(pageCount - 1,
    Number.isFinite(requestedPage) ? Math.floor(requestedPage) : 0));
  return {
    entries: filtered.slice(page * DISCOVERY_PAGE_SIZE, (page + 1) * DISCOVERY_PAGE_SIZE),
    matched: filtered.length,
    page,
    pageCount,
  };
}

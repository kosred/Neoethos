export type CanonicalNativeResearchPublished = {
  relativePath: string;
  byteCount: number;
  fileSha256: string;
  evidenceIdentitySha256: string;
  configuredPopulation: number;
  resolvedPopulation: number;
  populationCap: number;
  hardGrowthCap: number;
  termCap: number;
  selectedDeviceOrdinal: number;
  engine: string;
  parentH2dBytes: number;
  adaptiveH2dBytes: number;
  metricRows: number;
  metricBytes: number;
  consumerCompletionConfirmed: boolean;
  replayIdentitySealed: boolean;
};

export type CanonicalNativeResearchStatus = {
  state: string;
  stage: string;
  percent: number;
  leaseToken: string | null;
  cancellationRequested: boolean;
  failureStage: string | null;
  failureCode: string | null;
  failureDetail: string | null;
  published: CanonicalNativeResearchPublished | null;
};

export type CanonicalNativeResearchStartBody = {
  contractArtifact: {
    relativePath: string;
    expectedSha256: string;
  };
  population: number | null;
  populationAuto: boolean | null;
  maxIndicators: number | null;
};

export function nativeResearchStartBody(
  relativePath: string,
  expectedSha256: string,
  population: number | null,
  populationAuto: boolean | null,
  maxIndicators: number | null,
): CanonicalNativeResearchStartBody {
  return {
    contractArtifact: { relativePath, expectedSha256 },
    population,
    populationAuto,
    maxIndicators,
  };
}

export function boundedNativeResearchText(value: string, max = 320): string {
  const normalized = value.replace(/[\r\n\t]+/g, " ").trim();
  const characters = [...normalized];
  if (characters.length <= max) return normalized;
  if (max <= 1) return "…".slice(0, max);
  return `${characters.slice(0, max - 1).join("")}…`;
}

export function nativeResearchFailureSummary(
  status: CanonicalNativeResearchStatus,
): string {
  if (!status.failureStage && !status.failureCode && !status.failureDetail) return "";
  const stage = boundedNativeResearchText(status.failureStage ?? "unknown", 96);
  const code = boundedNativeResearchText(status.failureCode ?? "unknown", 96);
  const detail = boundedNativeResearchText(status.failureDetail ?? "", 320);
  return `stage=${stage} · code=${code}${detail ? ` · ${detail}` : ""}`;
}

export function nativeResearchPublishedSummary(
  published: CanonicalNativeResearchPublished,
): string {
  const summary = [
    `path=${boundedNativeResearchText(published.relativePath, 180)}`,
    `sha256=${boundedNativeResearchText(published.fileSha256, 64)}`,
    `bytes=${published.byteCount}`,
    `configured P=${published.configuredPopulation}`,
    `resolved P=${published.resolvedPopulation}`,
    `population cap=${published.populationCap}`,
    `hard cap=${published.hardGrowthCap}`,
    `T=${published.termCap}`,
    `device=${published.selectedDeviceOrdinal}`,
    `engine=${boundedNativeResearchText(published.engine, 48)}`,
    `H2D=${published.parentH2dBytes}+${published.adaptiveH2dBytes}`,
    `metrics=${published.metricRows} rows/${published.metricBytes} bytes`,
    `consumer=${published.consumerCompletionConfirmed ? "yes" : "no"}`,
    `replay=${published.replayIdentitySealed ? "yes" : "no"}`,
  ].join(" · ");
  return boundedNativeResearchText(summary, 800);
}

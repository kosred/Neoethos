import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/components/KChart.tsx", import.meta.url),
  "utf8",
);

test("live ticks cannot create or mutate exact trendbar OHLC candles", () => {
  for (const forbidden of [
    /TF_SECONDS/,
    /timeframeSeconds/i,
    /bucketMs/,
    /formingRef/,
    /barCbRef/,
    /subscribeBar/,
    /unsubscribeBar/,
    /cb\(bar\)/,
    /open:\s*price/,
    /high:\s*price/,
    /low:\s*price/,
    /close:\s*price/,
    /MN1:\s*2592000/,
  ]) {
    assert.doesNotMatch(source, forbidden);
  }

  assert.match(source, /className="kchart-live-price-marker"/);
  assert.match(source, /liveTick\.midPrice/);
  assert.match(source, /liveTick\.symbolName\s*===\s*symbol/);
});

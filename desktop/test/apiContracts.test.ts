import assert from "node:assert/strict";
import test from "node:test";

import {
  amendProtectionBody,
  dataImportBody,
  promoteStrategyBody,
} from "../src/apiContracts.ts";

test("amendProtectionBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = amendProtectionBody(42, 1.07125, 1.0845, true);

  assert.equal(
    JSON.stringify(body),
    '{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true}',
  );
});

test("dataImportBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = dataImportBody("C:/market-data/EURUSD.csv", "EURUSD", "M5");

  assert.equal(
    JSON.stringify(body),
    '{"sourcePath":"C:/market-data/EURUSD.csv","symbol":"EURUSD","timeframe":"M5"}',
  );
});

test("promoteStrategyBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = promoteStrategyBody("EURUSD", "M5");

  assert.equal(JSON.stringify(body), '{"symbol":"EURUSD","baseTf":"M5"}');
});

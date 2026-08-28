import { useState } from "react";

import {
  canonicalNativeResearchCancel,
  canonicalNativeResearchStart,
  enginesStatus,
} from "../api";
import { usePoll } from "../hooks";
import {
  boundedNativeResearchText,
  nativeResearchFailureSummary,
  nativeResearchPublishedSummary,
  nativeResearchStartBody,
} from "../nativeResearch";

function optionalPositiveInteger(raw: string, label: string): number | null {
  if (!raw.trim()) return null;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer or blank.`);
  }
  return value;
}

export default function NativeResearch() {
  const { data, error, reload } = usePoll(enginesStatus, 1_000);
  const [relativePath, setRelativePath] = useState("");
  const [expectedSha256, setExpectedSha256] = useState("");
  const [population, setPopulation] = useState("");
  const [populationAuto, setPopulationAuto] = useState("");
  const [maxIndicators, setMaxIndicators] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const status = data?.canonicalNativeResearch;
  const state = status?.state ?? "Idle";
  const running = state === "Queued" || state === "Running";
  const failure = status ? nativeResearchFailureSummary(status) : "";
  const published = status?.published
    ? nativeResearchPublishedSummary(status.published)
    : "";

  const start = async () => {
    const path = relativePath.trim();
    const sha256 = expectedSha256.trim();
    if (!path) {
      setMessage("Contract relative path is required.");
      return;
    }
    if (!/^[0-9a-f]{64}$/.test(sha256)) {
      setMessage("Expected SHA-256 must be exactly 64 lowercase hexadecimal characters.");
      return;
    }
    setBusy(true);
    setMessage("Requesting canonical Native Research…");
    try {
      const response = await canonicalNativeResearchStart(
        nativeResearchStartBody(
          path,
          sha256,
          optionalPositiveInteger(population, "Population"),
          populationAuto === "" ? null : populationAuto === "true",
          optionalPositiveInteger(maxIndicators, "Max indicators"),
        ),
      );
      setMessage(`Accepted · lease ${response.leaseToken}`);
      reload();
    } catch (cause) {
      setMessage(`Start failed: ${boundedNativeResearchText(String(cause), 400)}`);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    const leaseToken = status?.leaseToken;
    if (!leaseToken) {
      setMessage("No queued or running Native Research lease is available to cancel.");
      return;
    }
    setBusy(true);
    try {
      await canonicalNativeResearchCancel(leaseToken);
      setMessage(`Cancellation requested for lease ${leaseToken}.`);
      reload();
    } catch (cause) {
      setMessage(`Cancel failed: ${boundedNativeResearchText(String(cause), 400)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Native Research</h1>
      <p className="sub">
        Run the sealed canonical generation-zero research contract on the native lane. This
        action publishes research evidence only; it never launches model training.
      </p>

      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>
          {state.toUpperCase()}
        </span>
        {status?.stage && <span className="muted">stage · {status.stage}</span>}
        {status && <span className="muted">{status.percent.toFixed(2)}%</span>}
        {status?.cancellationRequested && (
          <span className="badge demo">CANCELLATION REQUESTED</span>
        )}
      </div>

      {error && (
        <div className="banner warn">{boundedNativeResearchText(error, 400)}</div>
      )}
      {failure && <div className="banner warn">Failure · {failure}</div>}
      {published && <div className="banner info">Published · {published}</div>}

      <h2>Exact contract</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label style={{ flex: 1 }}>
            Relative path
            <input
              value={relativePath}
              onChange={(event) => setRelativePath(event.target.value)}
              placeholder="contracts/generation-zero.json"
            />
          </label>
          <label style={{ flex: 2 }}>
            Expected SHA-256
            <input
              value={expectedSha256}
              onChange={(event) => setExpectedSha256(event.target.value)}
              placeholder="64 lowercase hexadecimal characters"
              spellCheck={false}
            />
          </label>
        </div>
        <div className="ticket-row">
          <label>
            Population override
            <input
              type="number"
              min="1"
              value={population}
              onChange={(event) => setPopulation(event.target.value)}
              placeholder="inherit"
            />
          </label>
          <label>
            Population auto
            <select
              value={populationAuto}
              onChange={(event) => setPopulationAuto(event.target.value)}
            >
              <option value="">inherit</option>
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </label>
          <label>
            Max indicators override
            <input
              type="number"
              min="1"
              value={maxIndicators}
              onChange={(event) => setMaxIndicators(event.target.value)}
              placeholder="inherit"
            />
          </label>
        </div>
        <div className="banner info">
          The app process owns the move-only lease until the worker reaches its real terminal
          state. Cancel uses the exact opaque lease token shown by live status.
        </div>
        <div className="btn-row">
          <button className="primary" disabled={busy || running} onClick={start}>
            {running ? "Native Research running…" : "Start Native Research"}
          </button>
          <button
            className="danger"
            disabled={busy || !running || !status?.leaseToken}
            onClick={cancel}
          >
            Cancel exact lease
          </button>
          <button disabled={busy} onClick={reload}>Refresh status</button>
        </div>
        {message && <div className="banner info">{message}</div>}
      </div>
    </div>
  );
}

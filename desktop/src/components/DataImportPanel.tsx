import { useState } from "react";
import { dataImport, pickDataFile } from "../api";
import {
  dataImportIdentityKey,
  dataOperationErrorText,
  expectedGenerationFor,
  recordDatasetGeneration,
  type DataImportBody,
  type DataImportSourceFormat,
  type DatasetGenerationReceipts,
} from "../apiContracts";
import { TimeframeSelect } from "./Select";

export function DataImportPanel({ onImported }: { onImported: () => void | Promise<unknown> }) {
  const [sourcePath, setSourcePath] = useState("");
  const [format, setFormat] = useState<DataImportSourceFormat>("csv");
  const [sourceNamespace, setSourceNamespace] = useState("operator-upload");
  const [symbol, setSymbol] = useState("EURUSD");
  const [timeframe, setTimeframe] = useState("H1");
  const [timestampConvention, setTimestampConvention] =
    useState<DataImportBody["barTimestampConvention"]>("bar_open");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [generationReceipts, setGenerationReceipts] = useState<DatasetGenerationReceipts>({});

  const browse = async () => {
    try {
      const path = await pickDataFile();
      if (path) setSourcePath(path);
    } catch (error) {
      setMessage(dataOperationErrorText(error));
    }
  };

  const importSelectedFile = async () => {
    const namespace = sourceNamespace.trim();
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!sourcePath) {
      setMessage("Choose a source file first.");
      return;
    }
    if (!namespace || !normalizedSymbol) {
      setMessage("Source namespace and symbol are required.");
      return;
    }

    const requestIdentity = dataImportIdentityKey(
      namespace,
      normalizedSymbol,
      timeframe,
      timestampConvention,
    );
    setBusy(true);
    setMessage("Importing and verifying canonical generation…");
    try {
      const outcome = await dataImport(
        sourcePath,
        format,
        namespace,
        normalizedSymbol,
        timeframe,
        timestampConvention,
        expectedGenerationFor(generationReceipts, requestIdentity),
      );
      setGenerationReceipts((current) =>
        recordDatasetGeneration(current, requestIdentity, outcome),
      );
      await onImported();
      setMessage(
        `Imported ${outcome.rowCount.toLocaleString()} rows as ${outcome.datasetIdentity} @ ${outcome.generation}.`,
      );
    } catch (error) {
      setMessage(`Import failed: ${dataOperationErrorText(error)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-labelledby="data-import-heading">
      <h2 id="data-import-heading">Import a local dataset</h2>
      <div className="ticket">
        <p className="muted small">
          CSV, TSV, JSON, Parquet, Arrow IPC and Vortex sources are converted into a verified
          canonical Vortex generation. Runtime consumers open only that published generation.
        </p>
        <div className="ticket-row">
          <button type="button" onClick={browse} disabled={busy}>Browse…</button>
          <label style={{ flex: 1 }}>
            Source file
            <input value={sourcePath} onChange={(event) => setSourcePath(event.target.value)} placeholder="Choose a file" />
          </label>
          <label>
            Format
            <select value={format} onChange={(event) => setFormat(event.target.value as DataImportSourceFormat)}>
              <option value="csv">CSV</option>
              <option value="tsv">TSV</option>
              <option value="json-array">JSON array</option>
              <option value="json-lines">JSON lines</option>
              <option value="parquet">Parquet</option>
              <option value="arrow-ipc-file">Arrow IPC file</option>
              <option value="arrow-ipc-stream">Arrow IPC stream</option>
              <option value="vortex">Vortex</option>
            </select>
          </label>
        </div>
        <div className="ticket-row">
          <label>Source namespace<input value={sourceNamespace} onChange={(event) => setSourceNamespace(event.target.value)} /></label>
          <label>Symbol<input value={symbol} onChange={(event) => setSymbol(event.target.value)} /></label>
          <label>Timeframe<TimeframeSelect value={timeframe} onChange={setTimeframe} /></label>
          <label>
            Timestamp meaning
            <select
              value={timestampConvention}
              onChange={(event) => setTimestampConvention(event.target.value as DataImportBody["barTimestampConvention"])}
            >
              <option value="bar_open">Bar open</option>
              <option value="bar_close">Bar close (rejected)</option>
              <option value="bar_end">Bar end (rejected)</option>
              <option value="unknown">Unknown (rejected)</option>
            </select>
          </label>
          <button type="button" className="primary" disabled={busy || !sourcePath} onClick={importSelectedFile}>
            {busy ? "Importing…" : "Import and verify"}
          </button>
        </div>
        {message && <div className="banner info" role="status">{message}</div>}
      </div>
    </section>
  );
}

import { hardwareInfo } from "../api";
import { usePoll } from "../hooks";

// The backend's legacy *Mb fields contain MiB (bytes / 1024²).
const gib = (mib: number) => `${(mib / 1024).toFixed(1)} GiB`;

export default function Hardware() {
  const { data, error, loading, reload } = usePoll(hardwareInfo, 3000);

  return (
    <div className="screen">
      <h1>Hardware</h1>
      <p className="sub">Compute resources available to discovery &amp; training</p>
      {loading && !data && <p className="muted" role="status">Detecting hardware…</p>}
      {error && <div className="banner warn" role="alert">Hardware could not refresh. Retained readings may be stale. {error} <button onClick={() => void reload()}>Retry</button></div>}
      {data && (
        <>
          <h2>CPU</h2>
          <div className="settings-grid">
            <div className="kv"><span>Model</span><b style={{ fontSize: 12 }}>{data.cpu.model}</b></div>
            <div className="kv"><span>Cores</span><b>{data.cpu.coresPhysical} phys · {data.cpu.coresLogical} logical</b></div>
            <div className="kv"><span>Load</span><b>{(data.cpu.loadAvg * 100).toFixed(0)}%</b></div>
          </div>
          <h2>Memory</h2>
          <div className="settings-grid">
            <div className="kv"><span>Total</span><b>{gib(data.ram.totalMb)}</b></div>
            <div className="kv"><span>Used</span><b>{gib(data.ram.usedMb)}</b></div>
            <div className="kv"><span>Available</span><b>{gib(data.ram.availableMb)}</b></div>
          </div>
          <h2>GPU</h2>
          <div className="settings-grid">
            <div className="kv"><span>Device</span><b style={{ fontSize: 12 }}>{data.gpu.name || "—"}</b></div>
            <div className="kv"><span>Backend</span><b>{data.gpu.kind}</b></div>
            <div className="kv"><span>Card detected</span><b className={data.gpu.available ? "buy" : "sell"}>{data.gpu.available ? "yes" : "no"}</b></div>
            <div className="kv">
              <span>GPU lane in this build</span>
              <b className={data.gpuSupport?.compiled ? "buy" : "sell"}>
                {data.gpuSupport?.compiled ? data.gpuSupport.backend.toUpperCase() : "not compiled"}
              </b>
            </div>
          </div>
          {/* Both conditions must hold before any GPU work happens. Saying so
              here is what stops a rented card from sitting idle unnoticed. */}
          {data.gpuSupport && !data.gpuSupport.compiled && data.gpu.available && (
            <div className="banner warn">
              <b>A GPU is present but this build cannot use it.</b> {data.gpuSupport.detail}
            </div>
          )}
          {data.gpuSupport?.compiled && !data.gpu.available && (
            <div className="banner warn">
              This build has the <b>{data.gpuSupport.backend.toUpperCase()}</b> lane compiled in,
              but no usable card was detected. A selected GPU run must fail closed;
              CPU research is a separate explicit execution choice.
            </div>
          )}
          {data.gpuSupport?.compiled && data.gpu.available && (
            <div className="banner info">
              GPU lane <b>{data.gpuSupport.backend.toUpperCase()}</b> is compiled in and a card is
              present. This is hardware detection, not proof that Discovery or training ran on the GPU.
              Device execution, memory admission and numerical validation still have to pass.
            </div>
          )}
        </>
      )}
    </div>
  );
}

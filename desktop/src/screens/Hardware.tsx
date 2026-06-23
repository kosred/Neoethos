import { hardwareInfo } from "../api";
import { usePoll } from "../hooks";

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

export default function Hardware() {
  const { data, error } = usePoll(hardwareInfo, 3000);

  return (
    <div className="screen">
      <h1>Hardware</h1>
      <p className="sub">Compute resources available to discovery &amp; training</p>
      {error && <div className="banner warn">{error}</div>}
      {data && (
        <>
          <h2>CPU</h2>
          <div className="settings-grid">
            <div className="kv"><span>Model</span><b style={{ fontSize: 12 }}>{data.cpu.model}</b></div>
            <div className="kv"><span>Cores</span><b>{data.cpu.cores_physical} phys · {data.cpu.cores_logical} logical</b></div>
            <div className="kv"><span>Load</span><b>{(data.cpu.load_avg * 100).toFixed(0)}%</b></div>
          </div>
          <h2>Memory</h2>
          <div className="settings-grid">
            <div className="kv"><span>Total</span><b>{gb(data.ram.total_mb)}</b></div>
            <div className="kv"><span>Used</span><b>{gb(data.ram.used_mb)}</b></div>
            <div className="kv"><span>Available</span><b>{gb(data.ram.available_mb)}</b></div>
          </div>
          <h2>GPU</h2>
          <div className="settings-grid">
            <div className="kv"><span>Device</span><b style={{ fontSize: 12 }}>{data.gpu.name || "—"}</b></div>
            <div className="kv"><span>Backend</span><b>{data.gpu.kind}</b></div>
            <div className="kv"><span>Available</span><b className={data.gpu.available ? "buy" : "sell"}>{data.gpu.available ? "yes" : "no"}</b></div>
          </div>
        </>
      )}
    </div>
  );
}

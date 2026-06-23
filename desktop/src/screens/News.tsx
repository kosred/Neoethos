import { useState } from "react";
import { newsFeed } from "../api";
import { usePoll } from "../hooks";

export default function News() {
  const [force, setForce] = useState(false);
  const { data, error, loading, reload } = usePoll(() => newsFeed(force), 0, [force]);

  // Shape: { items: [{title, source, url, summary, published_at}], aiSummary, aiAvailable, notice }
  const items: any[] = data?.items ?? data?.headlines ?? (Array.isArray(data) ? data : []);
  const briefing: string | undefined = data?.aiSummary || undefined;
  const notice: string | undefined = data?.notice || undefined;

  return (
    <div className="screen">
      <h1>News</h1>
      <p className="sub">Market headlines + AI briefing</p>

      <div className="btn-row">
        <button disabled={loading} onClick={() => { setForce(true); reload(); }}>Refresh</button>
      </div>
      {error && <div className="banner warn">{error}</div>}
      {notice && <div className="banner warn">{notice}</div>}

      {briefing && (
        <div className="banner info" style={{ whiteSpace: "pre-wrap" }}>
          <b>AI briefing</b>
          <div style={{ marginTop: 6 }}>{briefing}</div>
        </div>
      )}

      {items.length === 0 ? (
        <p className="muted">{loading ? "Loading…" : "No headlines available."}</p>
      ) : (
        <div className="news-list">
          {items.slice(0, 60).map((it, i) => (
            <div className="news-item" key={i}>
              <div className="news-title">{it.title ?? it.headline ?? "(untitled)"}</div>
              <div className="muted small">
                {(it.source ?? it.feed ?? "")}{it.published_at ? ` · ${it.published_at}` : ""}
              </div>
              {it.summary && <div className="news-summary">{it.summary}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

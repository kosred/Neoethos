import { useState } from "react";
import { newsFeed } from "../api";
import { usePoll } from "../hooks";

export default function News() {
  const [refreshVersion, setRefreshVersion] = useState(0);
  const { data, error, loading } = usePoll(
    () => newsFeed(refreshVersion > 0),
    0,
    refreshVersion,
  );

  const items = data?.items ?? [];
  const briefing: string | undefined = data?.aiSummary || undefined;
  const notice: string | undefined = data?.notice || undefined;

  return (
    <div className="screen">
      <h1>News</h1>
      <p className="sub">Market headlines + AI briefing</p>
      <p className="muted small">
        Opening or refreshing this page may fetch public feeds and request an AI briefing using
        your existing ChatGPT login. Results may be reused for up to 10 minutes. Broker settings
        do not control these requests.
      </p>

      <div className="btn-row">
        <button disabled={loading} onClick={() => setRefreshVersion((version) => version + 1)}>Refresh headlines &amp; briefing</button>
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
        <p className="muted">{error ? "Headlines could not be loaded." : loading ? "Loading…" : "No headlines available."}</p>
      ) : (
        <div className="news-list">
          {items.slice(0, 60).map((item) => (
            <div className="news-item" key={`${item.link}-${item.publishedMs ?? 0}`}>
              <div className="news-title">
                {item.link ? <a href={item.link} target="_blank" rel="noreferrer">{item.title || "(untitled)"}</a> : (item.title || "(untitled)")}
              </div>
              <div className="muted small">
                {item.source}{item.publishedMs ? ` · ${new Date(item.publishedMs).toLocaleString()}` : ""}
              </div>
              {item.blurb && <div className="news-summary">{item.blurb}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

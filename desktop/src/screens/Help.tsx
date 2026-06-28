type Sec = { id: string; icon: string; title: string; group: string; what: string; how: string[] };

const SECTIONS: Sec[] = [
  // ── Trade ──
  { id: "cockpit", icon: "🎯", title: "Trade (Cockpit)", group: "Trade",
    what: "The all-in-one trading desk: a market watch list, a live chart, an order ticket and your open positions on one screen.",
    how: [
      "Click a symbol in the watch list to load its chart.",
      "Change the chart timeframe and add an indicator overlay from the dropdowns.",
      "Use the order ticket (Lots + optional SL/TP) to place a manual BUY/SELL on the active account.",
      "Open positions show live P/L at the bottom with one-click close.",
    ] },
  { id: "dashboard", icon: "▦", title: "Dashboard", group: "Trade",
    what: "A quick overview of account, engine and broker status at a glance.",
    how: ["Read-only — a summary landing page. Use the sidebar to dive into any area."] },
  { id: "markets", icon: "📈", title: "Markets", group: "Trade",
    what: "A focused charting screen for one symbol at a time with indicator overlays.",
    how: ["Pick a symbol, timeframe and indicator from the dropdowns.", "Scroll back in time to load older history."] },
  { id: "marketwatch", icon: "👁", title: "Market Watch", group: "Trade",
    what: "Live streaming prices for all your symbols at once.",
    how: ["Type in the filter box to narrow the list.", "Prices update in real time via a push stream."] },
  { id: "positions", icon: "≡", title: "Positions", group: "Trade",
    what: "Place and manage manual trades; watch open positions with live profit/loss.",
    how: [
      "New market order: choose Symbol, BUY/SELL, Lots and optional SL/TP in pips.",
      "Modify SL/TP moves a stop (e.g. to breakeven); Close exits the position.",
      "These are real orders on the active account — check whether it's Demo or Live in the header.",
    ] },
  { id: "account", icon: "💳", title: "Account", group: "Trade",
    what: "Account identity, balance/equity and low-level order tools.",
    how: ["View balance, equity and the detected account type.", "Advanced order-by-symbol-id tools live here for testing."] },
  { id: "actions", icon: "✓", title: "Actions", group: "Trade",
    what: "A queue of pending actions the app wants you to confirm.",
    how: ["Review each item and approve or dismiss it."] },
  // ── Autopilot ──
  { id: "autopilot", icon: "🤖", title: "Autopilot", group: "Autopilot",
    what: "Run a discovered strategy automatically — dry-run on history, or live on the broker.",
    how: [
      "Pick a discovered portfolio from the list.",
      "Replay (dry-run) tests it on stored history with zero broker calls.",
      "The demo forward-test gate shows whether the strategy has earned the right to trade real money.",
      "Start live runs the bar→signal→order loop. On a Live account it's blocked until the gate passes; on Demo it always runs (that's how the track record is built).",
    ] },
  { id: "riskymode", icon: "🚀", title: "Risky Mode", group: "Autopilot",
    what: "An aggressive, fully-automatic mode that hunts for strategies to multiply a small account. Everything is computed for you.",
    how: [
      "Read-only by design: you cannot tweak its parameters — sizing, risk band and targets are set automatically.",
      "It uses half-Kelly sizing and goal-based ranking. Costs come from your real broker table.",
    ] },
  { id: "risk", icon: "🛡", title: "Risk", group: "Autopilot",
    what: "Position-sizing limits, drawdown guards and prop-firm presets that protect every automated trade.",
    how: [
      "Pick a preset (e.g. an FTMO-style profile) or review the current limits.",
      "Risk-per-trade, daily/total drawdown caps and max lot size are shown here.",
    ] },
  // ── Research ──
  { id: "discovery", icon: "🧬", title: "Discovery", group: "Research",
    what: "The strategy factory: a genetic search that breeds and tests thousands of rules and keeps the ones that survive out-of-sample validation.",
    how: [
      "Pick a Symbol and Base TF (or leave them on config defaults).",
      "Optionally open Advanced for population/generations/portfolio size.",
      "Start and watch the progress bar + counters; results appear in Strategy Lab / Autopilot.",
    ] },
  { id: "training", icon: "🎓", title: "Training", group: "Research",
    what: "Fits the machine-learning ensemble that acts as a regime filter on top of the discovered rules.",
    how: ["Pick the same Symbol + Base TF you discovered on, then Start.", "Validated on an 80/20 hold-out; models save to the model store."] },
  { id: "strategylab", icon: "⚗", title: "Strategy Lab", group: "Research",
    what: "The backtest quality gate between discovery and trading.",
    how: ["Check gate shows each promotion criterion and a PROMOTE/HOLD verdict.", "Promote to live copies a passing portfolio into the live set."] },
  { id: "strategyreport", icon: "📅", title: "Strategy Report", group: "Research",
    what: "A monthly journal + honest validation verdict for each discovered strategy, with €-based equity from €1000.",
    how: ["Pick a strategy to see month-by-month returns, validation flags and any honesty warnings."] },
  { id: "intelligence", icon: "🧠", title: "Intelligence", group: "Research",
    what: "An inventory of trained model artifacts and discovered strategy targets.",
    how: ["Read-only — confirms what models/strategies exist and their validation stats."] },
  // ── Data & Files ──
  { id: "files", icon: "🗂", title: "Files & Storage", group: "Data & Files",
    what: "Shows exactly where everything the app downloads, trains or logs is kept — one click to open each folder.",
    how: ["Click Open next to any row (config, data, models, cache, journal, logs) to reveal it in Explorer."] },
  { id: "data", icon: "🗄", title: "Data", group: "Data & Files",
    what: "Download historical bars from the broker and refresh real trading costs.",
    how: [
      "Pick Symbol + Timeframe + From date, then Fetch from broker.",
      "Refresh broker costs once so backtests use your account's real commission/swap.",
    ] },
  // ── Desk ──
  { id: "journal", icon: "📒", title: "Journal", group: "Desk",
    what: "A closed-trade log with computed stats (MyFxbook-style): win rate, profit factor, drawdown, expectancy.",
    how: ["Read-only — fills automatically as trades close on the account."] },
  { id: "news", icon: "📰", title: "News", group: "Desk",
    what: "Market headlines plus an AI briefing.",
    how: ["Press Refresh to pull the latest headlines + summary."] },
  { id: "aidesk", icon: "💬", title: "AI Desk", group: "Desk",
    what: "A chat assistant powered by your ChatGPT (Codex) subscription for market questions and help.",
    how: ["Sign in once, then type a message and Send."] },
  // ── System ──
  { id: "hardware", icon: "🖥", title: "Hardware", group: "System",
    what: "Detected CPU/GPU and the compute profile the engine will use.",
    how: ["Read-only — confirms what hardware discovery/training can use."] },
  { id: "advanced", icon: "🔧", title: "Advanced", group: "System",
    what: "Power-user tools: diagnostics, data import, config presets and raw YAML editing.",
    how: ["Run diagnostics for a health report.", "Import a CSV/Parquet file.", "Switch presets or hand-edit any setting (affects everything)."] },
  { id: "settings", icon: "⚙", title: "Settings", group: "System",
    what: "The single place to change configuration — discovery mode, risky goal, search tuning, compute, risk, news, broker. Each control writes config.yaml; only Discovery + Training actions live outside.",
    how: [
      "Discovery mode: Prop-firm (robust) vs Risky (multiply). Risky goal: start/target/horizon.",
      "Compute: auto / cpu / gpu. Risk & sizing: pick a preset; limits update.",
      "News gate: pause / allow / warn around high-impact events.",
      "Broker Setup: re-authenticate or pick a granted account if trading says it can't route.",
    ] },
  { id: "tuning", icon: "🧬", title: "Search tuning (anti-stagnation)", group: "System",
    what: "If Discovery stalls early or finds few strategies, these knobs (in Settings) widen/deepen the genetic search. The GA can settle on a local optimum — raise these to push it to keep exploring different indicators.",
    how: [
      "Indicator pool (prefilter_top_k): how many indicators the GA may use. RAISE first if it stalls — the #1 lever (50→120 gave a ~6× jump in strategies found).",
      "Explore patience (convergence_patience): flat generations before the GA gives up. Raise (e.g. 500) to search much longer.",
      "Diversity kick (stagnation_patience): flat generations before heavier mutation + fresh genes kick in. Lower = reacts sooner.",
      "Novelty reward (novelty_weight): 0 = off; 0.1–0.3 rewards DIFFERENT genes → more market-regime variety.",
      "Disable SMC gate: turn off the structural gate if it's over-constraining a pair.",
      "Save writes config.yaml; applies to the next Discovery run.",
    ] },
];

const GROUPS = ["Trade", "Autopilot", "Research", "Data & Files", "Desk", "System"];

export default function Help() {
  return (
    <div className="screen">
      <h1>Help &amp; Guide</h1>
      <p className="sub">What each part of NeoEthos does, and how to use it</p>

      <div className="help-panel open">
        <div className="help-body">
          <p><b>The typical workflow, start to finish:</b></p>
          <div className="help-step"><span className="help-step-n">1</span><div><b>Data</b> → download price history for the pairs you care about (and Refresh broker costs once).</div></div>
          <div className="help-step"><span className="help-step-n">2</span><div><b>Discovery</b> → search for strategies on that data. <b>Training</b> → fit the model filter.</div></div>
          <div className="help-step"><span className="help-step-n">3</span><div><b>Strategy Lab</b> → check the quality gate and promote good portfolios. <b>Strategy Report</b> → read the honest monthly verdict.</div></div>
          <div className="help-step"><span className="help-step-n">4</span><div><b>Autopilot</b> → replay a strategy, run it on a <b>Demo</b> account to build a track record, then (once the demo gate passes) go <b>Live</b>.</div></div>
          <div className="help-step"><span className="help-step-n">5</span><div><b>Positions</b> / <b>Journal</b> → manage trades and review results.</div></div>
        </div>
      </div>

      <div className="guide-toc">
        {SECTIONS.map((s) => (
          <a key={s.id} href={`#g-${s.id}`}>{s.icon} {s.title}</a>
        ))}
      </div>

      {GROUPS.map((g) => (
        <div key={g}>
          <h2 style={{ marginTop: 18 }}>{g}</h2>
          {SECTIONS.filter((s) => s.group === g).map((s) => (
            <div className="guide-sec" id={`g-${s.id}`} key={s.id}>
              <div className="tag">{g}</div>
              <h2>{s.icon} {s.title}</h2>
              <p>{s.what}</p>
              <ul>
                {s.how.map((h, i) => <li key={i}>{h}</li>)}
              </ul>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

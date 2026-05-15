# Wizard / Onboarding — Competitive Analysis

Compiled 2026-05-15 by the research agent in response to the operator
directive (verbatim, in Greek):

> "Υπάρχουν δεκάδες επιλογές στην εφαρμογή μέσω backend από τα δεδομένα
> μέχρι τον να κάνει αυτόματα τα πάντα μέχρι live trading με ένα
> κουμπί. Πρέπει να ψάξουμε λίγο online πώς το κάνουν οι μεγάλες
> εταιρίες."

Translation: "There are dozens of options in the app via the backend
from data, to doing everything automatically, to live trading with one
button. We need to look around online at how the big firms do it."

This document is a **research deliverable** — no code changes. It
surveys how the canonical retail-trading, algo-trading, crypto-exchange
and prop-firm platforms onboard a brand-new user from "Welcome" to a
first executed trade, distils the patterns that survive across all of
them, and proposes specific additions to the 10-step `forex-app`
wizard described in `docs/audits/research/installer_wizard_ux_spec.md`.

The companion read for *what UI we paint these flows in* is
`docs/audits/research/ui_ux_design_spec.md`; the companion read for
*which Spotware payloads we can call during each step* is
`docs/audits/research/ctrader_api_full_reference.md`. Both are
referenced inline.

---

## 0. Sources & methodology

Three sandbox conditions shaped this audit:

1. **WebFetch is sometimes blocked.** TradingView, MetaTrader 5,
   NinjaTrader and several Apple / FTMO subdomains return HTTP 403
   to WebFetch from this sandbox. Where this happened, the citation
   reads `(WebSearch excerpt, <date>)` and the inline quote comes
   from search-result excerpts that visibly quote the canonical
   page. The same fallback documented at
   `docs/audits/research/installer_wizard_ux_spec.md` §0.
2. **WebSearch is preferred** when the doc index is broad
   (multi-platform comparisons, prop-firm rules); WebFetch when
   one canonical page is enough.
3. **No paywall sources.** Everything cited is publicly indexed.

### 0.1 Citation index — external

- TradingView "Trading Panel" broker-connect walkthrough
  <https://www.tradingview.com/support/solutions/43000669310-what-should-i-do-to-be-able-to-trade-through-interactive-brokers-on-tradingview/>
  (WebSearch excerpt, 2026-05-15)
- TradingView brokerage-integration spec
  <https://www.tradingview.com/brokerage-integration/> (WebSearch
  excerpt, 2026-05-15)
- MetaTrader 5 *Open an Account*
  <https://www.metatrader5.com/en/terminal/help/startworking/acc_open>
  (WebSearch excerpt, 2026-05-15)
- MetaTrader 5 *Connect to an Account*
  <https://www.metatrader5.com/en/terminal/help/startworking/authorization>
  (WebSearch excerpt, 2026-05-15)
- MetaEditor *Creating a ready-made Expert Advisor — MQL5 Wizard*
  <https://www.metatrader5.com/en/metaeditor/help/mql5_wizard/wizard_ea_generate>
  (WebSearch excerpt, 2026-05-15)
- NinjaTrader 8 *Simulated Data Feed Connection*
  <https://ninjatrader.com/support/helpguides/nt8/simulated_data_feed_connection.htm>
  (WebSearch excerpt, 2026-05-15)
- ThinkOrSwim *paperMoney* — Charles Schwab Learn
  <https://www.schwab.com/learn/story/thinkorswim-papermoney-stock-trading-simulator>
  (WebSearch excerpt, 2026-05-15)
- tastytrade *Portfolio Margin Agreement & Risk Disclosure 2026*
  <https://assets.tastyworks.com/production/documents/portfolio_margin_customer_agreement.pdf>
  (WebSearch excerpt, 2026-05-15)
- tastytrade *Trading Permissions*
  <https://tastytrade.com/learn/accounts/account-resources/trading-permissions/>
  (WebSearch excerpt, 2026-05-15)
- cTrader Help — *Manage cBots and indicators*
  <https://help.ctrader.com/ctrader-algo/how-tos/all-algos/manage-cbots-and-indicators-using-algos/>
  (WebSearch excerpt, 2026-05-15)
- cTrader Help — *Start a cBot*
  <https://help.ctrader.com/ctrader-algo/documentation/cbots/start-a-cbot/>
  (WebSearch excerpt, 2026-05-15)
- cTrader Copy — *Invest in strategies*
  <https://help.ctrader.com/ctrader-copy/investing-in-strategies/>
  (WebSearch excerpt, 2026-05-15)
- QuantConnect — *Algorithm Wizard 2.0 / Strategy Builder*
  <https://www.quantconnect.com/blog/algorithm-lab-2-0/>
  (WebSearch excerpt, 2026-05-15)
- QuantConnect — *Strategy Library*
  <https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library>
  (WebSearch excerpt, 2026-05-15)
- TradeStation *Learning EasyLanguage Strategies* PDF
  <https://cdn.tradestation.com/uploads/Learning-EasyLanguage-Strategies.pdf>
  (WebSearch excerpt, 2026-05-15)
- Alpaca *Paper Trading*
  <https://docs.alpaca.markets/us/docs/paper-trading> (WebSearch
  excerpt, 2026-05-15)
- Alpaca *Getting Started with Trading API*
  <https://docs.alpaca.markets/us/docs/getting-started-with-trading-api>
  (WebSearch excerpt, 2026-05-15)
- FTMO *Trading Objectives*
  <https://ftmo.com/en/trading-objectives/> (WebSearch excerpt,
  2026-05-15)
- FTMO Academy *Maximum Daily Loss*
  <https://academy.ftmo.com/lesson/maximum-daily-loss/> (WebSearch
  excerpt, 2026-05-15)
- FTMO *Can I trade news?*
  <https://ftmo.com/en/faq/can-i-trade-news/> (WebSearch excerpt,
  2026-05-15)
- FTMO *Forbidden Trading Practices*
  <https://ftmo.com/en/forbidden-trading-practices/> (WebSearch
  excerpt, 2026-05-15)
- The5%ers *Instant Funding Rules*
  <https://the5ers.com/your-guide-to-passing-the-5ers-evaluation-program/>
  (WebSearch excerpt, 2026-05-15)
- E8 Markets *Rules 2026*
  <https://www.eafunded.com/firms/e8-markets> (WebSearch excerpt,
  2026-05-15)
- FundedNext *Is EA allowed?*
  <https://help.fundednext.com/en/articles/8020763-is-ea-allowed-in-fundednext>
  (WebSearch excerpt, 2026-05-15)
- eToro *CopyTrader — How It Works*
  <https://www.etoro.com/copytrader/how-it-works/> (WebSearch
  excerpt, 2026-05-15)
- eToro *CopyTrading Risks*
  <https://www.etoro.com/customer-service/copytrading-risks/>
  (WebSearch excerpt, 2026-05-15)
- eToro *Default Copy Stop Loss*
  <https://help.etoro.com/s/article/What-is-the-default-Copy-Stop-Loss-for-CopyTrader-and-Smart-Portfolios>
  (WebSearch excerpt, 2026-05-15)
- Interactive Brokers *TWS Order Presets — Precautionary Settings*
  <https://www.interactivebrokers.com/en/trading/tws-order-presets.php>
  (WebSearch excerpt, 2026-05-15)
- Interactive Brokers *Define Precautionary Settings* (TWS guide)
  <https://www.interactivebrokers.co.uk/en/software/tws.bak/usersguidebook/configuretws/define_precautionary_settings.htm>
  (WebSearch excerpt, 2026-05-15)
- Interactive Brokers *From Paper Trading To Real Trading*
  <https://www.interactivebrokers.com/campus/ibkr-quant-news/from-paper-trading-to-real-trading-monitoring-debug-and-go-live/>
  (WebSearch excerpt, 2026-05-15)
- IBKR *Paper Trading vs Live Trading*
  <https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/>
  (WebSearch excerpt, 2026-05-15)
- Robinhood *Instant Deposits and Options*
  <https://robinhood.com/us/en/support/articles/instant-deposits-and-options/>
  (WebSearch excerpt, 2026-05-15)
- Robinhood *Options Knowledge Center*
  <https://robinhood.com/us/en/support/articles/options-knowledge-center/>
  (WebSearch excerpt, 2026-05-15)
- FINRA Robinhood AWC 2021
  <https://www.finra.org/sites/default/files/2021-06/robinhood-financial-awc-063021.pdf>
  (WebSearch excerpt, 2026-05-15)
- Webull *Disclosures*
  <https://www.webull.com/disclosures> (WebSearch excerpt,
  2026-05-15)
- Binance *Complete Entity Verification*
  <https://www.binance.com/en/support/faq/detail/360015552032>
  (WebSearch excerpt, 2026-05-15)
- Coinbase Help — *What is Coinbase Advanced?*
  <https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/what-is-advanced-trade>
  (WebSearch excerpt, 2026-05-15)
- Kraken *Verification Levels & Features*
  <https://support.kraken.com/hc/en-us/sections/360000259346-Verification-levels-features>
  (WebSearch excerpt, 2026-05-15)
- Wealthfront *Risk Questionnaire*
  <https://www.wealthfront.com/risk-questionnaire> (WebSearch
  excerpt, 2026-05-15)
- 3Commas *DCA Bot*
  <https://3commas.io/dca-bots> (WebSearch excerpt, 2026-05-15)
- 3Commas *Grid Bot*
  <https://3commas.io/grid-bot> (WebSearch excerpt, 2026-05-15)
- TakeProfitTrader *Maximum Trailing Drawdown*
  <https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15170265979165-Rule-3-Do-Not-Hit-End-Of-Day-EOD-Maximum-Trailing-Drawdown>
  (WebSearch excerpt, 2026-05-15)
- Tradeify *Daily Loss Limit*
  <https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit>
  (WebSearch excerpt, 2026-05-15)
- FunderPro *News Trading*
  <https://funderpro.com/blog/news-trading-in-prop-firms-what-rules-you-must-follow-to-avoid-violations/>
  (WebSearch excerpt, 2026-05-15)
- BabyPips *Always Know Your Risk Exposure*
  <https://www.babypips.com/learn/forex/always-know-your-risk-exposure>
  (WebSearch excerpt, 2026-05-15)
- FXOpen *Margin Call and Stop Out*
  <https://support.fxopen.com/portal/en/kb/articles/margin-call-and-stop-out-6-8-2021>
  (WebSearch excerpt, 2026-05-15)
- EarnForex *AutoTrading Scheduler*
  <https://www.earnforex.com/metatrader-expert-advisors/AutoTrading-Scheduler/>
  (WebSearch excerpt, 2026-05-15)
- FIX *Session Layer Online*
  <https://www.fixtrading.org/standards/fix-session-layer-online/>
  (WebSearch excerpt, 2026-05-15)
- NN/G *Progressive Disclosure*
  <https://www.nngroup.com/articles/progressive-disclosure/>
  (WebSearch excerpt, 2026-05-15)
- IxDF *Progressive Disclosure*
  <https://ixdf.org/literature/topics/progressive-disclosure>
  (WebSearch excerpt, 2026-05-15)

### 0.2 Citation index — internal

- `docs/audits/research/installer_wizard_ux_spec.md` — current
  10-step wizard
- `docs/audits/research/ctrader_api_full_reference.md` — payload
  IDs / rate limits we can call from each step
- `docs/audits/research/ui_ux_design_spec.md` — TradingView + cTrader
  palette and panel idioms
- `crates/forex-core/src/contracts/temporal.rs` — 11 canonical
  timeframes, NO H2
- `crates/forex-core/src/domain/prop_firm.rs` —
  `PropFirmConstraints::FTMO_STANDARD`, 4 % monthly floor
- `crates/forex-app/src/app_services/ctrader_live_auth.rs` — OAuth
  loopback flow

### 0.3 What this audit is *not*

- It is not a feature-by-feature clone list. It distils patterns;
  forex-ai will not implement, e.g., a copy-trader marketplace.
- It does not propose abandoning the existing 10-step wizard; it
  proposes **additions, refinements, and a new branch** for the
  one-button autonomous mode the operator named.
- It does not change operator-locked invariants: 11 canonical
  timeframes (no H2), 4 % monthly **floor** (not ceiling) for
  Production mode, no synthetic data, no hardcoded values except
  `PropFirmConstraints::FTMO_STANDARD` + the 4 % floor.

---

## §1 — Retail charting & analytics platforms

### 1.1 TradingView

TradingView's onboarding pivots on the "Trading Panel" — a dock at
the bottom of any chart that holds the broker connection. From the
broker-integration spec:

> "Open the Trading Panel (click the arrow at the bottom of the
> platform). Choose your brokerage from the list of supported
> providers. Log in using your broker's credentials. Follow any
> additional verification steps your broker requires."
> (TradingView brokerage-integration page, WebSearch excerpt,
> 2026-05-15.)

The broker selection screen is itself the wizard, and the **first
mandatory choice on the login modal is Live vs Paper**:

> "Choose Live Trading or Paper Trading above the username field.
> The login popup changes color to help ensure you're correctly
> logging into your desired mode: **gray for live trading, red for
> simulated paper trading**." (Supa.is recap of TradingView's
> Trading Panel, WebSearch excerpt 2026-05-15, quoting the
> TradingView support article cited above.)

After login an agreement screen forces a typed signature:

> "Sign the agreement by typing your name and clicking I Agree."
> (Same source.)

For Interactive Brokers specifically TradingView even time-boxes
the link:

> "users who verified market data access receive 7 days of access
> (as of 2026-04) and then need to reconnect the broker account
> again in the Trading Panel." (WebSearch excerpt of TradingView
> support, 2026-05-15.)

**Patterns to lift:**

- **Live/Paper colour code at login.** The mode is part of the
  login chrome, not buried in settings.
- **Typed agreement signature** as the gate to first real-money
  action — heavier than a checkbox.
- **Broker-side time-boxing** of session credentials. Spotware
  refresh-token lifetime is owned by the broker; our wizard can
  surface "Token valid for N days" and re-prompt before expiry
  (see `ctrader_api_full_reference.md` §2.5).

### 1.2 MetaTrader 5

MT5's onboarding *is* a wizard. The official Help page describes
the first screen verbatim:

> "A broker is selected during the first step. If the desired
> company is not shown in the list, you can type its name and click
> 'Find your broker'. Alternatively, you can type the address of the
> server instead of the company name. Once you find the desired
> company, select it and click 'Next'." (MetaTrader 5 Help — *Open
> an Account*, WebSearch excerpt 2026-05-15.)

The second screen branches into three account types, again verbatim:

> "1. **Existing Account**: You will need to specify the account
> number, the password and the server name.
> 2. **Demo Account**: Demo accounts help users learn trading and
> test trading strategies. All trading operations only involve
> virtual money.
> 3. **Real Account**: Select this option to request opening of a
> real account. Trading operations are performed using real money
> on such accounts, therefore you will need to provide broker
> detailed information about yourself, as well as ID and proof of
> address." (Same source.)

The final screen surfaces the credentials inline:

> "Once an account is created on a selected server, details will be
> shown in the dialog window including the Login (account number)
> and Password (a master password, which allows trading from this
> account)." (Same source.)

The MetaEditor *MQL5 Wizard* — a separate wizard for strategy
authoring — also forces parameter discovery up-front:

> "Mandatory parameters created by default include Symbol – where
> you specify a symbol the EA is to work on in the Value field. If
> 'current', the EA works on any symbol. TimeFrame parameters allow
> you to specify a period the EA is to work on, and if 'current' is
> selected, the EA works on any chart period." (MetaEditor Help —
> *Creating a ready-made Expert Advisor*, WebSearch excerpt,
> 2026-05-15.)

**Patterns to lift:**

- **Branch on Demo / Existing / Real at step 2**, not as a hidden
  toggle. The forex-app wizard currently makes "Trading mode" a
  three-way radio inside Step 3 — closer to the MT5 branch than
  TradingView's binary, which is correct.
- **Symbol/timeframe defaults are explicit and editable**, with
  "current" as the catch-all. Our Step 5 already does this; the
  affordance for "leave at defaults" must be obvious.
- **Show the credentials at the end of the relevant step** so the
  user can copy them. For us this maps to surfacing the chosen
  `ctidTraderAccountId` at the end of Step 4.

### 1.3 NinjaTrader 8

NinjaTrader ships a wizard whose *first connection* is always a
simulated one:

> "The Simulated Data Feed connection is a default connection
> installed with NinjaTrader, and its purpose is to play
> internally generated market data for simulation." (NinjaTrader 8
> *Simulated Data Feed Connection*, WebSearch excerpt 2026-05-15.)

Crucially, the multi-provider toggle is itself behind a wizard:

> "To access the Simulated Data Feed connection, you need to turn
> on Multi-provider mode first by going to 'Tools' -> Options and
> making sure that the 'Multi Providers' option is enabled."
> (Same source.)

And a deliberately separate **Trend slider** that lets the user
drag synthetic price up or down for testing:

> "The Trend slider control will appear once connected to the
> Simulated Data Feed, and you can left mouse click on the slider
> and drag it up or down to cause the Simulated Data Feed to move
> in that direction." (Same source.)

**Patterns to lift:**

- **Default to Simulation, opt-in to Live.** A fresh install of
  NinjaTrader cannot accidentally route a real-money order — the
  simulated feed is wired first, and Live is a separate, named
  connection. Our forex-app wizard's Step 3 trading-mode default
  is "Forward test" which is the equivalent, but we should make
  the Live radio physically distant or two-step gated.
- **Synthetic *user-driven* test feed is a development tool, not
  a market datum.** NinjaTrader's Trend slider is fine because the
  user is the source of the move and the price is labelled
  "Simulated Data Feed". Our no-synthetic-data rule is about *real
  symbols with fabricated bars passed off as broker data* — that
  is forbidden; an explicit "Replay" mode is not.

### 1.4 ThinkOrSwim (Charles Schwab)

ThinkOrSwim's celebrated **paperMoney** toggle is one click from
anywhere:

> "Switching between Paper Money and Live is a single click. You
> can toggle between live trading and paper trading on the platform
> by clicking on the Trade button and selecting either Live Trading
> or paperMoney." (Schwab Learn — *thinkorswim paperMoney*,
> WebSearch excerpt 2026-05-15.)

> "On mobile devices, to switch to paper trading on the app, tap on
> the More icon on the bottom right corner of the screen and select
> paperMoney." (Same source.)

The Schwab Learn page also pitches paperMoney explicitly as the
**week-one default for new users**:

> "You can use paperMoney for several goals, including practicing
> on the platforms tools, practicing your trading strategy, and
> learning about advanced order entry tools before employing them
> in your live account. You can spend your first week or two here
> to learn the interface, test strategies, and practice multi-leg
> options orders before touching real capital." (Same source.)

**Patterns to lift:**

- **A persistent global toggle** between paper and live (think:
  status-bar pill) is more discoverable than a settings menu.
- **A documented "first week here, then live" suggestion** in
  copy. Our wizard's Step 10 currently launches the main app with
  "Run your first backtest" as the tour — adding "spend your first
  N days in Forward test before unlocking Live" copy is cheap and
  high-impact (see §6 below).

### 1.5 tastytrade

tastytrade gates **portfolio margin** behind a heavy multi-step
flow:

> "To apply for portfolio margin, you must answer questions related
> to your trading history and intentions, and thoroughly read
> through the Portfolio Margin Customer Agreement and Portfolio
> Margin Risk Disclosure Statement before proceeding. After
> reviewing your application information and the disclosure
> statements, if you agree to the conditions, you check a box
> acknowledging the legally binding signature, and once submitted,
> tastytrade will review your application within 3-5 business
> days." (tastytrade support — *How to Apply for Portfolio Margin*,
> WebSearch excerpt 2026-05-15.)

Equity-floor enforcement is bolted to the same flow:

> "tastytrade requires a minimum initial equity value of $125,000
> to begin trading with portfolio margin requirements, and a
> minimum maintenance equity value of $100,000 at any given time."
> (Same source.)

And — most relevant for our auto-flatten primitive — the broker
itself holds the kill switch:

> "tastytrade and the Clearing Firm reserve the right to liquidate
> positions in your PM Account to the extent necessary to eliminate
> the margin deficiency." (Same source.)

**Patterns to lift:**

- **Per-feature suitability gate**, not blanket "you're approved".
  Live trading, options, portfolio margin and futures are each a
  separate application.
- **Server-side auto-liquidation** is the canonical kill switch.
  Our forex-app cannot make the broker liquidate, but we can mirror
  the pattern client-side (auto-flatten on threshold).

---

## §2 — Algo / strategy platforms

### 2.1 cTrader cAlgo & cBots

The bot-import wizard is two clicks — the doc is direct:

> "To run a trading bot in the active symbol chart, click the Add
> cBot icon in the chart toolbar, choose a cBot from the list,
> specify your preferred instance parameters if applicable, then
> click Add to chart. After the cBot instance appears on the chart,
> click the Start cBot button." (cTrader Help — *Manage cBots*,
> WebSearch excerpt 2026-05-15.)

For users who want to author one from scratch the path is:

> "Go to the Algo app and click the New button under the cBots
> tab, then select the Blank option, enter a name, and click
> Create." (Same source.)

**Patterns to lift:**

- **Two-pane "list + instance parameters" wizard** for any
  user-instantiable object. We can re-use this for strategy
  selection in forex-app: left pane = strategy templates, right
  pane = parameters with defaults.
- **Explicit "Start cBot" button after Add.** No auto-start.
  Loading ≠ running. This is exactly the safeguard our wizard
  needs before any auto-trade mode goes live.

### 2.2 MetaTrader EA marketplace + MQL5 Wizard

Installation flow (MT5 marketplace):

> "A setup pop-up appears when dragging an EA onto a chart, where
> you can check the Inputs tab to verify your risk factors, then
> check the boxes marked Allow Modification of Signal Settings and
> Allow Algo Trading." (FOREX.com EA installation guide quoting
> MT5, WebSearch excerpt 2026-05-15.)

> "You can select the 'Inputs' tab where you can modify/change EA
> parameters. You can load or save EA settings parameter by
> clicking 'Load' or 'Save', and restore default EA settings by
> clicking 'Reset'." (Same.)

MT5 also has a **global panic button**:

> "MetaTrader has a button in the menu upper part called
> 'AutoTrading' that has 2 states: in green, it indicates that
> automatic trading is activated, and in red indicates that
> automatic trading is deactivated. You can use the autotrading
> button such as the called 'Panic button', it can stop all the
> operations in one click." (TradeAsy help, quoting MT4/5 chrome,
> WebSearch excerpt 2026-05-15.)

**Patterns to lift:**

- **Save / Load / Reset preset pattern** on every parameterised
  object. forex-app strategies should have this.
- **One global red "Stop all automation" button** in the
  always-visible chrome — not in a menu.

### 2.3 QuantConnect

QuantConnect's wizard pivoted from a one-shot creator to a
**strategy-builder + module library**:

> "QuantConnect has created an algorithm creation wizard for laying
> the groundwork for your algorithm, which lets you pull in
> assorted modules and quickly assemble them into auto-coded
> scaffolding." (QuantConnect blog — *Algorithm Lab 2.0*,
> WebSearch excerpt 2026-05-15.)

> "The Algorithm Wizard has been replaced with the new Strategy
> Builder. To add a framework module, click Add Module. To start
> from scratch with the IDE, click the Exit Builder Mode button."
> (Same source.)

The Strategy Library itself is one of the most-mature template
galleries in retail algo:

> "QuantConnect offers a collection of tutorials written by the
> team and community members to learn about trading strategies in
> the literature." (QuantConnect — *Strategy Library*, WebSearch
> excerpt 2026-05-15.)

**Patterns to lift:**

- **Builder mode vs IDE mode.** New users get the Builder
  (template-driven); power users escape to the raw config. forex-ai
  should expose the same dual entry: wizard-driven preset for new
  users, `config.yaml` for power users.
- **"Exit Builder Mode" must be reversible** — re-entering the
  wizard from raw config should not destroy hand edits. Our wizard's
  re-run pre-populates from `wizard_complete.json`
  (`installer_wizard_ux_spec.md` §5.1); we must extend this so
  config-yaml diff is preserved when re-running.

### 2.4 TradeStation EasyLanguage

EasyLanguage's bootstrap is template-first:

> "To create a new EasyLanguage strategy, click File – New –
> Window, select the EasyLanguage tab and click 'strategy'. Use
> step-by-step prompts to define inputs, calculations, and display
> settings, then once your strategy is saved, add it to a chart or
> RadarScreen to monitor its outputs." (TradeStation *Learning
> EasyLanguage Strategies* PDF, WebSearch excerpt 2026-05-15.)

The order-verb uniformity is worth a note:

> "EasyLanguage uses four order verbs to generate strategy orders
> that are uniform across all asset types: Stocks, Futures, Forex,
> and Options." (Same.)

**Patterns to lift:**

- **Uniform order primitives** across instruments. forex-app is
  forex-only, so this is a free win — our New / Cancel / Amend /
  ClosePosition surface from `ctrader_api_full_reference.md` §4.6–
  §4.8 already maps cleanly.
- **Step-by-step prompts define inputs, calculations, display.**
  This is the spine of any strategy-creation wizard.

### 2.5 Alpaca

Alpaca is the clearest API-key + mode-switch wizard among the
modern platforms:

> "Your paper trading account will have a different API key from
> your live account, and all you need to do to start using your
> paper trading account is to replace your API key and API
> endpoint with ones for the paper trading." (Alpaca docs —
> *Paper Trading*, WebSearch excerpt 2026-05-15.)

> "To use the paper trading api, set APCA-API-KEY-ID and
> APCA-API-SECRET-KEY to your paper credentials, and set the
> domain to paper-api.alpaca.markets. After you have tested your
> algo in the paper environment and are ready to start running
> your algo in the live environment, you can switch the domain to
> the live domain, and the credentials to your live credentials."
> (Same source.)

> "On your dashboard in the top left, you can select between using
> your paper trading or live accounts. It's important that you
> select the appropriate trading account before implementing a
> trading strategy." (Same.)

**Patterns to lift:**

- **Separate keys per environment** is a strong invariant. cTrader
  conflates "live" and "demo" under one client app — but each
  `ctidTraderAccountId` is environment-specific
  (`ctrader_api_full_reference.md` §2.6). Our wizard's Step 4.3
  account picker already exposes Live vs Demo per row; we should
  default-select Demo when both are present.
- **Explicit "select trading account before strategy" copy.** A
  banner in the main app: "Active account: Demo #12345. Strategies
  run against this account." Matches Alpaca's dashboard top-left.

### 2.6 QuantRocket

QuantRocket's first-run path is dominated by Docker setup, then a
JupyterLab "house" of notebooks. The search index returned only
one related thread (a login loop) — the public docs are not in
WebFetch range of this sandbox. I record this in §11 (Open
Questions) and do not derive patterns from speculation. A
fallback pattern observation: **JupyterLab as the post-onboarding
home** is a niche that doesn't apply to forex-ai (we have a native
egui app).

---

## §3 — Crypto exchanges (compare-and-contrast)

### 3.1 Binance

Binance is the canonical KYC ladder:

> "Binance provides three verification levels: Basic (Name,
> address, date of birth, and nationality with a lifetime limit
> of $300), Intermediate (Government-issued photo identification
> such as a passport, national identity card, or driver's
> license), and Advanced (Proof of address)." (Westafrica-Crypto-Hub
> summary of Binance KYC, citing Binance support FAQ, WebSearch
> excerpt 2026-05-15.)

> "Binance now strictly requires all users to complete KYC before
> trading or using fiat gateways, which helps prevent fraud, money
> laundering, and account theft." (Smalldrift recap, same.)

**Patterns to lift:**

- **Tiered unlocks** are the natural mapping of risk + capability.
  Forex-ai does not do KYC (we're a client, not a broker), but
  tiered unlocks **of automation capabilities** (Stage 1–4 in §6)
  is the directly-analogous design.

### 3.2 Coinbase — beginner vs Advanced Trade

Coinbase ships **two product surfaces** behind one toggle:

> "Users can toggle between 'Simple' and 'Advanced' views,
> allowing seamless switching depending on their trading task."
> (Coinbase help — *What is Coinbase Advanced?*, WebSearch
> excerpt 2026-05-15.)

> "Traders can enter Coinbase Advanced via a global toggle switch
> from the user profile to access advanced trading tools and
> features." (Same source.)

**Patterns to lift:**

- **One-toggle skill-level branch.** Beginner mode hides
  derivatives, advanced order types, and full Polars/Parquet
  inspection panels. Forex-ai's main app should expose a
  "Beginner / Advanced" toggle in the top-right that hides the
  raw research panes — and the wizard should ask which mode to
  start in (Step 3 extension; see §9).

### 3.3 Kraken — verification tiers as feature gates

> "Kraken has four verification levels: Starter, Express,
> Intermediate, and Pro, though the Express is only available to
> US citizens." (Kraken support — *Verification levels*, WebSearch
> excerpt 2026-05-15.)

> "Starter Tier: Offers limited access; no deposit or withdrawal
> capabilities and minimal trading features. Intermediate Tier:
> Intermediate accounts allow for unlimited cryptocurrency
> deposits and higher daily withdrawal limits, along with access
> to fiat currency deposits and withdrawals … margin and futures
> trading, as well as staking services. Pro Tier: Aimed at
> advanced traders and institutional clients, the Pro account
> offers the highest limits and access to exclusive features."
> (Same source.)

**Patterns to lift:**

- **Each tier unlocks *features*, not just limits.** A "Starter"
  forex-ai user could see backtesting + paper trading; "Intermediate"
  unlocks Live; "Pro" unlocks autonomous mode (§10). This is the
  staging path the operator's directive implies.

---

## §4 — Prop-firm dashboards

### 4.1 FTMO

FTMO is our **canonical reference** (operator policy at
`crates/forex-core/src/domain/prop_firm.rs:32` codifies
`FTMO_STANDARD`). The Trading Objectives page is verbatim:

> "Maximum Daily Loss: 5%." (FTMO *Trading Objectives*, WebSearch
> excerpt 2026-05-15.)

> "For the 2-Step Challenge: The maximum daily loss is always 5%
> of the initial balance of the account. For the 1-Step Challenge:
> The Maximum Daily Loss Amount is 3% of the Initial Simulated
> Capital." (FTMO Help, WebSearch excerpt 2026-05-15.)

The reset / kill-switch cadence is spelled out:

> "The Maximum Daily Loss Limit is recalculated daily at 00:00
> CE(S)T as the difference between: the account balance recorded
> at 00:00 CE(S)T of the current day and the Maximum Daily Loss
> Amount, which is 5% of the Initial Simulated Capital." (FTMO
> Academy — *Maximum Daily Loss*, WebSearch excerpt 2026-05-15.)

And the **news-trading restriction** that funded accounts inherit
on top of the daily loss:

> "Restrictions for trading during selected news releases apply
> only to the Standard account type. The Swing account type have
> no restrictions on trading during news releases. You are allowed
> to hold open positions on the targeted instruments if they were
> opened more than 2 minutes before the restricted event. Please
> note that if a Stop Loss or Take Profit is triggered within the
> restricted time window, this will also be considered a breach
> of the FTMO Account Agreement." (FTMO — *Can I trade news?*,
> WebSearch excerpt 2026-05-15.)

The forbidden-practices list is operator-relevant copy:

> "Forbidden Trading Practices" page calls out hedge-on-account,
> tick-scalping, latency arbitrage and HFT abuse. (FTMO
> *Forbidden Trading Practices*, WebSearch excerpt 2026-05-15.)

**Patterns to lift:**

- Our existing `FTMO_STANDARD` already encodes 5 % daily / 10 %
  overall / 10 % target / 4 % monthly floor / 10 trading-day
  minimum. The wizard's Step 3 surfaces this; nothing to add at
  the values.
- We **do not** yet model the **00:00 CE(S)T reset cadence**
  (this is a runtime invariant). Step 3 of the wizard should let
  the user pick the reset timezone (default CE(S)T to match FTMO),
  even though the actual enforcement is a runtime concern.
- We **do not** yet have a **news-blackout primitive** wired to
  Step 8 of the wizard (news/sentiment provider). The blackout
  rule is the same shape across FTMO + E8 + FunderPro — see §7.

### 4.2 The5%ers — instant funding

> "The Instant Funding program has no challenge or evaluation —
> you pay the fee, get funded immediately, and start trading real
> capital on day one." (The5%ers — *Rules & Tips to Succeed*,
> WebSearch excerpt 2026-05-15.)

> "For most Instant Funding accounts, the maximum overall drawdown
> is a static 5-6% of the initial balance. On a $20,000 account
> with a 5% max drawdown, your account equity can never drop
> below $19,000. If it does, the account is closed." (Same.)

And critically:

> "The5ers often mandates a stop-loss on every single position,
> with a maximum risk per trade (e.g., 1.5%). The 1.5% stop-loss
> rule means your risk on any single trade cannot exceed $300."
> (Same source.)

**Patterns to lift:**

- **Stop-loss is mandatory on every order.** This is operator-
  consistent: when in `PropFirmConstraints::FTMO_STANDARD` mode,
  the wizard's Step 3 should toggle a flag "Require SL on every
  order" defaulting on. Order placement without an SL is rejected
  at the order-router boundary.
- **Per-trade max loss is a hard limit.** 1.5 % default; capped
  by `max_daily_loss_pct=0.05` (`prop_firm.rs:32`). Surface a
  new "Per-trade max risk %" slider in Step 3 (default 1.0 %,
  range 0.1–2.0 %).

### 4.3 E8 Markets, FundedNext, others

E8 Markets is the clearest source on the news-window length:

> "EAs are allowed, but traders cannot run identical strategies
> across multiple accounts; each trader is restricted to one
> trading strategy per account. To prevent high-frequency trading
> (HFT) abuse, over 50% of trades must remain open for at least
> one minute. News trading is fully permitted on evaluation
> accounts, but on funded accounts trading is not allowed during
> the window of 5 minutes before to 5 minutes after a
> high-impact news release." (eafunded.com summary of E8 Markets
> rules, WebSearch excerpt 2026-05-15.)

FundedNext:

> "FundedNext allows the use of Expert Advisors (EAs), indicators,
> and trading bots on MetaTrader 4 (MT4) and MetaTrader 5 (MT5)
> platforms … FundedNext does not allow the use of EAs or
> automated trading on the cTrader platform." (FundedNext Help
> Center, WebSearch excerpt 2026-05-15.)

The last point matters: we are an **automated cTrader client**.
Operators using forex-ai on FundedNext via cTrader would violate
ToS. The wizard should call this out in Step 3 next to the
prop-firm preset dropdown:

> Inline warning: "FundedNext bans automated trading on cTrader.
> Select another firm preset or use the Live MT4/MT5 bridge (not
> yet implemented)."

**Patterns to lift:**

- **Per-firm ToS surface in the wizard.** Each prop-firm preset
  carries: max-daily-loss, max-overall-drawdown, monthly-target,
  trading-days-minimum, news-window-pre, news-window-post,
  HFT-rule (min-trade-duration), EA-allowed-flag,
  cTrader-allowed-flag, strategy-uniqueness-flag.
- **No silent enforcement.** When the user toggles to a firm that
  bans automation on cTrader, the wizard refuses to enable Live
  Auto, not just shows a banner.

### 4.4 Tradeify, TakeProfitTrader — daily-loss vs trailing drawdown

Tradeify spells out the soft-pause:

> "The Daily Loss Limit pauses your trading for the day when you
> reach a specified loss threshold. Unlike the Max Trailing
> Drawdown, hitting the DLL does NOT fail your account — you can
> resume trading the next session." (Tradeify Help — *Daily Loss
> Limit*, WebSearch excerpt 2026-05-15.)

TakeProfitTrader spells out the hard-stop:

> "The Max Trailing Drawdown is a 'hard breach' — if your account
> balance drops to or below your drawdown limit at ANY point, your
> account fails immediately. This is different from the Daily Loss
> Limit (soft breach) which only pauses trading. Importantly, the
> drawdown level is enforced in real time at all times, even on
> EOD accounts. If your balance touches the current drawdown
> level during a session, the account closes immediately
> regardless of what happens later that day." (TakeProfitTrader
> Help — Rule 3, WebSearch excerpt 2026-05-15.)

**Patterns to lift:**

- **Two kill switches: soft (pause-for-day) and hard
  (account-closed).** Forex-ai already has both as concepts in
  `FTMO_STANDARD` (`max_daily_loss_pct=0.05` is the soft;
  `max_overall_drawdown_pct=0.10` is the hard). The wizard should
  surface them with the same vocabulary FTMO/Tradeify use, so
  the user recognises them: "Daily-loss limit (soft pause for
  the rest of the day)" + "Overall drawdown (account closed)".

---

## §5 — "One-button live trading" patterns

The operator's directive names "live trading με ένα κουμπί" — live
trading with one button. Here is how the giants surround that
button so it isn't a footgun.

### 5.1 eToro CopyTrader

CopyTrader is *the* one-button trading product:

> "The setup process involves selecting your criteria to find an
> investor, reviewing their portfolio, performance, and strategy,
> hitting the COPY button, and choosing how much to allocate,
> which duplicates their positions automatically in real time and
> in direct proportion." (eToro — *Copy Systems Explained*,
> WebSearch excerpt 2026-05-15.)

The safeguards layered on top are explicit:

> "Copy Stop-Loss (CSL) is a built-in risk management feature
> that lets you set a stop-loss percentage across the entire copy
> trade to help protect your investment based on real-time profit
> and loss. If the value of your copy drops below the set amount,
> the system will automatically close the copy and return the
> remaining funds to your balance. By default, CSL is set to 40%
> of your invested amount, and you can manually adjust CSL to any
> value between 5% and 95% of your invested amount." (eToro Help
> — *Default Copy Stop Loss*, WebSearch excerpt 2026-05-15.)

> "Every Popular Investor has a daily risk score from 1 (lowest)
> to 10 (highest), with a score of 4-6 being moderate, and you
> should be cautious with anyone consistently above 7."
> (eToro — *CopyTrading Risks*, WebSearch excerpt 2026-05-15.)

> "Users may not be copied if the risk score associated with their
> account hits a certain threshold or if they have a certain
> amount of copiers or assets already." (Same source.)

> "The minimum amount required to copy a trader is $200." (Same.)

**Safeguards summary** (from eToro, mapped to our model):

| eToro primitive | Forex-ai equivalent / Step |
|----|----|
| Copy Stop-Loss 40 % default | Equity-stop on autonomous mode (§10) |
| Daily risk score 1–10 | Strategy "risk score" surfaced in Step 5 (templates) |
| Cap on copiers per provider | Position-correlation cap (§7) |
| $200 minimum allocation | "Capital-at-risk floor" disclosure (§8.2) |

### 5.2 Interactive Brokers TWS

IBKR's one-click trading is layered with **Precautionary Settings**
— a per-order safety check:

> "Precautionary values are used by the system as safety checks.
> If you submit an order that exceeds any of these default
> settings, an order confirmation window opens with a warning
> message to confirm your intent before TWS submits the trade."
> (IBKR — *TWS Order Presets*, WebSearch excerpt 2026-05-15.)

> "Precautionary settings include size limit, total value limit,
> and percentage constraint." (Same.)

> "If you would like to disable any of the precautionary settings,
> enter Zero in the boxes for the different limits." (Same.)

**Patterns to lift:**

- **Per-order safety check above thresholds.** Inside our
  autonomous mode this maps to: "Auto-confirm trades under N USD
  notional; otherwise pop a confirmation."
- **Zero = disabled** is a clear UX — it lets us replace radio
  buttons with numeric inputs that have an implicit "off".

### 5.3 Webull

Webull is the canonical one-tap mobile order:

> "When you submit your order, there is an order confirmation for
> the order details, you need to tap the confirm to finish your
> order. You can turn it off if you want." (Bitget Wiki recap of
> Webull docs, WebSearch excerpt 2026-05-15.)

And the disclosure is explicit:

> "For Mobile Traders: Use limit orders in low-liquidity
> situations and double-check quantities to avoid accidental large
> sales." (Webull *Disclosures*, WebSearch excerpt 2026-05-15.)

**Patterns to lift:**

- **User-disableable confirmation step** — but the *default* is
  on, and the disclosure copy stays even after the user opts out.
  The disclosure is in the platform's compliance posture, not the
  per-order modal.

### 5.4 Robinhood — instant deposit and options approval

Robinhood layers approval before unlock:

> "Robinhood will ask a series of questions about your trading
> experience, income, net worth, and investment goals … you'll
> need to complete the investor profile questionnaire providing
> experience level, annual income, net worth, employment status,
> and risk tolerance, and indicate the types of strategies you
> wish to trade." (Option Alpha summary of Robinhood options
> onboarding, WebSearch excerpt 2026-05-15.)

> "With Level 2 approval, you'd have access to some basic
> strategies, and with Level 3 approval, you'd have access to
> everything available with Level 2 approval and a few more
> advanced strategies." (Same.)

The FINRA settlement is the *cautionary* part of this pattern:

> FINRA found that "since Robinhood began offering options trading
> to customers in December 2017, the firm has failed to exercise
> due diligence before approving customers to trade options."
> (FINRA AWC for Robinhood Financial, 2021-06-30, WebSearch
> excerpt 2026-05-15.)

**Patterns to lift:**

- **Approval-level gates are themselves a compliance surface.** A
  trading system that *can* execute an order must verify the user
  is approved to do so. For forex-ai this maps to: the wizard's
  Step 3 monthly-target slider already enforces 4 % floor; the
  autonomous-mode unlock (§10) requires checkpoints (completed
  paper-trade phase + acknowledged risk warnings) before the Live
  button is clickable.
- **Don't be Robinhood.** "Did the user click 'I understand'?" is
  not enough. The wizard should record acknowledgement
  hashes + timestamps and refuse to unlock a higher tier without
  evidence that the previous tier was used.

---

## §6 — Demo → Paper → Live progression

This is the spine that all the platforms above implement, with
different vocabulary but identical shape.

### 6.1 Canonical four-stage staging

| Stage | Data | Execution | Capital | Primary signal | Citation |
|---|---|---|---|---|---|
| 1 — Demo / replay | Historical bars | Simulated fills | None | Backtest equity curve, Sharpe ≥ X | NinjaTrader Simulated Data Feed; MT5 *Strategy Tester* |
| 2 — Paper trading | Live bars | Simulated fills | None | Forward-test P&L, slippage estimate, win-rate | TradingView "Paper Trading" mode in Trading Panel; ThinkOrSwim paperMoney; Alpaca paper API |
| 3 — Live small account | Live bars | Real fills | Capped (e.g. demo-graduates to $200 minimum like eToro) | Real P&L over N days; ≥ 4 % monthly target hit | The5%ers Instant Funding $20k starter; FTMO $10k Challenge |
| 4 — Live full account | Live bars | Real fills | Uncapped | Steady ≥ 4 % monthly net profit, sub-daily-loss-limit, ≥ 10 trading days | FTMO Trading Objectives; Apex EOD account |

### 6.2 Gating between stages — operator-relevant criteria

For forex-ai, *staging is a wizard branch, not just settings*.
Here's the per-stage promotion-gate proposal:

- **Stage 1 → 2** (Demo → Paper): backtest passes on ≥ 6 months
  history (Step 6 download seeded), Sharpe-or-equivalent ≥ 1.0
  on at least one symbol/timeframe pair, and user clicks
  "Promote".
- **Stage 2 → 3** (Paper → Live small): ≥ 10 trading days in
  paper, ≥ 4 % monthly net profit hit, max-daily-loss never
  breached, **AND** user acknowledges live-trading disclosure
  (typed signature à la TradingView §1.1).
- **Stage 3 → 4** (Live small → Live full): ≥ 30 days at Stage 3,
  ≥ 4 % monthly net profit hit, equity never below soft daily-loss
  limit. (The 4 % monthly floor is operator-locked at
  `prop_firm.rs:36`; we use that as the per-stage threshold.)

Each gate writes a record into `wizard_progress.json` with the
checkpoint hash + timestamp — the same schema as
`installer_wizard_ux_spec.md` §5.

### 6.3 Why this is a wizard concern

The current wizard ends at Step 10 (Summary & Apply) with the
user in **Step 2 (Paper)** by default (Step 3.4 of the spec sets
`trading_mode = forward`). The wizard does NOT today:

- Surface the four-stage staging diagram.
- Record graduation evidence between stages.
- Block Live unlock until Stage 3 evidence exists.

§9 below proposes a **new Step 9.5 — "Stage roadmap"** and
expands Step 10's "Apply" outputs accordingly.

---

## §7 — Auto-trade safeguards big firms ship

A consolidated table; each entry is **a guardrail that survives
across multiple platforms**. forex-ai is required to implement
the ones marked **must-have** to be FTMO-safe.

| Guardrail | Citation | Forex-ai status | Must-have? |
|---|---|---|---|
| **Daily-loss kill switch** (soft pause) | FTMO 5 %; Tradeify DLL | `max_daily_loss_pct=0.05` in `FTMO_STANDARD` | Yes |
| **Overall drawdown kill switch** (hard close) | FTMO 10 %; TakeProfitTrader Max Trailing | `max_overall_drawdown_pct=0.10` in `FTMO_STANDARD` | Yes |
| **Per-trade max loss** | The5%ers 1.5 %; eToro CSL 40 % | Not currently in wizard — propose Step 3 addition | Yes |
| **Mandatory SL on every order** | The5%ers; FTMO ToS | Not currently in wizard — propose Step 3 toggle (default on) | Yes |
| **Heartbeat / connection-loss auto-flatten** | FIX session heartbeat + TestRequest; `ctrader_api_full_reference.md` §1.4 | cTrader uses 10-s heartbeat; runtime owns reconnect logic | Yes (runtime, not wizard) |
| **Cooldown after consecutive losses** | "Two-strike system" (Trading Reset); Tradeify | Not currently modelled | Optional (Risky Mode) |
| **News-event blackout** (e.g. NFP, FOMC, CPI) | FTMO 2 min ± high-impact; E8 5 min ±; FunderPro 2 min ± | Step 8 wires news provider; runtime enforcement missing | Yes |
| **Position-correlation cap** | BabyPips "double up" warning for EURUSD+GBPUSD ≈+0.93 | Not currently modelled | Optional (recommended) |
| **Volatility-spike auto-pause** | ATR-based volatility stop indicators; "Dynamic risk management" (Volatility Box) | Not currently modelled | Optional |
| **Maintenance window** (weekend, Asian session, broker maintenance) | EarnForex AutoTrading Scheduler; "close all on Friday 14:55 EST" | Not currently modelled | Yes (forex requires this) |
| **One-click global panic button** | MT4/5 AutoTrading toggle (red) | Not currently in app chrome; propose | Yes |
| **Server-side liquidation** | tastytrade PM clause; FTMO platform | Owned by broker — out of our scope | N/A |

### 7.1 Heartbeat — connection loss

The FIX-session contract that Spotware mimics (heartbeats every
N seconds; TestRequest on miss; force-disconnect on no-response)
is documented industry-wide:

> "Heartbeat and Test Request messages are used to maintain the
> integrity of connection … TestRequests will be sent in response
> to missed heartbeats, and may be sent periodically, and they
> must be responded to immediately with a Heartbeat containing
> the TestReqID or the session will be considered unresponsive
> and the connection terminated." (FIX Trading Community —
> *Session Layer Online*, WebSearch excerpt 2026-05-15.)

For cTrader specifically, `ctrader_api_full_reference.md` §1.4
already documents the 10-s heartbeat cadence. The wizard's role
is small: Step 4.4's "Account auth probe" should record the
heartbeat-tracker liveness as a final acceptance criterion.

### 7.2 News blackout — concrete window

Three independent prop-firm citations converge on **2–5 minute
windows** either side of high-impact events:

- FTMO Standard: 2 min before & after (§4.1).
- E8 Markets funded: 5 min before & after (§4.3).
- FunderPro: 2 min before & after (§4.3).

A reasonable default: **2 min before / 2 min after** for the
forex-ai news filter. The Step 8 wizard surface should expose
the window length as a numeric input with these defaults.

### 7.3 Maintenance window — concrete cadence

Forex's structural cadence is published by every broker:

> "Large institutional investors and banks typically do not
> operate over the weekend, so there is significantly less volume
> from Friday 5 pm EST through to Sunday at 7 pm EST." (Daytrading
> Forex Weekend Trading, WebSearch excerpt 2026-05-15.)

> "EAs can be requested with 'day of week' options to close all
> positions before weekend starts (for example, close all on
> Friday at 14:55 EST)." (Forex Factory thread, WebSearch excerpt
> 2026-05-15.)

Default proposed: auto-flatten and pause trading **Friday 16:00
ET → Sunday 18:00 ET**, and pause (don't flatten) during the low-
liquidity Tokyo open transition 17:00–18:00 ET each day.

### 7.4 Correlation cap

BabyPips frames the universal warning:

> "Opening a position in both EUR/USD and GBP/USD is the same as
> doubling up on a position, because both pairs would move in the
> same direction anyway. When implementing confirmation
> strategies with correlated pairs, consider risking 0.5% on each
> pair instead of 1% to maintain appropriate total exposure."
> (BabyPips — *Always Know Your Risk Exposure*, WebSearch excerpt
> 2026-05-15.)

> "GBPUSD and EURUSD show a strong positive correlation (around
> +93), meaning both pairs are highly correlated and generally
> move in the same direction." (Dukascopy reference, WebSearch
> excerpt 2026-05-15.)

**Concrete primitive**: a correlation matrix is recomputed on the
trailing 30-day window; if two open positions are above 0.7
correlation in the same direction, the second is rejected (unless
the user is in Risky Mode with an explicit override).

### 7.5 Volatility-spike auto-pause

The pattern is open-ended; LuxAlgo and Volatility Box recap the
idea:

> "Traders can adjust multipliers to match market conditions — for
> instance, during periods of high volatility, increasing the
> multiplier can provide a wider buffer, while reducing it during
> calmer markets can tighten the stops." (LuxAlgo blog,
> WebSearch excerpt 2026-05-15.)

For forex-ai, an N-sigma rule on rolling ATR is the operator-
canonical implementation. The wizard surfaces only the
on/off toggle and the σ threshold (default: 3.0 σ over a 14-bar
ATR window).

---

## §8 — UX patterns specific to onboarding

### 8.1 Risk-quiz gating

tastytrade and Robinhood both gate **derivatives** behind a
questionnaire. Robinhood is the cleaner shape:

> "You'll need to complete the investor profile questionnaire
> providing experience level, annual income, net worth, employment
> status, and risk tolerance." (See §5.4.)

For forex-ai, the equivalent gate is: **autonomous mode unlock
(§10)** requires the user to complete a "Forex risk acknowledgement
quiz" of 5 multiple-choice questions whose answers are clipped
to the constants we already enforce (e.g. "What is the max daily
loss for an FTMO Standard challenge?" → 5 %). Records hash +
timestamp.

### 8.2 Capital-disclosure step

eToro asks at the COPY step ($200 minimum); FTMO asks at the
**Challenge purchase** step (account size selection). Coinbase
asks **never** (KYC handles it). The pattern: **disclose the
*irrecoverable* capital before unlocking real-money execution.**

Proposal: between Step 9 (Auto-start) and Step 10 (Summary), a
new Step 9.5 — "Stage roadmap & capital disclosure" — that asks:

- "What is the maximum capital you can afford to lose on this
  account, in your account currency?"
- Optional. If filled, used as the soft per-account ceiling for
  autonomous mode's equity stop (not displayed as a hard limit
  but as a UI warning when paper-trade simulations imply this
  will be exceeded).

### 8.3 Risk-profile sliders

Industry convention is 1–10 (eToro, Wealthfront):

> "Many trading platforms use a 1–10 risk slider: 1–3 is
> conservative, 4–7 is balanced/moderate, and 8–10 is aggressive."
> (Schwab Learn / Stifel risk-tolerance classifications,
> WebSearch excerpt 2026-05-15.)

Wealthfront's own questionnaire maps to a 0.5–10 score:

> "When you sign up for an Automated Investing Account, you take
> a risk questionnaire that helps assign you a risk score from
> 0.5 (least risky) to 10 (highest risk)." (Wealthfront *Risk
> Questionnaire*, WebSearch excerpt 2026-05-15.)

**For forex-ai**, the slider maps to *strategy hyperparameters*,
not to portfolio allocation (we are not a robo-advisor). A
proposed mapping:

| Slider | Per-trade max risk | Max concurrent positions | News-blackout window | Volatility σ pause | Correlation cap |
|---|---|---|---|---|---|
| 1 (conservative) | 0.25 % | 1 | 5 min ± | 2.0 σ | 0.5 |
| 4 (moderate) | 0.75 % | 3 | 2 min ± | 3.0 σ | 0.7 |
| 7 (aggressive) | 1.5 % | 6 | 1 min ± | 4.0 σ | 0.8 |
| 10 (Risky Mode) | 2.5 % | 10 | none | 5.0 σ | 0.95 |

Slider position 10 unlocks "Risky Mode" — a **separate research
agent** owns its hyperparameters (operator note: 4 % monthly
target is a **floor for production**, Risky Mode can target
higher).

### 8.4 Strategy template gallery

QuantConnect's Strategy Library and Option Alpha's templates are
the industry references:

> "QuantConnect has added a dozen or so Alphas with plans to keep
> adding more so you'll have an enormous pool of template code to
> work from to seed your ideas." (QuantConnect blog, WebSearch
> excerpt 2026-05-15.)

> "Option Alpha offers pre-built bot templates shared in the
> Community that you can clone and edit or modify to fit your
> personal trading strategy, with new templates added weekly."
> (Option Alpha Templates, WebSearch excerpt 2026-05-15.)

3Commas ships **named presets** (DCA, Grid, Scalping bots) that
map roughly to forex equivalents:

> "3Commas' DCA Bot features over 11+ built-in indicators (like
> RSI, BB, EMA) for scalping using short 3-5 min timeframes."
> (3Commas DCA Bot page, WebSearch excerpt 2026-05-15.)

**For forex-ai**, the proposed initial template gallery (Step 5
extension) is six entries:

1. **Scalping EURUSD M1** — operator-named example. Risk slider
   pre-set at 4. Symbols: `EURUSD`. Timeframes: `M1, M3, M5`.
2. **Scalping majors M5** — Symbols: top 7 majors. Timeframes:
   `M5, M15`. Risk slider 3.
3. **Swing trading D1 majors** — Symbols: top 7 majors.
   Timeframes: `H4, D1, W1`. Risk slider 4.
4. **Trend following H1 baskets** — Symbols: top 28 majors.
   Timeframes: `H1, H4`. Risk slider 5.
5. **Mean-reversion H1 majors** — Symbols: 7 majors. Timeframes:
   `M30, H1, H4`. Risk slider 4.
6. **Custom (start from blank)** — fall-through to the existing
   Step 5 symbol/timeframe picker. No defaults.

### 8.5 "Quick-start" presets vs full customisation

QuantConnect's Builder/IDE duality is the model (§2.3): one
toggle, two product surfaces. Our wizard's equivalent: **"Use a
template" (default) vs "Build from scratch"** at the top of Step
5. Template path skips the symbol/timeframe picker for the
template defaults; Custom path is the existing flow.

---

## §9 — Mapping the patterns to forex-ai

This section walks every pattern in §1–§8 against the current
10-step wizard and proposes where each lands.

### 9.1 Pattern-to-step matrix

| Pattern (citation) | Current step | Proposed change |
|---|---|---|
| Live/Paper colour-coded login (§1.1) | Step 4 (cTrader OAuth) | Wrap the modal in mode-specific accent colour (gray for Demo, red for Live). Token resolves the mode after Step 4.3 account-pick — repaint Step 4.4 background red if Live. **UX patch.** |
| Typed agreement signature for Live (§1.1) | Step 10 (Apply) | If `trading_mode = live` in Step 3, **Apply** is gated behind a typed-signature modal: user types the broker-account number to confirm. **UX patch.** |
| MT5 Demo / Existing / Real branching (§1.2) | Step 3 | Already a three-way radio (Backtest/Forward/Live). **No change.** |
| Default to Simulation (§1.3) | Step 3 default | Already `forward` (Step 3.4). **No change.** |
| paperMoney 1-click toggle (§1.4) | Main app chrome (not wizard) | New Settings panel item: persistent toggle "Trading mode: Demo / Forward / Live small / Live full" with current-stage indicator. Wizard's Step 10 writes the initial state. **New main-app UI.** |
| First-week paperMoney copy (§1.4) | Step 10 Apply | Add to the post-Apply tour: "spend ≥ N days here in Forward test before unlocking Live small". **Copy.** |
| Per-feature suitability (§1.5) | Step 9.5 (new) | Risk acknowledgement quiz to unlock each higher stage. **New step.** |
| Two-pane list+params (§2.1) | Step 5 (new) | Template gallery as left pane, parameters as right pane. **Layout change.** |
| Save/Load/Reset preset (§2.2) | All steps with parameters | Add a per-step `[Save preset]` / `[Load preset]` / `[Reset to defaults]` footer. **UX patch.** |
| Global panic button (§2.2) | Main app chrome | Red "Halt all automation" button in the always-visible status bar. **New main-app UI.** |
| Builder vs IDE duality (§2.3) | Step 10 Apply | After Apply, open the Builder tour by default; "Exit Builder Mode" reveals `config.yaml` editor. **New main-app UI.** |
| Save Load Reset on strategy (§2.4) | Step 5 (new template gallery) | Same as above. |
| Separate keys per env (§2.5) | Step 4.3 | Already exposed per-account. **No change.** |
| Active-account banner (§2.5) | Main app | "Active account: Demo #12345" banner in main app top bar. **New main-app UI.** |
| Tiered KYC unlock (§3.1) | Stage 1–4 progression | Maps to §6.2 staging gates. **New mechanism.** |
| Beginner / Advanced toggle (§3.2) | Step 3 + main app | Add to Step 3 a "Interface mode" radio (Beginner / Advanced). Default Beginner. Beginner hides Polars debug panes, raw OAuth tokens, raw protobuf inspector. **New step input + main-app UI gate.** |
| Verification tiers (§3.3) | Stage 1–4 progression | Same as §3.1. |
| FTMO daily reset cadence (§4.1) | Step 3 | Add a "Daily-loss reset timezone" dropdown (default CE(S)T for FTMO). **New step input.** |
| Per-firm ToS surface (§4.3) | Step 3 prop-firm preset | Each preset now carries a full struct: max-daily-loss, max-overall-drawdown, news-window, HFT-rule, EA-allowed, cTrader-allowed. **New struct fields on `PropFirmConstraints`.** |
| FundedNext cTrader-ban warning (§4.3) | Step 3 | Inline warning when user picks FundedNext but cTrader account selected in Step 4. **New cross-step validator.** |
| Soft vs hard kill switch vocab (§4.4) | Step 3 | Label `max_daily_loss_pct` as "Daily-loss limit (soft pause)" and `max_overall_drawdown_pct` as "Overall drawdown (account closed)". **Copy.** |
| Copy Stop-Loss 40 % default (§5.1) | Step 9.5 (new) | "Equity stop %" slider for autonomous mode (default 20 %, range 5–95 %). **New step input.** |
| Strategy risk score (§5.1) | Step 5 (templates) | Each template carries a 1–10 risk score; surfaces in the gallery. **Template metadata.** |
| Per-order safety check (§5.2) | Runtime + Step 3 | Step 3 adds "Confirm orders above N USD notional" numeric. Runtime enforces. **New step input.** |
| Confirmation default-on (§5.3) | Same | Default 0 = always confirm; user can raise threshold. |
| Risk quiz (§5.4 / §8.1) | Step 9.5 (new) | 5-question multiple-choice. **New step.** |
| Demo→Paper→Live→Full (§6) | Step 9.5 (new) | "Stage roadmap" panel showing the four-stage progression + criteria. **New step.** |
| Per-trade max loss (§7) | Step 3 | "Per-trade max risk %" slider 0.1–2.0 % default 1.0 %. **New step input.** |
| Mandatory SL on every order (§7) | Step 3 | Toggle "Require Stop Loss on every order" default ON (when prop-firm preset is FTMO). **New step input.** |
| News blackout window (§7.2) | Step 8 | Numeric "Blackout window (minutes ± high-impact event)" default 2. **New step input.** |
| Maintenance window (§7.3) | Step 8 (extends) | Toggle "Auto-flatten Friday 16:00 ET / pause Sunday 16:00–18:00 ET" default ON. **New step input.** |
| Correlation cap (§7.4) | Step 8 (extends) | "Max correlation between concurrent open positions" slider 0.5–0.95 default 0.7. **New step input.** |
| Volatility σ pause (§7.5) | Step 8 (extends) | "Auto-pause when ATR > Nσ" numeric default 3.0 σ. **New step input.** |
| Risk-profile slider (§8.3) | Step 3 | "Risk profile" 1–10 slider (default 4). Drives the table in §8.3 — pre-fills all the inputs above. Power users can override. **New step input.** |
| Strategy template gallery (§8.4) | Step 5 (extends) | Six templates listed; "Custom" falls back to existing UI. **New gallery.** |
| Template / Custom split (§8.5) | Step 5 top | New radio at the top: "Use a template (recommended) / Build from scratch". |
| Stage roadmap (§6) | Step 9.5 (new) | One scrollable card listing all four stages with current-stage badge. |
| Capital disclosure (§8.2) | Step 9.5 (new) | "Capital you can afford to lose" optional numeric. |

### 9.2 New wizard step — Step 9.5

The proposed insertion sits between Step 9 (Auto-start) and
Step 10 (Summary) because it consolidates **everything the user
must acknowledge before any real-money action**, after all
mechanical setup is complete. Verbatim spec:

```
Step 9.5 — Autonomy & risk acknowledgement
- Purpose: gate autonomous mode + risk-quiz + capital disclosure
  + stage roadmap.
- Mockup: four collapsed cards.
  Card 1 "Stage roadmap" — the four-stage table (§6.1); current
  stage badge says "Stage 2 — Paper" because Step 3.4 trading_mode
  defaults to forward.
  Card 2 "Risk acknowledgement quiz" — five MCQ
  questions. Cannot Continue until 5/5 correct.
  Card 3 "Per-trade & per-day caps" — pre-filled by the Step 3
  risk-profile slider (§8.3). Editable.
  Card 4 "Autonomous mode" — collapsed by default. Expanding it
  reveals:
    - Toggle "Enable autonomous mode" (default off)
    - When on, shows: equity-stop slider, per-order confirmation
      threshold, news-blackout window, maintenance-window toggle,
      correlation cap, volatility σ pause, capital-loss
      disclosure field.
- Inputs: risk_quiz_score, risk_quiz_hash, autonomous_mode_enabled
  (bool), and the autonomous-mode struct (§10 below).
- Actions: writes `<data_path>/risk_acknowledgement.json` with
  quiz answers + timestamp + version + SHA-256.
- Skip: NOT ALLOWED if Step 3 trading_mode = live OR autonomous
  mode is enabled. Allowed otherwise.
- Back: Step 9.
- Time: 3–5 min (the quiz is the dominant cost).
```

### 9.3 Backend implications

| Change | Crate | New / Refinement |
|---|---|---|
| `PropFirmConstraints` struct extended with news/HFT/EA fields | `forex-core/src/domain/prop_firm.rs` | Refinement (new fields, default-on for FTMO) |
| Risk-profile slider → hyperparameter mapping | `forex-core/src/domain/risk_profile.rs` (new) | New module |
| Strategy template registry | `forex-strategies/src/templates/` (new) | New module |
| Stage roadmap state machine | `forex-app/src/domain/staging.rs` (new) | New module |
| Risk-quiz hashing + persistence | `forex-app/src/app_services/risk_ack.rs` (new) | New module |
| Autonomous-mode controller | `forex-app/src/autonomy/` (new) | **New top-level module** — see §10 |
| Wizard step 9.5 UI | `forex-app/src/ui/wizard/step_9_5.rs` (new) | New |
| Settings panel "Mode toggle" + "Panic button" | `forex-app/src/ui/main/status_bar.rs` (extends) | Refinement |

---

## §10 — "Do everything automatically" architecture

This is the operator-named "live trading με ένα κουμπί". The
spec below defines what that button does, the staging gates that
surround it, and the kill-switch hierarchy.

### 10.1 The button

In the main app, a single button labelled **"Autonomous Mode:
OFF — start"** appears in the right-hand status bar. Clicking it
opens a confirmation modal with the verbatim text:

> "Autonomous Mode will continuously discover, train, paper-trade,
> and (when criteria are met) live-trade strategies on your
> behalf. You can stop it at any time with the red **HALT**
> button. By enabling, you confirm you have completed the risk
> acknowledgement (Step 9.5) and understand the kill-switch
> hierarchy below."

The modal lists the kill-switch hierarchy verbatim (§10.4) and
shows two buttons: `[Cancel]` and `[Enable Autonomous Mode]`.

### 10.2 What autonomous mode does

| Phase | Action | Citation pattern |
|---|---|---|
| **Discovery** | Sweeps strategy templates against the historical cache (Step 6) on the chosen symbols/timeframes (Step 5). | QuantConnect template gallery (§2.3) |
| **Training** | For each promising template, trains hyperparameters via the existing `training_orchestrator` (`forex-models/src/training_orchestrator.rs`). | (internal) |
| **Paper trade** | Promotes the best-K strategies to Stage 2 — forward test on live cTrader streaming data with simulated execution. | TradingView Paper Trading (§1.1); ThinkOrSwim paperMoney (§1.4); Alpaca paper-api (§2.5) |
| **Live small** | After paper-stage gating criteria met (§6.2), promotes one strategy to Stage 3 — real cTrader account, capped notional (eToro $200 floor mapped to user-provided "capital you can afford to lose" from §8.2). | The5%ers Instant Funding $20k (§4.2); eToro CopyTrader (§5.1) |
| **Live full** | After 30 days at Stage 3 with criteria met, promotes to Stage 4 — uncapped. | FTMO Trading Objectives (§4.1) |

### 10.3 Per-stage promotion gates

Recapitulating §6.2 with the autonomous-mode lens. Each gate is
a **boolean function** in `forex-app/src/autonomy/gates.rs` (new
module). Gate vocabulary uses the FTMO / Tradeify vocabulary
verbatim where possible.

```
fn paper_to_live_small(stage_state: &StageState) -> Decision {
  if stage_state.days_in_paper < 10 { return Hold("Need ≥10 trading days"); }
  if stage_state.paper_monthly_net_profit_pct < 0.04 {
     return Hold("Need ≥4 % monthly net profit (operator policy)");
  }
  if stage_state.paper_daily_loss_breach_count > 0 {
     return Reject("Daily-loss limit breached during paper phase");
  }
  if !stage_state.risk_ack_hash_present { return Reject("Risk acknowledgement missing"); }
  if !stage_state.live_signature_present { return Hold("Typed signature required (TradingView pattern §1.1)"); }
  Promote
}
```

The operator-locked 4 % monthly **floor** at `prop_firm.rs:36` is
exactly the gate's threshold. *4 % is not the ceiling* — strategies
that exceed 4 % advance faster, but they still need the 10-day
minimum and the breach-count of zero.

### 10.4 Kill-switch hierarchy

Four tiers; each tier can interrupt all lower tiers.

| Tier | Trigger | Action | Citation |
|---|---|---|---|
| **T1 — Per-trade** | Order would exceed per-trade max risk % from Step 3 (§7 must-have) | Reject order at router before submit | The5%ers 1.5 % rule (§4.2); IBKR Precautionary Settings (§5.2) |
| **T2 — Per-day soft pause** | Realised + unrealised P&L hits daily-loss limit | Flatten all positions; disable new orders until next 00:00 in the reset timezone | FTMO 5 % daily loss (§4.1); Tradeify DLL (§4.4) |
| **T3 — Per-week / per-month cooldown** | Two daily-soft-pause hits in one week, OR monthly net < 4 % floor for 30 d | Stop autonomous mode; require user re-arming | Trading Reset two-strike rule (§7) |
| **T4 — Manual (HALT)** | User clicks the red HALT button in the status bar | Flatten everything, disable autonomous mode | MT4/5 AutoTrading panic button (§2.2) |
| **T0 — Server-side stop-out** | Broker margin level < N % | Broker closes positions; we observe via `ExecutionEvent` (2126) | FXOpen Margin Call & Stop Out (WebSearch excerpt 2026-05-15) |

T0 happens regardless of forex-ai's state; we just observe it
and surface the event verbatim.

### 10.5 Interaction with `PropFirmConstraints::FTMO_STANDARD`

The constants are operator-locked; autonomous mode is a *consumer*
of the constants, not a re-definer. Specifically:

- `max_daily_loss_pct=0.05` → T2 trigger.
- `max_overall_drawdown_pct=0.10` → T3-like hard stop (in addition
  to T2; if T3 fails to fire and equity nears -10 %, autonomous
  mode self-disables and triggers HALT).
- `challenge_profit_target_pct=0.10` → reporting only (per-challenge
  goal, not per-month).
- `min_monthly_net_profit_pct=0.04` → §10.3 promotion gate (also
  the per-month floor for staying at Stage 4 — if the user holds
  Stage 4 for 30 days with sub-4 % return, autonomous mode demotes
  to Stage 3 for re-evaluation).
- `min_trading_days=10` → §10.3 paper-to-live-small gate.

### 10.6 UI for monitoring autonomous mode

Refer to `docs/audits/research/ui_ux_design_spec.md` §1.8 (panel
layout — TradingView desktop) for the panel grammar. The proposed
"Autonomy" panel attaches to the right docking strip:

```
┌─ Autonomy ───────────────────────────────────────┐
│  Status:    ACTIVE                  [HALT (T4)]  │
│  Stage:     Stage 2 — Paper (4 of 10 days)       │
│  Best strategy: scalping_eurusd_m5 (score 7.2)   │
│                                                  │
│  Today                                           │
│    Trades:   12   Wins: 7   Losses: 5            │
│    P&L:      +0.42 %                             │
│    Daily-loss limit:   -5.00 % (T2 idle)         │
│    Overall drawdown:    -0.18 % (T3 idle)        │
│                                                  │
│  Stage-promotion gates                           │
│    [✓] Days in paper ≥ 10 (4/10)                 │
│    [ ] Monthly net P&L ≥ 4 % (-0.6 %)            │
│    [ ] Daily-loss breach count = 0 (0)           │
│    [✓] Risk acknowledgement on file              │
│                                                  │
│  Last 24h events                                 │
│    14:32  Cooldown — high-impact CPI window       │
│    11:08  T1 reject — would exceed per-trade 1 % │
│    08:15  Stage 1 → Stage 2 promotion             │
└──────────────────────────────────────────────────┘
```

Token references for the design (refer to
`ui_ux_design_spec.md`):

- Surface: `color.surface.card` (`#171A21` dark / `#F6F8FB` light)
- Text: `color.text.primary`
- Active-status pill: `color.success` background, `color.text.inverse`
- HALT button: `color.danger` solid; the only `color.danger`-
  solid element in the entire window — TradingView convention
  per `ui_ux_design_spec.md` §1.2.
- Gate ticks: `color.success` for ✓, `color.text.muted` for ☐.

### 10.7 What autonomous mode is NOT

- **Not a synthetic-data generator.** Stages 1–2 use real
  historical / live data; Stage 3–4 use real money. No fabricated
  fills or fabricated bars.
- **Not a guarantee.** The 4 % monthly floor is a *target*; the
  mode demotes Stage 4 → Stage 3 if it underperforms.
- **Not a substitute for the typed signature.** The signature
  (§1.1) is collected in the modal at first enable; subsequent
  re-enables within 30 days don't re-prompt.
- **Not unilateral.** Anything in the kill-switch hierarchy
  trumps it. The user clicking HALT is final.

---

## §11 — Open questions

Honest list — things the public docs don't pin and the operator
will need to resolve.

### 11.1 Exact circuit-breaker hysteresis

FTMO docs say "the limit is recalculated at 00:00 CE(S)T" but do
not publish the hysteresis when an account *touches* the limit
intra-day and then recovers — does the soft pause linger until
midnight, or unlock as soon as equity recovers? The verbiage at
FTMO Academy (§4.1) implies linger-until-midnight; TakeProfitTrader
(§4.4) confirms hard close on any touch. **Operator decision:**
adopt linger-until-midnight as the safer default.

### 11.2 Risky Mode boundary

The operator's 4 % monthly target is a floor for Production; Risky
Mode (separate research agent) can target higher. The boundary
between the two modes — at what slider position (§8.3) does Risky
Mode kick in? — is not yet operator-decided. Proposal: slider
positions 1–7 are Production; 8–9 are "Aggressive Production";
10 is Risky Mode (and unlocks a separate sub-wizard owned by the
Risky Mode agent).

### 11.3 News-event data source canon

`forex-ai`'s Step 8 surfaces a news provider but doesn't choose
one. FTMO, E8 and FunderPro converge on **high-impact red-folder
events**, but the canon source (ForexFactory, Investing.com,
DailyFX) is operator preference. Default proposed:
ForexFactory's RSS feed, since it's free and red-folder events
are tagged.

### 11.4 cTrader-vs-MT5 platform choice for prop firms

FundedNext bans automation on cTrader (§4.3). FTMO permits cTrader.
Other firms vary. The wizard's prop-firm preset must encode
`cTrader_allowed: bool` per firm — we have public sources only
for FTMO, FundedNext, E8 today; the other firms (MyForexFunds,
FundedTrader) need a manual pass.

### 11.5 QuantRocket onboarding shape

WebFetch on the QuantRocket docs failed in this sandbox, and
WebSearch returned only one unrelated thread. The QuantRocket
JupyterLab-style flow is omitted from §2; this is an information
gap, not a deliberate choice. If the operator wants it surfaced,
a follow-up research agent can WebFetch from a different IP.

### 11.6 cTrader "Copy" interaction

cTrader Copy ships a stand-alone equity-stop primitive (§5.1
mapping) at the broker side, independent of cBots. Whether
forex-ai users can opt-in to broker-side copy on top of
autonomous mode is an open question (likely yes, but it requires
a separate `ProtoOA*` exchange in the cTrader OpenAPI that we
have not yet vendored).

### 11.7 macOS notarization for autonomous mode

`installer_wizard_ux_spec.md` §10.2 already flags the macOS
notarization concern. Autonomous mode binds the same loopback
ports + opens the system browser — the existing entitlement
profile should suffice, but it's worth a re-notarize after the
new module lands.

### 11.8 Risk-quiz content

The five MCQ questions for Step 9.5's risk acknowledgement quiz
are placeholders here. The exact content is a compliance
question — should the wizard authors run it past the operator's
counsel before shipping? Recommendation: yes, but ship a v0 set
that covers (i) what is the daily-loss limit (5 %), (ii) what
happens on overall drawdown breach (account closed), (iii) what
is the news-blackout window (2 min ±), (iv) what is the per-
trade max risk (slider-dependent), (v) what does the HALT button
do (flatten + disable).

### 11.9 Stage 4 → Stage 3 demotion ToS conflict

Demoting Stage 4 → Stage 3 on a sub-4 % month implies capping
the user's notional unilaterally. If the user is on an FTMO
funded account, capping notional is **our** decision, not the
broker's — but FTMO has no rule that demotes a passing account.
Operator decision: should autonomous mode demote when the user
is on a funded account? Proposal: no — once at Stage 4 on a
funded account, demotion only happens on explicit user action.
On a personal account, demotion fires.

### 11.10 Strategy-uniqueness rule across prop-firm accounts

E8 (§4.3) bans identical strategies across accounts. Autonomous
mode, if attached to two accounts on the same prop firm, could
breach this. Proposal: in Step 4.3 (account picker), when two or
more accounts on the *same broker_id* are selected, surface a
warning and refuse to enable autonomous mode on more than one
of them by default.

---

## §12 — Methodology notes

- Operator-policy values read directly from working copy at
  `/home/user/forex-ai/`.
- External UX guidance attributed inline; "WebSearch excerpt"
  indicates WebFetch 403 with WebSearch excerpt used as fallback
  (same procedure as
  `docs/audits/research/installer_wizard_ux_spec.md` §0).
- cTrader payload IDs / names taken from
  `docs/audits/research/ctrader_api_full_reference.md`, itself
  built from vendored `.proto` at
  `crates/forex-app/proto/`.
- The 11-canonical-timeframe rule (no H2) and 4 % monthly
  **floor** for Production are operator directives recorded at
  `crates/forex-core/src/contracts/temporal.rs:17–24` and
  `crates/forex-core/src/domain/prop_firm.rs:36` (both
  2026-05-14).
- No code changed — research only.

---

## §13 — Glossary of new identifiers (proposed)

For implementers — these are the identifiers this audit proposes.
None exist in the codebase today; each is a hook for a follow-up
agent.

| Identifier | Proposed location | Purpose |
|---|---|---|
| `Stage` enum (`Demo`, `Paper`, `LiveSmall`, `LiveFull`) | `forex-app/src/domain/staging.rs` | §6 staging |
| `StageState` | `forex-app/src/domain/staging.rs` | gate evaluation |
| `RiskProfile` (1–10) | `forex-core/src/domain/risk_profile.rs` | §8.3 slider |
| `RiskProfileMapping` | same | slider → hyperparameter map |
| `RiskAcknowledgement` (quiz answers + hash) | `forex-app/src/app_services/risk_ack.rs` | §5.4, §8.1 |
| `StrategyTemplate` registry | `forex-strategies/src/templates/` | §8.4 |
| `AutonomousMode` controller | `forex-app/src/autonomy/mod.rs` | §10 |
| `KillSwitchTier` enum | `forex-app/src/autonomy/kill_switch.rs` | §10.4 |
| `PromotionGate` trait | `forex-app/src/autonomy/gates.rs` | §10.3 |
| `NewsBlackoutWindow` | `forex-core/src/domain/news_filter.rs` (extend) | §7.2 |
| `MaintenanceWindow` | `forex-core/src/domain/maintenance.rs` (new) | §7.3 |
| `CorrelationCap` | `forex-core/src/domain/risk.rs` (new) | §7.4 |
| `VolatilitySigmaPause` | `forex-core/src/domain/risk.rs` (new) | §7.5 |
| `WizardStep9_5State` | `forex-app/src/ui/wizard/step_9_5.rs` | §9.2 new step |
| `PropFirmConstraints` (extended fields) | `forex-core/src/domain/prop_firm.rs` (extend) | §9.3 |

Cross-reference to existing identifiers documented in
`installer_wizard_ux_spec.md` §12 — no rename of those.

---

## §14 — Acceptance criteria for the *competitive analysis*

(Distinct from the wizard's own acceptance criteria at
`installer_wizard_ux_spec.md` §11.) This document is acceptable if:

1. Every external claim carries an inline URL citation with a
   "WebSearch excerpt" marker or explicit WebFetch confirmation.
2. Every pattern lifted from §1–§8 lands in §9 against a
   specific wizard step (existing or new).
3. The proposed Step 9.5 has a complete spec (purpose, inputs,
   actions, skip/back/cancel, time) following the
   `installer_wizard_ux_spec.md` template.
4. Autonomous mode (§10) is defined as a state machine with
   explicit promotion gates, an explicit kill-switch hierarchy,
   and an explicit interaction with `FTMO_STANDARD`.
5. All operator invariants are preserved: 11 canonical
   timeframes (no H2 ever surfaced in any proposed UI), 4 %
   monthly floor (never reduced; Risky Mode is a separate
   branch), no synthetic data (Stage 1 uses historical replay,
   not fabricated bars), `PropFirmConstraints::FTMO_STANDARD` is
   the only hardcoded prop-firm preset until additional firms
   are operator-approved.
6. Open questions (§11) name the missing-information cases
   honestly.
7. No code is changed; all proposed identifiers are flagged as
   new and unimplemented in §13.

---

— END —

(External citations are enumerated in §0.1; internal references
in §0.2; new identifiers in §13. No separate "sources cited"
appendix is repeated here.)

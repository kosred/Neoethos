# dxtrade.rs split — draft (pending verification)

## Τι έγινε

`crates/forex-app/src/app_services/dxtrade.rs` (2787 γραμμές) τεμαχίστηκε σε
**5 submodules** ως draft. Active source παραμένει το ορίτζιναλ
`dxtrade.rs` — το split βρίσκεται στο `dxtrade_split_draft/` ώστε ο
build να μη σπάσει μέχρι να επιβεβαιωθεί από τον operator.

```
crates/forex-app/src/app_services/
├── dxtrade.rs                   2787 γρ. — ACTIVE source (μένει αμετάβλητο)
└── dxtrade_split_draft/         DRAFT — πιο μικρά αρχεία
    ├── mod.rs           1369    doc + module decls + re-exports + bundle + test block (intact)
    ├── auth.rs           290    Phase D3.1
    ├── orders.rs         577    Phase D3.2
    ├── streaming.rs      433    Phase D3.3
    └── transport.rs      185    HTTP transport + shared helpers (current_unix_seconds, truncate_for_log)
                       ─────
                        2854    +67 γραμμές vs original (το overhead είναι imports + module declarations)
```

Κανένα impl αρχείο > 600 γραμμές. Όλα τα impls split, όλα τα tests intact στο `mod.rs`.

## Cross-module wiring που εφαρμόστηκε

- `transport.rs` εκθέτει `pub(super) fn current_unix_seconds()` + `pub(super) fn truncate_for_log()` ώστε όλα τα submodules να τα μοιράζονται.
- `orders.rs` εκθέτει `pub(super) fn validate_session_for_trading()` ώστε το `streaming.rs` να μη το ξαναγράφει.
- `mod.rs` κάνει `pub use auth::*` / `orders::*` / `streaming::*` / `transport::*` για τα symbols που χρησιμοποιούσε ο εξωτερικός κόσμος (TradingSession, wizard, Flutter API). Άρα ΔΕΝ χρειάζεται κανείς call site από εξωτερικά να αλλάξει imports.

## Γιατί draft (και όχι swap-in αμέσως)

Το Linux sandbox μου **δεν τρέχει cargo check** στο workspace
(τα Cargo.toml κάποιων crates στο mount έχουν stale truncated
content — γνωστό symptom όλη τη μέρα). Δεν μπορώ να επιβεβαιώσω
ότι:

- Τα tests στο `mod.rs` βρίσκουν όλα τα symbols (`format_price`, `url_path_escape`, `generate_order_code`, `parse_timeout_seconds`, κλπ.) — μερικά είναι ιδιωτικά στο submodule και το test block ίσως χρειάζεται extra re-exports σε scope.
- Δεν υπάρχει circular dep (μάλλον δεν υπάρχει: orders → auth → transport).
- Όλα τα `pub(super)` markers δουλεύουν με το test block.

## Πώς να ολοκληρώσεις το split (5-min job σε Windows)

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-app\src\app_services

# 1. Activate the split
Remove-Item dxtrade.rs
Rename-Item dxtrade_split_draft dxtrade

# 2. Compile
cd ..\..\..\..\
cargo check -p forex-app
```

**Αν cargo βρει errors**, τα πιο πιθανά είναι:
- `error[E0603]: function 'format_price' is private` — αυτό σημαίνει ότι ο test block στο `mod.rs` χρησιμοποιεί helpers που έμειναν `fn` στο `orders.rs`. Λύση: αλλάζω σε `pub(super) fn` εκείνες τις συναρτήσεις (`format_price`, `url_path_escape`, `generate_order_code`, `opposite_side`). Είναι κανονικά 4-5 σημεία.
- `error[E0432]: unresolved import` — κάποιο symbol δεν έχει re-exported στο `mod.rs` `pub use`. Πρόσθεσέ το στη λίστα του αντίστοιχου submodule.

**Αν το build περάσει και τα tests περάσουν** (`cargo test -p forex-app dxtrade`), commit ως:

```bash
git add crates/forex-app/src/app_services/dxtrade/
git rm crates/forex-app/src/app_services/dxtrade.rs
git commit -m "refactor · dxtrade.rs (2787 γρ.) → dxtrade/{transport,auth,orders,streaming,mod}.rs

Each submodule ≤ 600 lines. Implementations split; the giant test block
stays inline in mod.rs via pub-use re-exports. Cross-module helpers
(current_unix_seconds, truncate_for_log) moved into transport.rs;
validate_session_for_trading hoisted to pub(super) in orders.rs so
streaming.rs can reuse it. No external API change."
```

**Αν το build σπάσει** και θες rollback σε ένα βήμα:

```powershell
Rename-Item dxtrade dxtrade_split_draft
# original dxtrade.rs is still there, untouched
```

## Επόμενα στόχοι refactor (από τα 7 file-level `#![allow(dead_code)]`)

Αυτά είναι "gaps από εναλλαγές" που λες — files που έγραψα/έγραψες χθες/προχθές αλλά δεν έχουν consumer ακόμα:

| File | Lines | Status / Why dead_code |
|---|---|---|
| `ctrader_history.rs` | 1050 | Awaiting Flutter API wiring (REST `/history`) — D3.4-equivalent |
| `ctrader_messages.rs` | 901 | Protobuf envelope helpers; production path uses subset only |
| `ctrader_proto_messages.rs` | — | Wire-format builders for proto transport (feature-gated) |
| `ctrader_session.rs` | — | Session lifecycle for the protobuf transport (feature-gated) |
| `pnl.rs` | — | PnL aggregation API surface, waiting for chrome wire-up |
| `ui/theme.rs` | — | Theme tokens not yet referenced by every panel (egui) |
| `ui/wizard/migration.rs` | — | Wizard state migration scaffold |

Για κάθε ένα: ο `#![allow(dead_code)]` πρέπει να φύγει είτε wiring (αν είναι ώριμο), είτε διαγραφή/move (αν είναι κενός σκελετός). Αυτό απαιτεί cross-module read που χρειάζεται ζωντανή cargo check για να επιβεβαιώσω. Αφήνω αυτό για ξεχωριστή φάση.

## Disk status

- 102 GB free σταθερό
- Δεν τράβηξα τίποτα από δίσκο σε αυτό το round (μόνο 5 νέα .rs αρχεία ~150 KB total)

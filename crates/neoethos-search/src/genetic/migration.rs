//! Distributed island-model migration (Federation Phase 1).
//!
//! Each participating node runs its NORMAL `neoethos-search` GA on a combo as
//! one island — unchanged, and never-OOM (memory is capped to the node's own
//! hardware). Every [`INTERVAL`] generations the island publishes its top
//! [`ELITES`] genes here; the mesh sidecar gossips them to peers and pushes
//! peers' elites back in. Received migrants join the next generation and are
//! re-scored on THIS node's data before they can survive — the same
//! deterministic re-verification doctrine as everywhere else, so no cross-node
//! trust (and no shared RNG, no barrier, no determinism) is required.
//!
//! Research basis (POOL_SPEC.md): asynchronous migration, INTERVAL-dominant
//! (size minor), elitist send / replace-worst-if-better, panmictic via gossip.
//!
//! **OFF BY DEFAULT.** When [`migration_enabled`] is false (the default, and
//! the case for every non-distributed run) both hook points in the GA loop are
//! no-ops, so the search is byte-identical to a single-machine run. Only the
//! mesh sidecar turns it on, per process.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use super::strategy_gene::Gene;

/// Migrate every N generations. Interval is the dominant factor in the
/// literature; a mid value balances diversity inflow against churn.
pub const INTERVAL: usize = 15;
/// Elites sent per migration. Size is a minor factor — a few is plenty.
pub const ELITES: usize = 4;
/// Cap on buffered incoming migrants so a chatty swarm can't grow memory.
const INCOMING_CAP: usize = 64;

static ENABLED: AtomicBool = AtomicBool::new(false);
static OUTGOING: Mutex<Vec<Gene>> = Mutex::new(Vec::new());
static INCOMING: Mutex<Vec<Gene>> = Mutex::new(Vec::new());

/// Turn migration on/off for THIS process (the mesh sidecar calls this).
pub fn set_migration_enabled(on: bool) {
    ENABLED.store(on, Ordering::SeqCst);
}

/// Whether migration is active. False (default) ⇒ the GA hooks are no-ops.
pub fn migration_enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

/// GA → buffer: publish this island's current elites (replaces the last set).
pub fn publish_elites(elites: Vec<Gene>) {
    if let Ok(mut o) = OUTGOING.lock() {
        *o = elites;
    }
}

/// Mesh → wire: drain the elites to gossip to peers.
pub fn take_outgoing() -> Vec<Gene> {
    OUTGOING.lock().map(|mut o| std::mem::take(&mut *o)).unwrap_or_default()
}

/// Mesh → buffer: a peer's elites arrived; queue them for the GA (capped).
pub fn push_incoming(genes: Vec<Gene>) {
    if let Ok(mut i) = INCOMING.lock() {
        i.extend(genes);
        if i.len() > INCOMING_CAP {
            let overflow = i.len() - INCOMING_CAP;
            i.drain(0..overflow);
        }
    }
}

/// GA ← buffer: drain migrants to fold into the next generation.
pub fn take_incoming() -> Vec<Gene> {
    INCOMING.lock().map(|mut i| std::mem::take(&mut *i)).unwrap_or_default()
}

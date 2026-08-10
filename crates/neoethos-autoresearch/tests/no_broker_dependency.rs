//! Non-negotiable 5, enforced at the dependency edge rather than by review.
//!
//! `docs/autoresearch-loop.md` §16: *"the crate does not depend on
//! `neoethos-app` or `neoethos-trader`, so it is structurally incapable of
//! reaching a broker."*
//!
//! That sentence is only true while the manifest says so, and a manifest is one
//! line away from saying something else. This test reads the manifest and fails
//! if it ever does — because "the loop never places an order" has to be a
//! property of the build graph, not a promise in a doc comment.

use std::path::PathBuf;

/// Crates whose presence would give this one a path to the broker, the order
/// routes, or `live_portfolio.json`.
const FORBIDDEN: &[&str] = &["neoethos-app", "neoethos-trader"];

fn manifest() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("reading {}: {err}", path.display()))
}

#[test]
fn the_loop_cannot_reach_a_broker() {
    let text = manifest();
    // Only the dependency sections matter; the reasoning comment at the top
    // names both crates on purpose.
    let deps = text
        .split_once("[dependencies]")
        .map(|(_, rest)| rest)
        .unwrap_or(&text);
    for forbidden in FORBIDDEN {
        for line in deps.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            assert!(
                !line.starts_with(forbidden),
                "neoethos-autoresearch has taken a dependency on {forbidden}: {line:?}\n\n\
                 The autoresearch loop PROPOSES; the operator promotes. It must never place an \
                 order, never contact a broker and never write live_portfolio.json, and the way \
                 that is guaranteed is that the code to do so is not linked into it \
                 (docs/autoresearch-loop.md §16, non-negotiable 5). If this dependency is really \
                 needed, the design has changed and this test is the wrong thing to delete."
            );
        }
    }
}

#[test]
fn the_loop_writes_only_under_its_own_session_directory() {
    // A companion to the dependency check: the only paths this crate constructs
    // for writing are derived from `SessionStore::dir()`. Grepping the source is
    // crude, but it catches the one mistake that matters — a literal path into
    // the trading side's store.
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    visit(&src, &mut |path, text| {
        for (index, line) in text.lines().enumerate() {
            let lowered = line.to_ascii_lowercase();
            if lowered.contains("live_portfolio") && !lowered.trim_start().starts_with("//") {
                offenders.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "the autoresearch crate names live_portfolio outside a comment:\n{}",
        offenders.join("\n")
    );
}

fn visit(dir: &std::path::Path, f: &mut impl FnMut(&std::path::Path, &str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(text) = std::fs::read_to_string(&path) {
                f(&path, &text);
            }
        }
    }
}

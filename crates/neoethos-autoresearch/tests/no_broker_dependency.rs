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
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("reading {}: {err}", path.display()))
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

/// The file name this crate must never build a path to.
const TRADING_ARTIFACT: &str = "live_portfolio";

/// Constructs that turn a string into a filesystem path or an open handle.
///
/// A line that names the trading artifact AND does one of these is building a
/// path to it; a line that only names it is prose.
const PATH_CONSTRUCTS: &[&str] = &[
    ".join(",
    "PathBuf",
    "Path::",
    "File::",
    "fs::",
    "OpenOptions",
    "include_str!",
    "include_bytes!",
];

#[test]
fn the_loop_writes_only_under_its_own_session_directory() {
    // A companion to the dependency check: the only paths this crate constructs
    // for writing are derived from `SessionStore::dir()`. Grepping the source is
    // crude, but it catches the one mistake that matters — a literal path into
    // the trading side's store.
    //
    // WHAT THIS DOES *NOT* FLAG, and why it must not:
    //
    // Non-negotiable 5 requires the loop to SAY, in its own reports, that it
    // proposes and never writes the trading artifact — `judge.rs` prints it on
    // every promotion and `verdict.rs` prints it under every verdict. Those
    // sentences necessarily contain the artifact's name. A guard that failed on
    // the word would be a guard that pressured a future author to delete the
    // operator's warning in order to make the build green. So the guard tests
    // the PROPERTY (a path is being formed) rather than the SPELLING, via two
    // independent nets that both have to be evaded:
    //
    //   1. the name used as a path COMPONENT — preceded by a quote, a separator
    //      or a `format!` interpolation, which is how every real path literal
    //      begins (`"live_portfolio.json"`, `"{dir}/live_portfolio.json"`);
    //   2. the name on a line that also builds a path or opens a file.
    //
    // Prose says "... never writes live_portfolio.json." — preceded by a SPACE
    // and forming no path — and is caught by neither.
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    visit(&src, &mut |path, text| {
        for (index, line) in text.lines().enumerate() {
            let lowered = line.to_ascii_lowercase();
            if lowered.trim_start().starts_with("//") {
                continue;
            }
            let Some(at) = lowered.find(TRADING_ARTIFACT) else {
                continue;
            };

            // Net 1 — used as a path component.
            let preceded_by = lowered[..at].chars().next_back();
            let as_path_component =
                matches!(preceded_by, Some('"') | Some('/') | Some('\\') | Some('}'));

            // Net 2 — named on a line that forms a path or opens a file.
            let forms_a_path = PATH_CONSTRUCTS.iter().any(|c| line.contains(c));

            if as_path_component || forms_a_path {
                offenders.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "the autoresearch crate builds a path to the trading side's {TRADING_ARTIFACT} \
         artifact:\n{}\n\nThe loop PROPOSES; the operator promotes. It may name this file in a \
         report it prints, and it may never write one (docs/autoresearch-loop.md §16, \
         non-negotiable 5).",
        offenders.join("\n")
    );
}

/// The guard above must still fire on the mistake it exists for.
///
/// Without this, a future tightening that accidentally matched nothing would
/// leave a test that passes forever while checking nothing — which is worse than
/// no test, because it reads like coverage.
#[test]
fn the_path_guard_still_catches_a_real_path() {
    let offending = [
        r#"    let p = store.join("live_portfolio.json");"#,
        r#"    let p = format!("{dir}/live_portfolio.json");"#,
        r#"    let p = PathBuf::from("live_portfolio.json");"#,
        r#"    std::fs::write(live_portfolio_path(), bytes)?;"#,
    ];
    for line in offending {
        let lowered = line.to_ascii_lowercase();
        let at = lowered
            .find(TRADING_ARTIFACT)
            .expect("the fixture names it");
        let preceded_by = lowered[..at].chars().next_back();
        let as_path_component =
            matches!(preceded_by, Some('"') | Some('/') | Some('\\') | Some('}'));
        let forms_a_path = PATH_CONSTRUCTS.iter().any(|c| line.contains(c));
        assert!(
            as_path_component || forms_a_path,
            "the guard would have MISSED a real path: {line}"
        );
    }

    // And it must let the operator-facing prose through.
    let prose = "  \"PROMOTED - the loop does not write live_portfolio.json.\"";
    let lowered = prose.to_ascii_lowercase();
    let at = lowered.find(TRADING_ARTIFACT).unwrap();
    let preceded_by = lowered[..at].chars().next_back();
    assert!(
        !matches!(preceded_by, Some('"') | Some('/') | Some('\\') | Some('}'))
            && !PATH_CONSTRUCTS.iter().any(|c| prose.contains(c)),
        "the guard would flag the warning the operator is supposed to read"
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

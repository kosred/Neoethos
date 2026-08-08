//! Tier-1 test 6.1.4 (design doc): spawn the REAL binary, run
//! `initialize` → `tools/list` over stdio, and assert every stdout line
//! parses as JSON-RPC. Catches any stray print or misdirected log — the
//! classic Windows MCP corruption failure. No backend needed: the binary
//! only touches the backend inside tool calls.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn stdout_carries_json_rpc_exclusively() {
    let exe = env!("CARGO_BIN_EXE_neoethos-mcp");
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the neoethos-mcp binary");

    // Watchdog: if the server never answers or never exits, kill it so the
    // test fails with an assertion instead of hanging CI.
    let watchdog = {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let pid = child.id();
        let handle = std::thread::spawn(move || {
            if rx
                .recv_timeout(std::time::Duration::from_secs(120))
                .is_err()
            {
                // Best-effort kill by pid; the reader below then sees EOF.
                let _ = Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .output();
            }
        });
        (tx, handle)
    };

    {
        let stdin = child.stdin.as_mut().expect("stdin piped");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"stdout-hygiene-test","version":"0.0.0"}}}}}}"#
        )
        .expect("write initialize");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
        )
        .expect("write initialized notification");
        writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}"#)
            .expect("write tools/list");
    }
    // Close stdin: EOF ends the stdio transport and the server exits.
    drop(child.stdin.take());

    let stdout = child.stdout.take().expect("stdout piped");
    let mut saw_initialize = false;
    let mut saw_tools_list = false;
    let mut stdout_lines = 0usize;
    for line in BufReader::new(stdout).lines() {
        let line = line.expect("read stdout line");
        if line.trim().is_empty() {
            continue;
        }
        stdout_lines += 1;
        let value: serde_json::Value = serde_json::from_str(&line).unwrap_or_else(|e| {
            panic!(
                "NON-JSON on stdout (protocol corruption — a stray print or a log writing to \
                 the wrong stream): {e}\nline: {line:?}"
            )
        });
        assert_eq!(
            value.get("jsonrpc").and_then(serde_json::Value::as_str),
            Some("2.0"),
            "stdout line is JSON but not JSON-RPC: {line:?}"
        );
        if value.get("id") == Some(&serde_json::json!(1)) {
            saw_initialize = true;
            assert!(
                value.get("result").is_some(),
                "initialize must succeed, got: {line:?}"
            );
        }
        if value.get("id") == Some(&serde_json::json!(2)) {
            saw_tools_list = true;
            let tools = value["result"]["tools"]
                .as_array()
                .unwrap_or_else(|| panic!("tools/list must return a tools array, got: {line:?}"));
            assert_eq!(
                tools.len(),
                62,
                "tools/list over the wire must expose 62 tools"
            );
        }
    }
    assert!(saw_initialize, "no initialize response seen on stdout");
    assert!(saw_tools_list, "no tools/list response seen on stdout");
    assert!(stdout_lines >= 2, "expected at least the two responses");

    let status = child.wait().expect("wait for the binary to exit");
    // Disarm the watchdog only after the process is gone.
    drop(watchdog.0);
    watchdog.1.join().expect("watchdog thread joins");
    assert!(
        status.success(),
        "the binary must exit cleanly on stdin EOF, got: {status:?}"
    );
}

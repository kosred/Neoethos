use std::fs;

fn code_only(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    let mut block_depth = 0usize;
    while i < bytes.len() {
        if block_depth > 0 {
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                block_depth += 1;
                out.push_str("  ");
                i += 2;
            } else if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                block_depth -= 1;
                out.push_str("  ");
                i += 2;
            } else {
                out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            out.push_str("  ");
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            block_depth = 1;
            out.push_str("  ");
            i += 2;
            continue;
        }
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            out.push(' ');
            i += 1;
            while i < bytes.len() {
                let c = bytes[i];
                out.push(if c == b'\n' { '\n' } else { ' ' });
                i += 1;
                if c == b'\\' && i < bytes.len() {
                    out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                    i += 1;
                } else if c == quote {
                    break;
                }
            }
            continue;
        }
        if bytes[i] == b'r' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b'#' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'"' {
                let hashes = j - (i + 1);
                for _ in i..=j {
                    out.push(' ');
                }
                i = j + 1;
                while i < bytes.len() {
                    if bytes[i] == b'"'
                        && i + hashes < bytes.len()
                        && bytes[i + 1..=i + hashes].iter().all(|c| *c == b'#')
                    {
                        out.push(' ');
                        i += 1;
                        for _ in 0..hashes {
                            out.push(' ');
                            i += 1;
                        }
                        break;
                    }
                    out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                    i += 1;
                }
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn brace_depth_at(code: &str, index: usize) -> usize {
    code[..index].bytes().fold(0, |depth, byte| match byte {
        b'{' => depth + 1,
        b'}' => depth.saturating_sub(1),
        _ => depth,
    })
}

fn executable_call_positions(code: &str, call: &str) -> Vec<usize> {
    code.match_indices(call)
        .filter_map(|(index, _)| (brace_depth_at(code, index) == 0).then_some(index))
        .collect()
}

#[test]
fn production_generation_v1_uses_metrics_only_host_contract() {
    let source = fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/gpu_native/prototype_b_population_eval.rs"),
    )
    .expect("production population evaluator source must be readable");
    let code = code_only(&source);
    let start_anchor = "let ((rows, counters, host_prep, device_elapsed), residency_counters)";
    let end_anchor = "\n    let session_rebuilt";
    assert_eq!(code.match_indices(start_anchor).count(), 1);
    assert_eq!(code.match_indices(end_anchor).count(), 1);
    let start = code.find(start_anchor).expect("generation block start");
    let end = code[start..]
        .find(end_anchor)
        .map(|offset| start + offset)
        .expect("generation block end");
    let block = &code[start..end];
    for forbidden in [
        ".evaluate(",
        ".wait(",
        ".read_metrics(",
        ".read_residency_counters_v1(",
        "accepted_trade_total",
    ] {
        assert!(
            !block.contains(forbidden),
            "legacy token {forbidden:?} remains"
        );
    }
    let bind = "bind_exact_native_population_view_v1(";
    assert_eq!(block.match_indices(bind).count(), 1);
    let bind_start = start + block.find(bind).expect("resident bind");
    let marker = "|session| {";
    let marker_start = bind_start
        + code[bind_start..]
            .find(marker)
            .expect("resident bind closure");
    let normalized_bind_prefix: String = code[bind_start..marker_start + marker.len()]
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert_eq!(
        normalized_bind_prefix,
        "bind_exact_native_population_view_v1(native_device,|session|{"
    );
    let closure_open = marker_start + marker.len() - 1;
    let mut nested_depth = 0usize;
    let closure_close = code[closure_open + 1..]
        .bytes()
        .enumerate()
        .find_map(|(offset, byte)| match byte {
            b'{' => {
                nested_depth += 1;
                None
            }
            b'}' if nested_depth == 0 => Some(closure_open + 1 + offset),
            b'}' => {
                nested_depth -= 1;
                None
            }
            _ => None,
        })
        .expect("resident bind closure must have a matching close brace");
    assert!(
        closure_close < end,
        "resident bind closure must end inside generation block"
    );
    let closure_code = &code[closure_open + 1..closure_close];
    let enqueue = ".enqueue_metrics_only_v1(&native_settings)";
    let consume = ".consume_host_metrics_v1()";
    let enqueues = executable_call_positions(closure_code, enqueue);
    let consumes = executable_call_positions(closure_code, consume);
    assert_eq!(
        enqueues.len(),
        1,
        "exactly one executable metrics enqueue required"
    );
    assert_eq!(
        consumes.len(),
        1,
        "exactly one executable host-metrics consume required"
    );
    assert!(enqueues[0] < consumes[0], "enqueue must precede consume");
    let statement_start = closure_code[..enqueues[0]]
        .rfind(';')
        .map_or(0, |index| index + 1);
    let statement_end = consumes[0]
        + closure_code[consumes[0]..]
            .find(';')
            .expect("host metrics expression must terminate with a semicolon");
    let host_metrics_statement = &closure_code[statement_start..=statement_end];
    assert!(
        host_metrics_statement.contains("let host_metrics"),
        "enqueue and consume must belong to the named host_metrics owner binding"
    );
    assert_eq!(
        host_metrics_statement.match_indices(';').count(),
        1,
        "host_metrics binding must have exactly one terminating semicolon"
    );
    let enqueue_end = enqueues[0] + enqueue.len();
    let normalized_statement: String = host_metrics_statement
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert_eq!(
        normalized_statement,
        "lethost_metrics=session.enqueue_metrics_only_v1(&native_settings)?.consume_host_metrics_v1()?;"
    );
    let between_enqueue_and_consume = &closure_code[enqueue_end..consumes[0]];
    for exit_token in [
        "{", "}", ";", "return", "break", "continue", "bail!", "ensure!",
    ] {
        assert!(
            !between_enqueue_and_consume.contains(exit_token),
            "exit token {exit_token:?} must not occur between metrics enqueue and consume"
        );
    }
    for control in ["return", "break", "continue"] {
        for (index, _) in closure_code.match_indices(control) {
            if index < enqueues[0] && brace_depth_at(closure_code, index) == 0 {
                panic!("control-flow token {control:?} precedes metrics enqueue at closure depth");
            }
        }
    }
}

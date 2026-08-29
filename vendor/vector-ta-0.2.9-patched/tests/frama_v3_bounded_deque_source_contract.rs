use std::fs;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current directory must be readable"))
}

fn read(relative: impl AsRef<Path>) -> String {
    fs::read_to_string(manifest_dir().join(relative))
        .expect("the reviewed FRAMA source must be readable")
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let tail = &source[from..];
    let to = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing section end after {start}: {end}"));
    &tail[..to]
}

fn assert_before(source: &str, first: &str, second: &str) {
    let first_at = source
        .find(first)
        .unwrap_or_else(|| panic!("missing earlier source token: {first}"));
    let second_at = source
        .find(second)
        .unwrap_or_else(|| panic!("missing later source token: {second}"));
    assert!(
        first_at < second_at,
        "expected `{first}` before `{second}`, got {first_at} >= {second_at}"
    );
}

fn tail_from<'a>(source: &'a str, start: &str) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("missing tail start: {start}"));
    &source[from..]
}

#[test]
fn host_evenized_window_cap_is_shared_and_precedes_allocation_or_dispatch() {
    let source = read("src/indicators/moving_averages/frama.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("FRAMA production section must exist");

    assert!(production.contains("pub const FRAMA_MAX_WINDOW: usize = 1024;"));
    assert!(production.contains("fn frama_evenized_window_v3("));
    assert!(production.contains("if evenized > FRAMA_MAX_WINDOW"));

    let prepare = section(
        production,
        "fn frama_prepare<'a>(",
        "fn frama_compute_into(",
    );
    assert!(prepare.contains("frama_evenized_window_v3(window, len)?"));

    let scalar = section(
        production,
        "pub fn frama_scalar(",
        "pub struct FramaBatchRange",
    );
    assert_before(
        scalar,
        "frama_evenized_window_v3(window, len)?",
        "alloc_with_nan_prefix(len, warm)",
    );

    let batch = section(
        production,
        "fn frama_batch_inner(",
        "fn frama_batch_inner_into(",
    );
    assert_before(
        batch,
        "frama_batch_admission_v3(high, low, close, sweep)?",
        "make_uninit_matrix(rows, cols)",
    );
    assert_before(
        batch,
        "frama_batch_inner_into(high, low, close, sweep, kern, parallel, out)?",
        "ManuallyDrop::new(buf_mu)",
    );

    let batch_into = section(
        production,
        "fn frama_batch_inner_into(",
        "pub struct FramaStream",
    );
    assert_before(
        batch_into,
        "frama_batch_admission_v3(high, low, close, sweep)?",
        "let do_row = |row: usize, dst: &mut [f64]| unsafe",
    );

    let stream = section(
        production,
        "pub fn try_new(params: FramaParams)",
        "fn reset_finite_segment_v3",
    );
    assert_before(
        stream,
        "frama_evenized_window_v3(window, 0)?",
        "buffer: vec![(f64::NAN, f64::NAN, f64::NAN); n]",
    );
}

#[test]
fn strict_f64_cuda_is_bounded_o_n_and_uses_the_v3_transition_schedule() {
    let source = read("kernels/cuda/moving_averages/frama_kernel.cu");
    let wrapper = read("src/cuda/neoethos_f64_wrapper.rs");
    let kernel = tail_from(&source, "void frama_neo_batch_f64(");

    assert!(source.contains("#define NEO_FRAMA_MAX_WINDOW 1024"));
    assert!(source.contains("#define NEO_FRAMA_HALF_DEQUE_CAPACITY 513"));
    assert!(source.contains("struct NeoFramaDequeF64V3"));
    assert!(source.contains("neo_frama_push_max_f64_v3("));
    assert!(source.contains("neo_frama_push_min_f64_v3("));
    assert!(kernel.contains("if (win > NEO_FRAMA_MAX_WINDOW)"));

    for storage in [
        "left_max_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY]",
        "left_min_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY]",
        "right_max_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY]",
        "right_min_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY]",
    ] {
        assert!(kernel.contains(storage), "missing bounded deque: {storage}");
    }

    assert!(!kernel.contains("while (j + 1 < mid)"));
    assert!(!kernel.contains("while (j + 1 < i)"));
    assert!(!kernel.contains("DBL_MIN_FINITE"));

    assert_before(
        kernel,
        "const double max1 =",
        "const int idx_out = i - win;",
    );
    let transition = tail_from(kernel, "const int idx_out = i - win;");
    assert_before(
        transition,
        "const int idx_out = i - win;",
        "neo_frama_expire_f64_v3(&left_max, idx_out);",
    );
    assert_before(
        transition,
        "neo_frama_expire_f64_v3(&right_min, crossing);",
        "neo_frama_push_max_f64_v3(&left_max, crossing, high);",
    );
    assert_before(
        transition,
        "neo_frama_push_min_f64_v3(&left_min, crossing, low);",
        "neo_frama_push_max_f64_v3(&right_max, i, high);",
    );

    let wrapper_production = wrapper
        .split("#[cfg(test)]")
        .next()
        .expect("strict-f64 wrapper production section must exist");
    assert!(wrapper_production.contains("pub const FRAMA_MAX_WINDOW: usize = 1024;"));
    assert!(wrapper_production.contains("F64Kernel::Frama => Some(FRAMA_MAX_WINDOW),"));
    let compatibility = wrapper_production
        .split("pub fn sweep(")
        .nth(1)
        .expect("compatibility sweep must exist")
        .split("pub fn sweep_resident_v3(")
        .next()
        .expect("compatibility sweep must end before resident sweep");
    let resident = wrapper_production
        .split("pub fn sweep_resident_v3(")
        .nth(1)
        .expect("resident sweep must exist")
        .split("fn launch_chunk(")
        .next()
        .expect("resident sweep must end before launch");
    assert_before(
        compatibility,
        "if let Some(max) = kernel.max_period()",
        "DeviceBuffer::<f64>::uninitialized(output_elems)",
    );
    assert_before(
        resident,
        "if let Some(maximum) = kernel.max_period()",
        "DeviceBuffer::<f64>::uninitialized_async(output_elements",
    );
}

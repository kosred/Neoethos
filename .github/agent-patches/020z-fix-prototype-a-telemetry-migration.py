from pathlib import Path

path = Path(__file__).with_name("021-prototype-a-telemetry.py")
text = path.read_text(encoding="utf-8")
old = '''cube = replace_once(
    cube,
    '        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];\\n',
    '        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];\\n        transfer_telemetry::record_streamed_dataset_upload(\\n            ind_window.len().saturating_mul(std::mem::size_of::<F>())\\n                + smc_window.len().saturating_mul(std::mem::size_of::<i32>()),\\n        );\\n',
    "fused streamed dataset telemetry",
)
'''
new = '''stream_anchor = '        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];\\n'
stream_replacement = stream_anchor + '        transfer_telemetry::record_streamed_dataset_upload(\\n            ind_window.len().saturating_mul(std::mem::size_of::<F>())\\n                + smc_window.len().saturating_mul(std::mem::size_of::<i32>()),\\n        );\\n'
stream_matches = cube.count(stream_anchor)
if stream_matches != 2:
    raise RuntimeError(
        f"fused streamed dataset telemetry: expected both precision branches, found {stream_matches}"
    )
cube = cube.replace(stream_anchor, stream_replacement, 2)
'''
if text.count(old) != 1:
    raise RuntimeError("could not patch streamed telemetry migration exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

from pathlib import Path

path = Path("crates/neoethos-search/src/gpu_native/prototype_c_gpu.rs")
text = path.read_text(encoding="utf-8")
old_dispatch = '''pub fn try_prototype_c_gpu_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    #[cfg(feature = "gpu-cuda")]
    {
        let client = crate::cubecl_eval::create_gpu_client(None)?;
        return launch_sparse_first_hit::<CudaRuntime>(&client, path, events);
    }
    #[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
    {
        let client = crate::cubecl_eval::create_gpu_client(None)?;
        return launch_sparse_first_hit::<WgpuRuntime>(&client, path, events);
    }
    #[allow(unreachable_code)]
    bail!("Prototype C GPU path requires gpu-cuda or gpu-vulkan")
}
'''
new_dispatch = '''#[cfg(feature = "gpu-cuda")]
pub fn try_prototype_c_gpu_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_sparse_first_hit::<CudaRuntime>(&client, path, events)
}

#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
pub fn try_prototype_c_gpu_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_sparse_first_hit::<WgpuRuntime>(&client, path, events)
}
'''
if text.count(old_dispatch) != 1:
    raise RuntimeError(f"Prototype C dispatch block: expected one match, found {text.count(old_dispatch)}")
text = text.replace(old_dispatch, new_dispatch, 1)
old_readback = '''    let exit_bars = i32::from_bytes(&client.read_one(exit_bar_handle)).to_vec();
    let exit_reasons = i32::from_bytes(&client.read_one(exit_reason_handle)).to_vec();
'''
new_readback = '''    let exit_bar_bytes = client
        .read_one(exit_bar_handle)
        .context("read Prototype C exit bars")?;
    let exit_reason_bytes = client
        .read_one(exit_reason_handle)
        .context("read Prototype C exit reasons")?;
    let exit_bars = i32::from_bytes(exit_bar_bytes.as_ref()).to_vec();
    let exit_reasons = i32::from_bytes(exit_reason_bytes.as_ref()).to_vec();
'''
if text.count(old_readback) != 1:
    raise RuntimeError(f"Prototype C readback block: expected one match, found {text.count(old_readback)}")
text = text.replace(old_readback, new_readback, 1)
path.write_text(text, encoding="utf-8")
print("fixed generic CubeCL Prototype C dispatch and typed readbacks")

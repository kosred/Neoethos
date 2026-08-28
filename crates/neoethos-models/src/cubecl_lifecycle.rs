use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use anyhow::{Result, bail};
use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::{ComputeClient, Runtime},
};
use cubecl_common::stream_id::StreamId;

struct CubeClResidencyState {
    active: usize,
    clients: HashMap<(StreamId, usize), ComputeClient<CudaRuntime>>,
}

fn active_residency_scopes() -> &'static Mutex<CubeClResidencyState> {
    static STATE: OnceLock<Mutex<CubeClResidencyState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(CubeClResidencyState {
            active: 0,
            clients: HashMap::new(),
        })
    })
}

/// Keeps CubeCL's allocator pools alive across a complete model hot loop.
///
/// Low-level kernel entrypoints also enter a nested scope, so direct calls
/// synchronously clean their device and pinned-host pages. An outer evolution
/// scope keeps those pools hot across every generation and releases them only
/// after the final kernel handle has gone out of scope.
pub(crate) struct CubeClResidencyScope {
    active: bool,
}

pub(crate) fn cubecl_residency_scope() -> CubeClResidencyScope {
    let mut state = active_residency_scopes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.active = state
        .active
        .checked_add(1)
        .expect("model CubeCL residency scope count overflowed");
    CubeClResidencyScope { active: true }
}

fn record_cubecl_device(cuda_ordinal: usize, client: &ComputeClient<CudaRuntime>) {
    let stream = StreamId::current();
    let mut cleanup_client = client.clone();
    // SAFETY: this clone is retained only to clean the exact stream that made
    // the allocations. Kernel work continues through the caller's client.
    unsafe { cleanup_client.set_stream(stream) };

    let mut state = active_residency_scopes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        state.active > 0,
        "a model CubeCL client was created outside a residency scope"
    );
    state
        .clients
        .entry((stream, cuda_ordinal))
        .or_insert(cleanup_client);
}

pub(crate) fn cubecl_cuda_client(cuda_ordinal: usize) -> ComputeClient<CudaRuntime> {
    let client = CudaRuntime::client(&CudaDevice::new(cuda_ordinal));
    record_cubecl_device(cuda_ordinal, &client);
    client
}

fn release_cubecl_devices(
    clients: impl IntoIterator<Item = ((StreamId, usize), ComputeClient<CudaRuntime>)>,
) -> Result<()> {
    let mut failures = Vec::new();
    for ((stream, cuda_ordinal), client) in clients {
        if let Err(error) = cubecl::future::block_on(client.sync()) {
            failures.push(format!(
                "CUDA ordinal {cuda_ordinal} stream {stream:?} pre-cleanup synchronization failed: {error:?}"
            ));
            continue;
        }
        client.memory_cleanup();
        if let Err(error) = cubecl::future::block_on(client.sync()) {
            failures.push(format!(
                "CUDA ordinal {cuda_ordinal} stream {stream:?} pool-cleanup synchronization failed: {error:?}"
            ));
        }
    }
    if !failures.is_empty() {
        bail!("model CubeCL cleanup failed: {}", failures.join("; "));
    }
    Ok(())
}

fn release_cubecl_residency_scope() -> Result<()> {
    let clients = {
        let mut state = active_residency_scopes()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(state.active > 0, "model CubeCL residency scope underflow");
        state.active -= 1;
        if state.active != 0 {
            return Ok(());
        }
        state.clients.drain().collect::<Vec<_>>()
    };
    release_cubecl_devices(clients)
}

impl Drop for CubeClResidencyScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if let Err(error) = release_cubecl_residency_scope() {
            if std::thread::panicking() {
                eprintln!("model CubeCL cleanup also failed during unwind: {error:#}");
            } else {
                panic!("model CubeCL cleanup failed: {error:#}");
            }
        }
    }
}

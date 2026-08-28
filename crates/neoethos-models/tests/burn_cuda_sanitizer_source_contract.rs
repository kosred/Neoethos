const ROOT_MANIFEST: &str = include_str!("../../../Cargo.toml");
const ROOT_LOCK: &str = include_str!("../../../Cargo.lock");
const MODELS_MANIFEST: &str = include_str!("../Cargo.toml");
const MODELS_LIB: &str = include_str!("../src/lib.rs");
const BURN_MODELS: &str = include_str!("../src/burn_models.rs");
const SOFT_ACTOR_CRITIC: &str = include_str!("../src/soft_actor_critic.rs");
const CUBECL_LIFECYCLE: &str = include_str!("../src/cubecl_lifecycle.rs");
const BURN_CUDA_LIFECYCLE: &str = include_str!("burn_cuda_lifecycle.rs");
const CUBECL_RUNTIME_STREAM_POOL: &str =
    include_str!("../../../vendor/cubecl-runtime-0.10.0-patched/src/stream/base.rs");
const CUBECL_RUNTIME_MULTI_STREAM: &str =
    include_str!("../../../vendor/cubecl-runtime-0.10.0-patched/src/stream/event.rs");
const CUBECL_RUNTIME_SERVER: &str =
    include_str!("../../../vendor/cubecl-runtime-0.10.0-patched/src/server/base.rs");
const CUBECL_RUNTIME_CLIENT: &str =
    include_str!("../../../vendor/cubecl-runtime-0.10.0-patched/src/client.rs");
const CUBECL_RUNTIME_MEMORY_USAGE: &str =
    include_str!("../../../vendor/cubecl-runtime-0.10.0-patched/src/memory_management/base.rs");
const CUBECL_CUDA_SERVER: &str =
    include_str!("../../../vendor/cubecl-cuda-0.10.0-patched/src/compute/server.rs");
const CUBEK_TILING_LAUNCH: &str =
    include_str!("../../../vendor/cubek-matmul-0.2.0-patched/src/launch/launch_tiling.rs");
const CUBEK_CONVOLUTION_LAUNCH: &str =
    include_str!("../../../vendor/cubek-convolution-0.2.0-patched/src/launch/base.rs");
const CUBEK_CONVOLUTION_SIMPLE_ROUTINE: &str =
    include_str!("../../../vendor/cubek-convolution-0.2.0-patched/src/routines/simple.rs");
const CUBEK_CONVOLUTION_SPECIALIZED_ROUTINE: &str =
    include_str!("../../../vendor/cubek-convolution-0.2.0-patched/src/routines/specialized.rs");
const CUBEK_CONVOLUTION_FORWARD_ARGS: &str =
    include_str!("../../../vendor/cubek-convolution-0.2.0-patched/src/kernels/forward/args.rs");

fn package_block<'a>(lockfile: &'a str, package_name: &str) -> &'a str {
    lockfile
        .split("[[package]]")
        .find(|block| block.contains(&format!("name = \"{package_name}\"")))
        .unwrap_or_else(|| panic!("Cargo.lock is missing {package_name}"))
}

#[test]
fn workspace_selects_the_patched_cubek_matmul_source() {
    assert!(
        ROOT_MANIFEST.contains("cubek-matmul = { path = \"vendor/cubek-matmul-0.2.0-patched\" }"),
        "the workspace must select the capability-filtered CubeK source"
    );

    let cubek_lock = package_block(ROOT_LOCK, "cubek-matmul");
    assert!(cubek_lock.contains("version = \"0.2.0\""));
    assert!(
        !cubek_lock.contains("source = \"registry+") && !cubek_lock.contains("checksum = "),
        "Cargo.lock must identify cubek-matmul as the selected path package"
    );
}

#[test]
fn cubecl_cleanup_enumerates_every_initialized_cuda_stream() {
    assert!(
        ROOT_MANIFEST
            .contains("cubecl-runtime = { path = \"vendor/cubecl-runtime-0.10.0-patched\" }"),
        "the workspace must select the device-wide CubeCL runtime cleanup backport"
    );

    let runtime_lock = package_block(ROOT_LOCK, "cubecl-runtime");
    assert!(runtime_lock.contains("version = \"0.10.0\""));
    assert!(
        !runtime_lock.contains("source = \"registry+") && !runtime_lock.contains("checksum = "),
        "Cargo.lock must identify cubecl-runtime as the selected path package"
    );

    let stream_ids = CUBECL_RUNTIME_STREAM_POOL
        .split("pub fn stream_ids")
        .nth(1)
        .expect("StreamPool must expose initialized stream ids")
        .split("pub fn get_mut")
        .next()
        .expect("StreamPool::get_mut must follow stream_ids");
    assert!(stream_ids.contains("self.streams[..self.max_streams]"));
    assert!(stream_ids.contains(".filter_map("));
    assert!(
        !stream_ids.contains("get_mut("),
        "enumeration must not initialize empty stream slots"
    );

    assert!(
        CUBECL_RUNTIME_MULTI_STREAM
            .contains("pub fn stream_ids(&self) -> impl Iterator<Item = StreamId> + '_")
    );
    assert!(CUBECL_RUNTIME_MULTI_STREAM.contains("self.streams.stream_ids()"));
    assert!(
        CUBECL_RUNTIME_SERVER.contains("fn stream_ids(&self) -> Vec<StreamId>"),
        "ComputeServer must expose a device-wide stream enumeration boundary"
    );

    let memory_usage = CUBECL_RUNTIME_CLIENT
        .split("pub fn memory_usage")
        .nth(1)
        .expect("ComputeClient::memory_usage is missing")
        .split("pub fn enumerate_devices")
        .next()
        .expect("device enumeration must follow memory usage");
    assert!(memory_usage.contains(".stream_ids()"));
    assert!(memory_usage.contains("try_fold(MemoryUsage::default()"));
    assert!(memory_usage.contains("acc.combine(server.memory_usage(id)?)"));
    assert!(
        CUBECL_RUNTIME_MEMORY_USAGE.contains("#[derive(Debug, Clone, Default, PartialEq, Eq)]"),
        "device-wide memory usage needs an explicit zero report for an unused device"
    );

    let cleanup = CUBECL_RUNTIME_CLIENT
        .split("pub fn memory_cleanup")
        .nth(1)
        .expect("ComputeClient::memory_cleanup is missing")
        .split("pub fn profile")
        .next()
        .expect("profiling must follow memory cleanup");
    assert!(cleanup.contains("for id in server.stream_ids()"));
    assert!(cleanup.contains("server.memory_cleanup(id);"));

    assert!(CUBECL_CUDA_SERVER.contains("fn stream_ids(&self) -> Vec<StreamId>"));
    assert!(CUBECL_CUDA_SERVER.contains("self.streams.stream_ids().collect()"));
}

#[test]
fn tma_capability_is_rejected_before_any_tma_binding_or_allocation() {
    let launch = CUBEK_TILING_LAUNCH
        .split("pub fn launch_ref_tma")
        .nth(1)
        .expect("CubeK must retain launch_ref_tma")
        .split("fn launch_inner_ref")
        .next()
        .expect("CubeK must retain launch_inner_ref after launch_ref_tma");

    let capability_guard = launch
        .find("client.properties().features.tma.contains(Tma::Base)")
        .expect("TMA launch must preflight the runtime capability");
    let unavailable = launch
        .find("MatmulAvailabilityError::TmaUnavailable")
        .expect("unsupported TMA must return the normal unavailable result");
    let lhs_preparation = launch
        .find("let lhs = match matrix_batch_layout")
        .expect("TMA launch must still prepare its lhs after the guard");
    let tensor_map_launch = launch
        .find("launch_inner_ref::<R, TensorMapArgs")
        .expect("TMA launch must still use TensorMapArgs on capable devices");

    assert!(capability_guard < unavailable);
    assert!(unavailable < lhs_preparation);
    assert!(lhs_preparation < tensor_map_launch);
}

#[test]
fn workspace_selects_the_patched_cubek_convolution_source() {
    assert!(
        ROOT_MANIFEST
            .contains("cubek-convolution = { path = \"vendor/cubek-convolution-0.2.0-patched\" }")
    );

    let convolution_lock = package_block(ROOT_LOCK, "cubek-convolution");
    assert!(convolution_lock.contains("version = \"0.2.0\""));
    assert!(
        !convolution_lock.contains("source = \"registry+")
            && !convolution_lock.contains("checksum = "),
        "Cargo.lock must identify cubek-convolution as the selected path package"
    );
}

#[test]
fn convolution_tma_is_rejected_before_tensor_map_descriptor_apis() {
    let launch = CUBEK_CONVOLUTION_LAUNCH
        .split("pub fn launch_ref")
        .nth(1)
        .expect("CubeK convolution must retain its unified launch_ref")
        .split("fn dispatch_routine")
        .next()
        .expect("routine dispatch must follow convolution launch_ref");
    let preflight = launch
        .split("let (algorithm, tile_kind, forced_matmul)")
        .nth(1)
        .expect("convolution launch must resolve its algorithm first")
        .split("// Backward-data does not currently support")
        .next()
        .expect("the operation-specific guard must follow capability preflight");

    let tma_algorithms = preflight
        .find("matches!(")
        .expect("both convolution TMA algorithms need one capability gate");
    assert!(preflight.contains("ConvAlgorithm::SimpleAsyncTma"));
    assert!(preflight.contains("ConvAlgorithm::SpecializedTma"));
    let capability = preflight
        .find("client.properties().features.tma.contains(Tma::Base)")
        .expect("convolution TMA must preflight the exact runtime capability");
    let unavailable = preflight
        .find("MatmulAvailabilityError::TmaUnavailable")
        .expect("unsupported convolution TMA must use the normal availability error");
    let dispatch = launch
        .find("dispatch_routine::<R, N_SPATIAL>")
        .expect("convolution launch must retain routine dispatch");
    assert!(tma_algorithms < capability && capability < unavailable && unavailable < dispatch);

    assert!(CUBEK_CONVOLUTION_SIMPLE_ROUTINE.contains("into_tensor_handle_tma("));
    assert!(CUBEK_CONVOLUTION_SPECIALIZED_ROUTINE.contains("into_tensor_handle_tma("));
    assert!(CUBEK_CONVOLUTION_FORWARD_ARGS.contains("ViewArg::new_tensor_map_im2col"));
    assert!(CUBEK_CONVOLUTION_FORWARD_ARGS.contains("ViewArg::new_tensor_map_tiled"));
}

#[test]
fn burn_cuda_enables_and_exposes_the_shared_cubecl_cleanup_scope() {
    assert!(MODELS_MANIFEST.contains(
        "burn-cuda-backend = [\"dep:burn-cuda\", \"dep:cubecl\", \"dep:cubecl-common\", \"cubecl/cuda\"]"
    ));
    assert!(MODELS_LIB.contains("feature = \"burn-cuda-backend\"\n))]\nmod cubecl_lifecycle;"));
    assert!(BURN_MODELS.contains(
        "pub fn burn_cuda_residency_scope(cuda_ordinal: usize) -> BurnCudaResidencyScope"
    ));

    let pre_sync = CUBECL_LIFECYCLE
        .find("block_on(client.sync())")
        .expect("CubeCL cleanup must synchronize before releasing pools");
    let cleanup = CUBECL_LIFECYCLE
        .find("client.memory_cleanup();")
        .expect("CubeCL cleanup must release device and pinned-host pools");
    let post_sync = CUBECL_LIFECYCLE[cleanup + 1..]
        .find("block_on(client.sync())")
        .map(|offset| cleanup + 1 + offset)
        .expect("CubeCL cleanup must synchronize after releasing pools");
    assert!(pre_sync < cleanup && cleanup < post_sync);
}

#[test]
fn burn_fusion_handles_are_drained_before_cubecl_pool_cleanup() {
    let scope_drop = BURN_MODELS
        .split("impl Drop for BurnCudaResidencyScope")
        .nth(1)
        .expect("Burn CUDA scope needs an explicit fusion-aware destructor")
        .split("fn active_burn_backend_name")
        .next()
        .expect("Burn backend naming must follow the CUDA scope destructor");

    let fusion_sync = scope_drop
        .find("<InferBackend as Backend>::sync(&CudaDevice::new(self.cuda_ordinal))")
        .expect("Burn Fusion's queued tensor drops must be drained");
    let cubecl_cleanup = scope_drop
        .find("drop(self.cubecl_residency.take())")
        .expect("the shared CubeCL scope must run after Fusion releases its handles");
    assert!(
        fusion_sync < cubecl_cleanup,
        "Burn Fusion must release its device handles before CubeCL cleans the allocator pool"
    );
}

#[test]
fn every_burn_cuda_lifecycle_owns_an_outer_cleanup_scope() {
    assert!(BURN_MODELS.contains("let _cuda_residency = burn_cuda_residency_scope(0);"));

    let deep_helper = BURN_CUDA_LIFECYCLE
        .split("fn exercise_deep_cuda_lifecycle")
        .nth(1)
        .expect("deep CUDA lifecycle helper is missing")
        .split("macro_rules! deep_cuda_lifecycle_gate")
        .next()
        .expect("deep CUDA lifecycle macro must follow its helper");
    assert!(deep_helper.contains("let _cuda_residency = burn_cuda_residency_scope(CUDA_ORDINAL);"));

    for gate in [
        "fn burn_cuda_exit_agent_lifecycle_gpu0",
        "fn burn_cuda_sac_lifecycle_gpu0",
    ] {
        let body = BURN_CUDA_LIFECYCLE
            .split(gate)
            .nth(1)
            .unwrap_or_else(|| panic!("missing gate {gate}"));
        let residency = body
            .find("let _cuda_residency = burn_cuda_residency_scope(CUDA_ORDINAL);")
            .unwrap_or_else(|| panic!("{gate} is missing its outer cleanup scope"));
        let model = body
            .find("let mut agent =")
            .unwrap_or_else(|| panic!("{gate} is missing its CUDA agent"));
        assert!(
            residency < model,
            "{gate} must create its cleanup scope before CUDA handles"
        );
    }
}

#[test]
fn burn_low_level_gate_observes_fusion_handle_quiescence_before_pool_cleanup() {
    assert!(
        MODELS_MANIFEST.contains(
            "[dev-dependencies]\nburn-fusion = { version = \"0.21\", features = [\"test-util\"] }"
        ),
        "Fusion handle introspection must remain a test-only dependency"
    );

    let gate = BURN_MODELS
        .split("fn burn_cuda_auto_precision_runs_three_epoch_real_kernels_in_fp32")
        .nth(1)
        .expect("Burn CUDA low-level kernel gate is missing")
        .split("fn burn_cuda_rejects_cpu_and_malformed_device_policies")
        .next()
        .expect("the next Burn CUDA unit gate must follow the low-level gate");

    let inspector = gate
        .find("FusionInspector::install(StreamId::current())")
        .expect("the low-level gate must observe Burn's exact Fusion stream");
    let baseline_sync = gate
        .find("<InferBackend as Backend>::sync(&CudaDevice::new(0))")
        .expect("the inspector baseline must follow a quiescent Fusion sync");
    let baseline = gate
        .find("fusion_inspector.set_baseline()")
        .expect("the low-level gate must exclude pre-existing device handles");
    let work_scope = gate
        .find("let kernel_result = (|| -> anyhow::Result<()> {")
        .expect("all CUDA handles must be naturally dropped in an inner work scope");
    let training = gate
        .find("train_model_with_report_on_device::<TrainBackend")
        .expect("the diagnostic gate must retain real CUDA training");
    let post_scope_sync = gate
        .rfind("<InferBackend as Backend>::sync(&CudaDevice::new(0))")
        .expect("Fusion must be drained after every work-scope handle is dropped");
    let leaked_handles = gate
        .find("fusion_inspector.new_handles_since_baseline()")
        .expect("the gate must report exact new Fusion handle IDs");
    let kernel_result_check = gate
        .find("kernel_result?")
        .expect("the original CUDA lifecycle result must still be propagated");

    assert!(
        inspector < baseline_sync
            && baseline_sync < baseline
            && baseline < work_scope
            && work_scope < training
            && training < post_scope_sync
            && post_scope_sync < leaked_handles
            && leaked_handles < kernel_result_check,
        "Fusion must be measured only after the inner CUDA work scope is fully dropped"
    );
    assert!(
        gate.contains("assert!(\n            leaked_handles.is_empty(),")
            && gate.contains("{leaked_handles:?}"),
        "the RTX RED must fail with the exact leaked Fusion handle IDs"
    );
    assert!(
        !gate.contains("drop(model")
            && !gate.contains("drop(_trained")
            && !gate.contains("drop(infer_model")
            && !gate.contains("drop(probabilities"),
        "the diagnostic must prove normal lexical lifetime behavior, not test-local drops"
    );
}

#[test]
fn burn_validation_uses_the_official_non_autodiff_inner_backend() {
    let training_functions = [
        "pub fn train_model<B, M>",
        "pub fn train_model_with_report<B, M>",
        "pub fn train_model_with_report_on_device<B, M>",
        "pub fn train_model_with_report_with_selection<B, M>",
        "pub fn train_model_with_report_with_selection_and_precision<B, M>",
        "pub fn train_model_with_report_with_external_val<B, M>",
    ];
    for function in training_functions {
        let item = BURN_MODELS
            .split(function)
            .nth(1)
            .unwrap_or_else(|| panic!("the Burn training function {function} is missing"))
            .split("\npub fn ")
            .next()
            .expect("the next public function must delimit this training item");
        assert!(
            item.contains("M::InnerModule: BurnForward<B::InnerBackend>"),
            "{function} must prove that the official inner module can run validation"
        );
    }

    let training = BURN_MODELS
        .split("pub fn train_model_with_report_with_external_val<B, M>")
        .nth(1)
        .expect("the Burn training leaf function is missing")
        .split("\npub fn predict_proba")
        .next()
        .expect("prediction must follow the Burn training leaf function");

    let validation = training
        .split("// Validation on holdout")
        .nth(1)
        .expect("the holdout validation block is missing")
        .split("epochs_ran = epoch + 1;")
        .next()
        .expect("the epoch accounting must follow holdout validation");

    assert!(
        validation.contains("let valid_model = model.valid();"),
        "validation must convert the training module with Burn's official AutodiffModule::valid API"
    );
    assert!(validation.contains("array2_to_tensor_with_dtype::<B::InnerBackend>"));
    assert!(validation.contains("labels_to_tensor::<B::InnerBackend>"));
    assert!(validation.contains("BurnForward::forward_pass(&valid_model, x_val)"));
    assert!(validation.contains("cross_entropy_loss::<B::InnerBackend>("));
    assert!(
        !validation.contains("array2_to_tensor_with_dtype::<B>(")
            && !validation.contains("labels_to_tensor::<B>(")
            && !validation.contains("BurnForward::forward_pass(&model, x_val)"),
        "validation must not create a final-epoch autodiff graph that has no backward pass"
    );
}

#[test]
fn sac_stop_gradient_work_uses_the_official_non_autodiff_inner_backend() {
    let target_update = SOFT_ACTOR_CRITIC
        .split("fn soft_update_targets(&mut self)")
        .nth(1)
        .expect("the SAC target update is missing")
        .split("/// Forward the full training batch")
        .next()
        .expect("SAC batch training must follow the target update");
    assert!(target_update.contains("self.target_critic1.valid()"));
    assert!(target_update.contains("self.critic1.valid()"));
    assert!(target_update.contains("self.target_critic2.valid()"));
    assert!(target_update.contains("self.critic2.valid()"));
    assert!(target_update.contains("as AutodiffModule<TrainBackend>>::from_inner"));

    let training = SOFT_ACTOR_CRITIC
        .split("fn update_on_batch(&mut self, batch: &[SacTuple])")
        .nth(1)
        .expect("the SAC batch-training function is missing")
        .split("/// Train from the canonical typed feature frame")
        .next()
        .expect("the public SAC training entrypoint must follow batch training");

    for inner_tensor in [
        "next_states: Tensor<InferBackend, 2>",
        "rewards: Tensor<InferBackend, 2>",
        "not_done: Tensor<InferBackend, 2>",
    ] {
        assert!(
            training.contains(inner_tensor),
            "SAC stop-gradient input must use the inner backend: {inner_tensor}"
        );
    }
    for valid_module in [
        "self.temperature.valid()",
        "self.actor.valid()",
        "self.target_critic1.valid()",
        "self.target_critic2.valid()",
        "self.critic1.valid()",
        "self.critic2.valid()",
    ] {
        assert!(
            training.contains(valid_module),
            "SAC stop-gradient module must cross AutodiffModule::valid: {valid_module}"
        );
    }
    assert!(training.contains("let critic_states = states.inner();"));
    assert!(training.contains("log_probs.inner()"));
    assert!(training.contains("probs.inner()"));
    assert!(training.contains("Tensor::<TrainBackend, 2>::from_inner"));
    assert!(
        !training.contains(".detach()"),
        "SAC must not leave detached autodiff graphs alive; constants belong on the inner backend"
    );

    let helpers = SOFT_ACTOR_CRITIC
        .split("fn polyak_update(")
        .nth(1)
        .expect("the SAC Polyak helper is missing")
        .split("fn scalar_from_tensor")
        .next()
        .expect("the scalar helper must follow SAC Polyak helpers");
    assert!(helpers.contains("dst: SacCriticNet<InferBackend>"));
    assert!(helpers.contains("src: &SacCriticNet<InferBackend>"));
    assert!(helpers.contains("dst: nn::Linear<InferBackend>"));
    assert!(helpers.contains("src: &nn::Linear<InferBackend>"));
    assert!(
        !helpers.contains("TrainBackend"),
        "Polyak target updates must never build an autodiff graph"
    );
}

#[test]
fn sac_runtime_inference_never_builds_an_autodiff_graph() {
    let inference = SOFT_ACTOR_CRITIC
        .split("fn policy_probabilities_f64(&self, state: &[f64])")
        .nth(1)
        .expect("the typed SAC inference function is missing")
        .split("pub fn predict_runtime(")
        .next()
        .expect("the SAC runtime entrypoint must follow typed inference");

    assert!(
        inference.contains("Tensor::<InferBackend, 1>::from_data"),
        "runtime SAC inputs must be allocated directly on the non-autodiff backend"
    );
    assert!(
        inference.contains("let inference_actor = self.actor.valid();")
            && inference.contains("inference_actor\n            .policy(state_tensor)"),
        "runtime SAC actor inference must use Burn's official validation module"
    );
    assert!(
        !inference.contains("Tensor::<TrainBackend") && !inference.contains("self.actor.policy("),
        "runtime SAC inference must not create an autodiff graph with no backward pass"
    );
}

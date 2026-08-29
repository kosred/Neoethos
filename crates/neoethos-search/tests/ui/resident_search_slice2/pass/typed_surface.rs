use neoethos_gpu_cuda::resident_search_slice2_v3 as gpu;
use neoethos_search::resident_search_slice2_v3 as search;

fn gpu_calibration() -> gpu::ResidentArchiveKnnCalibrationReceiptV2 { panic!() }
fn gpu_chain() -> gpu::ResidentSearchGenerationChainV3 { panic!() }
fn gpu_ranked() -> gpu::ResidentSearchRankEnqueuedV3 { panic!() }
fn gpu_staged() -> gpu::ResidentSearchArchiveStagedV3 { panic!() }
fn gpu_pending() -> gpu::ResidentSearchTerminalPendingV3 { panic!() }
fn gpu_terminal() -> gpu::ResidentSearchTerminalReceiptV3 { panic!() }
fn gpu_try_complete() -> gpu::ResidentSearchTryCompleteV3 { panic!() }
fn gpu_error() -> gpu::ResidentSearchTransitionErrorV3 { panic!() }
fn gpu_rejection() -> gpu::ResidentSearchRejectedAuthorityV3<gpu::ResidentSearchGenerationChainV3> {
    panic!()
}
fn full_deadline() -> search::FullResidentDiscoveryDeadlineReceiptV1 { panic!() }

fn take_calibration(_: search::ResidentArchiveKnnCalibrationReceiptV2) {}
fn take_chain(_: search::ResidentSearchGenerationChainV3) {}
fn take_ranked(_: search::ResidentSearchRankEnqueuedV3) {}
fn take_staged(_: search::ResidentSearchArchiveStagedV3) {}
fn take_pending(_: search::ResidentSearchTerminalPendingV3) {}
fn take_terminal(_: search::ResidentSearchTerminalReceiptV3) {}
fn take_try_complete(_: search::ResidentSearchTryCompleteV3) {}
fn take_error(_: search::ResidentSearchTransitionErrorV3) {}
fn take_rejection(
    _: search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
) {}
fn take_deadline(_: search::FullResidentDiscoveryDeadlineReceiptV1) {}

fn main() {
    take_calibration(gpu_calibration());
    take_chain(gpu_chain());
    take_ranked(gpu_ranked());
    take_staged(gpu_staged());
    take_pending(gpu_pending());
    take_terminal(gpu_terminal());
    take_try_complete(gpu_try_complete());
    take_error(gpu_error());
    take_rejection(gpu_rejection());
    take_deadline(full_deadline());

    let _: fn(search::ResidentSearchGenerationChainV3) -> Result<
        search::ResidentSearchRankEnqueuedV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    > = search::ResidentSearchGenerationChainV3::enqueue_score_and_rank_v3;
    let _: fn(search::ResidentSearchGenerationChainV3) -> Result<
        search::ResidentSearchTerminalPendingV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    > = search::ResidentSearchGenerationChainV3::enqueue_terminal_seal_v3;
    let _: fn(search::ResidentSearchRankEnqueuedV3) -> Result<
        search::ResidentSearchArchiveStagedV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchRankEnqueuedV3>,
    > = search::ResidentSearchRankEnqueuedV3::enqueue_stage_archive_from_rank_v3;
    let _: fn(search::ResidentSearchArchiveStagedV3) -> Result<
        search::ResidentSearchGenerationChainV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchArchiveStagedV3>,
    > = search::ResidentSearchArchiveStagedV3::enqueue_evolve_and_publish_v3;
    let _: fn(search::ResidentSearchTerminalPendingV3) -> Result<
        search::ResidentSearchTryCompleteV3,
        search::ResidentSearchTransitionErrorV3,
    > = search::ResidentSearchTerminalPendingV3::try_complete_v3;
    let _: fn(
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    ) -> (
        search::ResidentSearchTransitionErrorV3,
        search::ResidentSearchGenerationChainV3,
    ) = search::ResidentSearchRejectedAuthorityV3::<
        search::ResidentSearchGenerationChainV3,
    >::into_parts_v3;

    let _ = search::ResidentSearchTryCompleteV3::NotReady;
    let _ = search::ResidentSearchTryCompleteV3::Complete;
}

use neoethos_gpu_cuda::resident_search_slice2_v3::ResidentArchiveKnnCalibrationReceiptV2;
use neoethos_search::resident_search_slice2_v3::FullResidentDiscoveryDeadlineReceiptV1;

fn calibration() -> ResidentArchiveKnnCalibrationReceiptV2 {
    loop {}
}

fn require_full_deadline(_: FullResidentDiscoveryDeadlineReceiptV1) {}

fn main() {
    require_full_deadline(calibration());
}

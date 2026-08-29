use std::sync::Arc;

use neoethos_app::app_services::canonical_native_discovery::{
    CanonicalNativeResearchIntentV1, CanonicalNativeResearchJobHandleV1,
    CanonicalNativeResearchStartErrorV1, start_canonical_native_research_lane_v1,
};
use neoethos_app::app_services::jobs::JobKind;
use neoethos_core::Settings;
use neoethos_search::CanonicalNativeRuntimeInstallReceiptV1;

#[test]
fn canonical_native_research_lane_has_a_distinct_typed_start_boundary() {
    let _: fn(
        Arc<Settings>,
        Arc<CanonicalNativeRuntimeInstallReceiptV1>,
        CanonicalNativeResearchIntentV1,
    )
        -> Result<CanonicalNativeResearchJobHandleV1, CanonicalNativeResearchStartErrorV1> =
        start_canonical_native_research_lane_v1;

    assert_eq!(
        JobKind::CanonicalNativeResearch.as_str(),
        "canonical_native_research"
    );
}

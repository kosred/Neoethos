use crate::canonical_native_generation_zero_publication_v1::{
    CanonicalNativeGenerationZeroPublicationErrorV1,
    CanonicalNativeGenerationZeroPublicationGateRejectionV1,
    CanonicalNativeGenerationZeroPublicationReceiptV1,
    publish_canonical_native_generation_zero_research_result_v1,
};
use crate::canonical_native_generation_zero_result_v1::{
    CanonicalNativeGenerationZeroCompactJsonSealV1,
    CanonicalNativeGenerationZeroResearchResultViewV1,
};
use crate::canonical_native_root_io_v1::SealedCanonicalRootV1;

#[test]
fn high_level_generation_zero_publisher_api_exists() {
    let _: fn(
        &SealedCanonicalRootV1,
        &CanonicalNativeGenerationZeroResearchResultViewV1<'_>,
        &CanonicalNativeGenerationZeroCompactJsonSealV1,
        fn() -> Result<(), CanonicalNativeGenerationZeroPublicationGateRejectionV1>,
    ) -> Result<
        CanonicalNativeGenerationZeroPublicationReceiptV1,
        CanonicalNativeGenerationZeroPublicationErrorV1,
    > = publish_canonical_native_generation_zero_research_result_v1;
}

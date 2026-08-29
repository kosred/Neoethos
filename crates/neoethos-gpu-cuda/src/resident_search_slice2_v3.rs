#[cfg(feature = "cuda")]
use crate::resident_trim_prefilter_v1::ResidentTrimmedPopulationSessionV1;

#[cfg(feature = "cuda")]
struct ResidentSearchAuthorityStateV3 {
    session: ResidentTrimmedPopulationSessionV1,
}

#[cfg(feature = "cuda")]
enum ResidentSearchRejectedTransitionV3 {
    ScoreAndRank,
    TerminalSeal,
    StageArchiveFromRank,
    EvolveAndPublish,
}

pub struct ResidentArchiveKnnCalibrationReceiptV2 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchGenerationChainV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchRankEnqueuedV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchArchiveStagedV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchTerminalPendingV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchTerminalReceiptV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

pub enum ResidentSearchTryCompleteV3 {
    NotReady(ResidentSearchTerminalPendingV3),
    Complete(ResidentSearchTerminalReceiptV3),
}

pub struct ResidentSearchTransitionErrorV3 {
    #[cfg(feature = "cuda")]
    inner: ResidentSearchRejectedTransitionV3,
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchRejectedAuthorityV3<A> {
    error: ResidentSearchTransitionErrorV3,
    authority: A,
}

impl ResidentSearchGenerationChainV3 {
    pub fn enqueue_score_and_rank_v3(
        self,
    ) -> Result<ResidentSearchRankEnqueuedV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.inner.session;
            Err(reject_resident_search_transition_v3(
                self,
                ResidentSearchRejectedTransitionV3::ScoreAndRank,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.inner {}
        }
    }

    pub fn enqueue_terminal_seal_v3(
        self,
    ) -> Result<ResidentSearchTerminalPendingV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.inner.session;
            Err(reject_resident_search_transition_v3(
                self,
                ResidentSearchRejectedTransitionV3::TerminalSeal,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.inner {}
        }
    }
}

impl ResidentSearchRankEnqueuedV3 {
    pub fn enqueue_stage_archive_from_rank_v3(
        self,
    ) -> Result<ResidentSearchArchiveStagedV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.inner.session;
            Err(reject_resident_search_transition_v3(
                self,
                ResidentSearchRejectedTransitionV3::StageArchiveFromRank,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.inner {}
        }
    }
}

impl ResidentSearchArchiveStagedV3 {
    pub fn enqueue_evolve_and_publish_v3(
        self,
    ) -> Result<ResidentSearchGenerationChainV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.inner.session;
            Err(reject_resident_search_transition_v3(
                self,
                ResidentSearchRejectedTransitionV3::EvolveAndPublish,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.inner {}
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    pub(crate) fn from_ranked_v3(
        ranked: ResidentSearchRankEnqueuedV3,
    ) -> ResidentSearchArchiveStagedV3 {
        match ranked.inner {}
    }
}

#[cfg(feature = "cuda")]
impl Drop for ResidentArchiveKnnCalibrationReceiptV2 {
    fn drop(&mut self) {
        let _ = &self.inner;
    }
}

#[cfg(feature = "cuda")]
impl Drop for ResidentSearchTerminalReceiptV3 {
    fn drop(&mut self) {
        let _ = &self.inner;
    }
}

#[cfg(feature = "cuda")]
impl Drop for ResidentSearchTransitionErrorV3 {
    fn drop(&mut self) {
        let _ = &self.inner;
    }
}

impl ResidentSearchTerminalPendingV3 {
    pub fn try_complete_v3(
        self,
    ) -> Result<ResidentSearchTryCompleteV3, ResidentSearchTransitionErrorV3> {
        #[cfg(feature = "cuda")]
        {
            let _ = &self.inner.session;
            Ok(ResidentSearchTryCompleteV3::NotReady(self))
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.inner {}
        }
    }
}

impl<A> ResidentSearchRejectedAuthorityV3<A> {
    pub fn into_parts_v3(self) -> (ResidentSearchTransitionErrorV3, A) {
        #[cfg(feature = "cuda")]
        let _ = &self.error.inner;
        (self.error, self.authority)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn start_resident_search_slice2_v3(
    session: ResidentTrimmedPopulationSessionV1,
) -> ResidentSearchGenerationChainV3 {
    ResidentSearchGenerationChainV3 {
        inner: ResidentSearchAuthorityStateV3 { session },
    }
}

#[cfg(feature = "cuda")]
fn reject_resident_search_transition_v3<A>(
    authority: A,
    inner: ResidentSearchRejectedTransitionV3,
) -> ResidentSearchRejectedAuthorityV3<A> {
    ResidentSearchRejectedAuthorityV3 {
        error: ResidentSearchTransitionErrorV3 { inner },
        authority,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_authority_state_is_private_move_only_and_fail_closed() {
        let source = include_str!("resident_search_slice2_v3.rs");
        let trim_source = include_str!("resident_trim_prefilter_v1.rs");

        assert_eq!(source.matches(concat!("pub ", "struct ")).count(), 8);
        assert_eq!(source.matches(concat!("pub ", "enum ")).count(), 1);
        assert_eq!(source.matches(concat!("    pub ", "fn ")).count(), 6);
        assert_eq!(
            source
                .matches(concat!("inner: core::convert::", "Infallible,"))
                .count(),
            7
        );
        assert_eq!(
            source
                .matches(concat!("inner: ResidentSearchAuthority", "StateV3,"))
                .count(),
            6
        );
        assert!(source.contains("struct ResidentSearchAuthorityStateV3 {"));
        assert!(source.contains("session: ResidentTrimmedPopulationSessionV1,"));
        assert!(source.contains("pub(crate) fn start_resident_search_slice2_v3("));
        assert_eq!(
            source.matches(concat!("                ", "self,")).count(),
            4
        );
        assert!(source.contains("Ok(ResidentSearchTryCompleteV3::NotReady(self))"));
        assert!(trim_source.contains("pub fn begin_resident_search_slice2_v3("));
    }
}

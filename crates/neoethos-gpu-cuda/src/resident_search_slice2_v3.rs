#[cfg(feature = "cuda")]
use crate::resident_feature_store_v3::ResidentFeatureStoreConsumerLeaseV3;
#[cfg(feature = "cuda")]
use crate::resident_generation_v1::SealedResidentGenerationPlanV1;
#[cfg(feature = "cuda")]
use crate::resident_search_slice2_admission_v2::ResidentSearchSlice2ValidatedRuntimeAuthorityV2;
#[cfg(feature = "cuda")]
use crate::resident_search_v2::{
    ResidentSearchSlice2NativeErrorV3, ResidentSearchSlice2NativeOwnerV3,
    ResidentSearchSlice2NativeTryCompleteV3,
};
#[cfg(feature = "cuda")]
use crate::resident_trim_prefilter_v1::ResidentTrimmedPopulationSessionV1;
#[cfg(feature = "cuda")]
use crate::{NeoPopulationSettings, ScenarioDescriptor};

#[cfg(feature = "cuda")]
struct ResidentSearchStartAuthorityV3 {
    runtime: ResidentSearchSlice2ValidatedRuntimeAuthorityV2,
    plan: SealedResidentGenerationPlanV1,
    smc_weights: [f64; 11],
    smc_gate_disabled: bool,
    settings: NeoPopulationSettings,
    scenarios: Box<[ScenarioDescriptor]>,
}

#[cfg(feature = "cuda")]
struct ResidentSearchAuthorityStateV3 {
    session: Option<ResidentTrimmedPopulationSessionV1>,
    calibration: Option<ResidentArchiveKnnCalibrationReceiptV2>,
    native: Option<ResidentSearchSlice2NativeOwnerV3>,
    settings: Option<NeoPopulationSettings>,
    completion: Option<ResidentFeatureStoreConsumerLeaseV3>,
    poisoned: bool,
}

#[cfg(feature = "cuda")]
enum ResidentSearchRejectedTransitionV3 {
    Native(ResidentSearchSlice2NativeErrorV3),
    MissingAuthority,
    TrimLifetime,
}

pub struct ResidentArchiveKnnCalibrationReceiptV2 {
    #[cfg(feature = "cuda")]
    inner: Option<ResidentSearchStartAuthorityV3>,
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
    #[cfg(feature = "cuda")]
    retained_terminal_authority: Option<ResidentSearchAuthorityStateV3>,
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
            let mut state = self.inner;
            if state.poisoned {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchGenerationChainV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            }
            if state.native.is_none() {
                let Some(mut calibration) = state.calibration.take() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                let Some(start) = calibration.inner.take() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                let Some(session) = state.session.as_mut() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                let population = match session.take_population_session_for_slice2_v3() {
                    Ok(population) => population,
                    Err(_) => {
                        state.poisoned = true;
                        return Err(reject_resident_search_transition_v3(
                            ResidentSearchGenerationChainV3 { inner: state },
                            ResidentSearchRejectedTransitionV3::TrimLifetime,
                        ));
                    }
                };
                let mut native = match population.begin_resident_search_slice2_native_v3(
                    start.plan,
                    start.smc_weights,
                    start.smc_gate_disabled,
                    start.runtime.into_native_bind_authority_v2(),
                ) {
                    Ok(native) => native,
                    Err(error) => {
                        state.poisoned = true;
                        return Err(reject_resident_search_transition_v3(
                            ResidentSearchGenerationChainV3 { inner: state },
                            ResidentSearchRejectedTransitionV3::Native(error),
                        ));
                    }
                };
                if let Err(error) = native.upload_resident_scenarios_v3(&start.scenarios) {
                    state.native = Some(native);
                    state.poisoned = true;
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ));
                }
                state.settings = Some(start.settings);
                state.native = Some(native);
            }
            let native = state.native.take().expect("validated Slice2 native owner");
            let settings = state.settings.expect("validated Slice2 settings");
            match native.enqueue_score_and_rank_v3(&settings) {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchRankEnqueuedV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
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
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchGenerationChainV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_terminal_seal_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchTerminalPendingV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
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
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchRankEnqueuedV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_stage_archive_from_rank_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchArchiveStagedV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchRankEnqueuedV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
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
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchArchiveStagedV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_evolve_and_publish_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchGenerationChainV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchArchiveStagedV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
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
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(ResidentSearchTransitionErrorV3 {
                    inner: ResidentSearchRejectedTransitionV3::MissingAuthority,
                    retained_terminal_authority: Some(state),
                });
            };
            match native.try_complete_terminal_v3() {
                Ok(ResidentSearchSlice2NativeTryCompleteV3::NotReady(native)) => {
                    state.native = Some(native);
                    Ok(ResidentSearchTryCompleteV3::NotReady(
                        ResidentSearchTerminalPendingV3 { inner: state },
                    ))
                }
                Ok(ResidentSearchSlice2NativeTryCompleteV3::Complete(native)) => {
                    let population = match native.release_terminal_v3() {
                        Ok(population) => population,
                        Err(rejected) => {
                            let (error, native) = rejected.into_parts_v3();
                            state.native = Some(native);
                            state.poisoned = true;
                            return Err(ResidentSearchTransitionErrorV3 {
                                inner: ResidentSearchRejectedTransitionV3::Native(error),
                                retained_terminal_authority: Some(state),
                            });
                        }
                    };
                    let Some(session) = state.session.take() else {
                        return Err(ResidentSearchTransitionErrorV3 {
                            inner: ResidentSearchRejectedTransitionV3::MissingAuthority,
                            retained_terminal_authority: Some(state),
                        });
                    };
                    match session.complete_resident_search_slice2_v3(population) {
                        Ok(completion) => {
                            state.completion = Some(completion);
                            Ok(ResidentSearchTryCompleteV3::Complete(
                                ResidentSearchTerminalReceiptV3 { inner: state },
                            ))
                        }
                        Err(_) => {
                            state.poisoned = true;
                            Err(ResidentSearchTransitionErrorV3 {
                                inner: ResidentSearchRejectedTransitionV3::TrimLifetime,
                                retained_terminal_authority: Some(state),
                            })
                        }
                    }
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(ResidentSearchTransitionErrorV3 {
                        inner: ResidentSearchRejectedTransitionV3::Native(error),
                        retained_terminal_authority: Some(state),
                    })
                }
            }
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
pub(crate) fn seal_resident_archive_knn_calibration_receipt_v2(
    authority: ResidentSearchSlice2ValidatedRuntimeAuthorityV2,
    plan: SealedResidentGenerationPlanV1,
    smc_weights: [f64; 11],
    smc_gate_disabled: bool,
    settings: NeoPopulationSettings,
    scenarios: Box<[ScenarioDescriptor]>,
) -> ResidentArchiveKnnCalibrationReceiptV2 {
    ResidentArchiveKnnCalibrationReceiptV2 {
        inner: Some(ResidentSearchStartAuthorityV3 {
            runtime: authority,
            plan,
            smc_weights,
            smc_gate_disabled,
            settings,
            scenarios,
        }),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn start_resident_search_slice2_v3(
    session: ResidentTrimmedPopulationSessionV1,
    calibration: ResidentArchiveKnnCalibrationReceiptV2,
) -> ResidentSearchGenerationChainV3 {
    ResidentSearchGenerationChainV3 {
        inner: ResidentSearchAuthorityStateV3 {
            session: Some(session),
            calibration: Some(calibration),
            native: None,
            settings: None,
            completion: None,
            poisoned: false,
        },
    }
}

#[cfg(feature = "cuda")]
fn reject_resident_search_transition_v3<A>(
    authority: A,
    inner: ResidentSearchRejectedTransitionV3,
) -> ResidentSearchRejectedAuthorityV3<A> {
    ResidentSearchRejectedAuthorityV3 {
        error: ResidentSearchTransitionErrorV3 {
            inner,
            retained_terminal_authority: None,
        },
        authority,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_authority_state_is_private_move_only_and_native_wired() {
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
        assert!(source.contains("struct ResidentSearchAuthorityStateV3 {"));
        assert!(source.contains("session: Option<ResidentTrimmedPopulationSessionV1>,"));
        assert!(source.contains("native: Option<ResidentSearchSlice2NativeOwnerV3>,"));
        assert!(source.contains("settings: Option<NeoPopulationSettings>,"));
        assert!(source.contains("pub(crate) fn start_resident_search_slice2_v3("));
        for native_transition in [
            "native.enqueue_score_and_rank_v3(&settings)",
            "native.enqueue_stage_archive_from_rank_v3()",
            "native.enqueue_evolve_and_publish_v3()",
            "native.enqueue_terminal_seal_v3()",
            "native.try_complete_terminal_v3()",
            "native.release_terminal_v3()",
        ] {
            assert!(
                source.contains(native_transition),
                "missing {native_transition}"
            );
        }
        assert!(source.contains("ResidentSearchSlice2NativeTryCompleteV3::NotReady(native)"));
        assert!(!source.contains(concat!(
            "ResidentSearchRejectedTransitionV3::",
            "ScoreAndRank"
        )));
        assert!(!source.contains(concat!(
            "Ok(ResidentSearchTryCompleteV3::",
            "NotReady(self))"
        )));
        assert!(trim_source.contains("pub fn begin_resident_search_slice2_v3("));
    }

    #[test]
    fn slice2_start_requires_and_retains_opaque_calibration_receipt() {
        let source = include_str!("resident_search_slice2_v3.rs").replace("\r\n", "\n");
        let trim_source = include_str!("resident_trim_prefilter_v1.rs").replace("\r\n", "\n");

        let calibration_receipt = concat!(
            "pub ",
            "struct ResidentArchiveKnnCalibrationReceiptV2",
            " {\n    #[cfg(feature = \"cuda\")]\n    inner: Option<ResidentSearchStartAuthorityV3>,"
        );
        let private_start = "pub(crate) fn start_resident_search_slice2_v3(\n    session: ResidentTrimmedPopulationSessionV1,\n    calibration: ResidentArchiveKnnCalibrationReceiptV2,\n) -> ResidentSearchGenerationChainV3";
        let public_start = "pub fn begin_resident_search_slice2_v3(\n        self,\n        calibration: crate::resident_search_slice2_v3::ResidentArchiveKnnCalibrationReceiptV2,\n    ) -> crate::resident_search_slice2_v3::ResidentSearchGenerationChainV3";

        assert!(source.contains(
            "use crate::resident_search_slice2_admission_v2::ResidentSearchSlice2ValidatedRuntimeAuthorityV2;"
        ));
        assert!(source.contains(calibration_receipt));
        assert!(source.contains(private_start));
        assert!(trim_source.contains(public_start));
        for retained in [
            "plan: SealedResidentGenerationPlanV1,",
            "smc_weights: [f64; 11],",
            "smc_gate_disabled: bool,",
            "settings: NeoPopulationSettings,",
            "scenarios: Box<[ScenarioDescriptor]>,",
        ] {
            assert!(
                source.contains(retained),
                "missing retained start authority {retained}"
            );
        }
        assert!(trim_source.contains("start_resident_search_slice2_v3(self, calibration)"));

        assert!(!source.contains(
            "session: ResidentTrimmedPopulationSessionV1,\n) -> ResidentSearchGenerationChainV3"
        ));
        assert!(
            !trim_source.contains("pub fn begin_resident_search_slice2_v3(\n        self,\n    )")
        );
        assert!(source.contains("pub(crate) fn seal_resident_archive_knn_calibration_receipt_v2("));
        assert!(source.contains("authority: ResidentSearchSlice2ValidatedRuntimeAuthorityV2,"));
        for forbidden in [
            concat!("impl ResidentArchiveKnn", "CalibrationReceiptV2"),
            concat!("Clone for ResidentArchiveKnn", "CalibrationReceiptV2"),
            concat!("Copy for ResidentArchiveKnn", "CalibrationReceiptV2"),
            concat!("Default for ResidentArchiveKnn", "CalibrationReceiptV2"),
            concat!("mint_", "calibration"),
            concat!("fixture_", "calibration"),
            concat!("calibration_", "binding(&self)"),
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden fabrication seam: {forbidden}"
            );
        }
    }
}

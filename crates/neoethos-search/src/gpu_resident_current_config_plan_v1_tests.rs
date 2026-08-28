use std::path::PathBuf;

use neoethos_core::Settings;

use super::*;

const REQUIRED_RATE_V1: u64 = 223_106_667;

#[derive(Clone, Copy)]
struct AdmissionFixtureV1 {
    selected_device_ordinal: u32,
    identities: [[u8; 32]; 8],
    measured_rate: u64,
    phase_one_free_bytes: u64,
    allocator_context_reserve_bytes: u64,
    required_workspace_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    full_discovery_reserve_bytes: u64,
}

impl Default for AdmissionFixtureV1 {
    fn default() -> Self {
        Self {
            selected_device_ordinal: 0,
            identities: [
                [0x11; 32], [0x12; 32], [0x13; 32], [0x14; 32], [0x15; 32], [0x22; 32], [0x33; 32],
                [0x44; 32],
            ],
            measured_rate: 300_000_000,
            phase_one_free_bytes: 8_000_000_000,
            allocator_context_reserve_bytes: 1_000_000_000,
            required_workspace_bytes: 2_500_000_000,
            trim_prefilter_reserved_bytes: 300_000_000,
            full_discovery_reserve_bytes: 3_000_000_000,
        }
    }
}

impl AdmissionFixtureV1 {
    fn seal(self) -> CurrentConfigResidentSearchAdmissionFactsV1 {
        CurrentConfigResidentSearchAdmissionFactsV1::test_fixture_v1(
            self.selected_device_ordinal,
            self.identities[0],
            self.identities[1],
            self.identities[2],
            self.identities[3],
            self.identities[4],
            self.identities[5],
            self.identities[6],
            self.identities[7],
            self.measured_rate,
            self.phase_one_free_bytes,
            self.allocator_context_reserve_bytes,
            self.required_workspace_bytes,
            self.trim_prefilter_reserved_bytes,
            self.full_discovery_reserve_bytes,
        )
    }
}

fn repo_config() -> Settings {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("config.yaml");
    Settings::from_yaml(&path).expect("the shipped headless config must pass the production loader")
}

fn config_and_runtime() -> (DiscoveryConfig, GeneticSearchRuntimeOverrides) {
    let settings = repo_config();
    (
        DiscoveryConfig::from_settings(&settings),
        GeneticSearchRuntimeOverrides::from_settings(&settings),
    )
}

fn seal(
    config: &DiscoveryConfig,
    runtime: &GeneticSearchRuntimeOverrides,
    admission: AdmissionFixtureV1,
) -> Result<SealedCurrentConfigResidentSearchPlanV1, CurrentConfigResidentSearchPlanErrorV1> {
    seal_current_config_resident_search_plan_v1(config, runtime, 1_000_000, 500, admission.seal())
}

#[test]
fn shipped_headless_config_seals_exact_current_config_requirements() {
    let (config, runtime) = config_and_runtime();
    let plan = seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap();

    assert_eq!(plan.population(), 200);
    assert_eq!(plan.maximum_generations(), 20_000);
    assert_eq!(plan.maximum_runtime_millis(), 3_600_000);
    assert_eq!(plan.maximum_terms_per_gene(), 16);
    assert_eq!(plan.parent_row_range(), 0..1_000_000);
    assert_eq!(plan.prefilter_fit_row_range(), 0..800_000);
    assert_eq!(plan.outer_holdout_row_range(), 800_000..1_000_000);
    assert_eq!(plan.prefilter_top_k(), 240);
    assert_eq!(plan.prefilter_min_per_timeframe(), 6);
    assert_eq!(plan.immutable_base_scenario_count(), 200);
    assert_eq!(plan.novelty_weight().to_bits(), 0.2_f64.to_bits());
    assert_eq!(plan.novelty_neighbors(), 15);
    assert_eq!(plan.permanent_archive_capacity(), 50_000);
    assert_eq!(plan.archive_min_net().to_bits(), 0.0_f64.to_bits());
    assert_eq!(plan.maximum_archive_knn_distance_count(), 200_796_000_000);
    assert_eq!(plan.gene_signature_word_count(), 4);
    assert_eq!(
        plan.maximum_archive_knn_popcount_word_count(),
        803_184_000_000
    );
    assert_eq!(
        plan.required_archive_knn_popcount_words_per_second(),
        REQUIRED_RATE_V1
    );
    assert!(plan.archive_knn_budget_admitted());
    assert_eq!(plan.trim_prefilter_reserved_bytes(), 300_000_000);
    assert_eq!(plan.required_workspace_bytes(), 2_500_000_000);
    assert_eq!(plan.full_discovery_reserve_bytes(), 3_000_000_000);
    assert_ne!(plan.plan_identity_sha256(), [0; 32]);
}

#[test]
fn novelty_and_calibration_are_explicit_fail_closed_identity_inputs() {
    let (config, runtime) = config_and_runtime();
    let baseline = seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap();

    let mut changed_k = runtime.clone();
    changed_k.novelty_neighbors = 14;
    let changed_k = seal(&config, &changed_k, AdmissionFixtureV1::default()).unwrap();
    assert_ne!(
        baseline.plan_identity_sha256(),
        changed_k.plan_identity_sha256()
    );

    let mut changed_calibration = AdmissionFixtureV1::default();
    changed_calibration.identities[7] = [0x45; 32];
    let changed_calibration = seal(&config, &runtime, changed_calibration).unwrap();
    assert_ne!(
        baseline.plan_identity_sha256(),
        changed_calibration.plan_identity_sha256()
    );

    let mut under_budget = AdmissionFixtureV1::default();
    under_budget.measured_rate = REQUIRED_RATE_V1 - 1;
    assert_eq!(
        seal(&config, &runtime, under_budget).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::ArchiveKnnBudgetExceeded
    );
}

#[test]
fn invalid_novelty_or_admission_fails_before_native_allocation() {
    let (config, baseline) = config_and_runtime();
    for invalid_neighbors in [0, config.population] {
        let mut runtime = baseline.clone();
        runtime.novelty_neighbors = invalid_neighbors;
        assert_eq!(
            seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
            CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyNeighbors
        );
    }
    for invalid_weight in [-0.0, f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut runtime = baseline.clone();
        runtime.novelty_weight = invalid_weight;
        assert_eq!(
            seal(&config, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
            CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyWeight
        );
    }

    let mut invalid_identity = AdmissionFixtureV1::default();
    invalid_identity.identities[3] = [0; 32];
    assert_eq!(
        seal(&config, &baseline, invalid_identity).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InvalidAdmissionFacts
    );
    let invalid_reserve = AdmissionFixtureV1 {
        trim_prefilter_reserved_bytes: 3_000_000_001,
        ..AdmissionFixtureV1::default()
    };
    assert_eq!(
        seal(&config, &baseline, invalid_reserve).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InvalidAdmissionFacts
    );
}

#[test]
fn generic_symbol_or_account_geometry_fails_before_native_allocation() {
    let (baseline, runtime) = config_and_runtime();
    assert_eq!(baseline.evaluation_symbol, "EURUSD");
    assert_eq!(baseline.evaluation_account_currency, "GBP");

    let mut changed_symbol = baseline.clone();
    changed_symbol.evaluation_symbol = "GBPUSD".to_owned();
    assert_eq!(
        seal(&changed_symbol, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics
    );

    let mut changed_account = baseline.clone();
    changed_account.evaluation_account_currency = "USD".to_owned();
    assert_eq!(
        seal(&changed_account, &runtime, AdmissionFixtureV1::default()).unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics
    );
}

#[test]
fn trim_and_archive_arithmetic_refuses_short_or_unrepresentable_runs() {
    let (mut config, runtime) = config_and_runtime();
    assert_eq!(
        seal_current_config_resident_search_plan_v1(
            &config,
            &runtime,
            79,
            500,
            AdmissionFixtureV1::default().seal(),
        )
        .unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::InsufficientRows
    );

    config.generations = usize::MAX;
    assert_eq!(
        seal_current_config_resident_search_plan_v1(
            &config,
            &runtime,
            10_000,
            usize::MAX,
            AdmissionFixtureV1::default().seal(),
        )
        .unwrap_err(),
        CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow
    );
}

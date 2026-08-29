#[path = "support/bayesian_r5.rs"]
mod bayesian_r5;

use bayesian_r5::{
    EvidenceDimensions, GitIdentity, KernelActivity, TimingReceipt, TransferActivity,
    TransferDirection, assert_public_cpu_matches_oracle, fixture_cases, hash_f64_matrix,
    hash_labels, validate_cuda_evidence,
};
use std::time::Duration;

#[test]
fn public_cpu_posterior_and_probabilities_match_independent_oracle_for_all_cases() {
    assert_public_cpu_matches_oracle(&fixture_cases());
}

#[test]
fn semantic_cuda_validator_accepts_all_named_bayesian_stages_with_meaningful_work() {
    let dimensions = EvidenceDimensions {
        train_rows: 1_000_000,
        feature_columns: 64,
        oos_rows: 131_071,
        classes: 3,
    };
    let kernels = [
        KernelActivity::new("neoethos_bayesian_preprocess_f64", 0, 500_000, 4096),
        KernelActivity::new("neoethos_bayesian_map_update_f64", 500_000, 1_500_000, 4096),
        KernelActivity::new("neoethos_bayesian_hessian_f64", 1_500_000, 2_500_000, 4096),
        KernelActivity::new("neoethos_bayesian_cholesky_f64", 2_500_000, 3_500_000, 3),
        KernelActivity::new("neoethos_bayesian_inference_f64", 3_500_000, 4_500_000, 512),
    ];
    let transfers = [
        TransferActivity::new(
            TransferDirection::HostToDevice,
            0,
            1,
            dimensions.minimum_host_to_device_bytes(),
        ),
        TransferActivity::new(
            TransferDirection::DeviceToHost,
            4_500_000,
            4_500_001,
            dimensions.minimum_device_to_host_bytes(),
        ),
    ];

    let evidence = validate_cuda_evidence(
        "bayes_logit_bayesian_ovr_cuda[gpu:0]",
        dimensions,
        &kernels,
        &transfers,
    )
    .expect("all five named Bayesian stages and meaningful transfers must pass");

    assert_eq!(evidence.named_stage_count, 5);
    assert!(evidence.total_kernel_duration_ns >= 4_500_000);
    assert!(evidence.total_grid_blocks >= dimensions.minimum_grid_blocks());
}

#[test]
fn generic_decoy_kernel_plus_cpu_result_is_rejected_with_every_missing_stage() {
    let dimensions = EvidenceDimensions {
        train_rows: 1_000_000,
        feature_columns: 128,
        oos_rows: 131_071,
        classes: 3,
    };
    let kernels = [KernelActivity::new("vector_add_decoy", 0, 5_000_000, 1)];
    let transfers = [TransferActivity::new(
        TransferDirection::HostToDevice,
        0,
        1,
        8,
    )];

    let errors = validate_cuda_evidence(
        "bayes_logit_bayesian_ovr_cpu",
        dimensions,
        &kernels,
        &transfers,
    )
    .expect_err("a decoy CUDA launch around CPU work must fail");
    let joined = errors.join("\n");
    for required in [
        "native CUDA backend",
        "preprocessing",
        "MAP update",
        "Hessian",
        "Cholesky",
        "inference",
        "host-to-device bytes",
        "device-to-host bytes",
        "meaningful grid work",
    ] {
        assert!(
            joined.contains(required),
            "decoy rejection did not report `{required}`:\n{joined}"
        );
    }
}

#[test]
fn one_disguised_kernel_cannot_impersonate_five_bayesian_stages() {
    let dimensions = EvidenceDimensions {
        train_rows: 1_000_000,
        feature_columns: 64,
        oos_rows: 131_071,
        classes: 3,
    };
    let kernels = [KernelActivity::new(
        "neoethos_bayesian_preprocess_map_update_hessian_cholesky_inference_decoy",
        0,
        5_000_000,
        dimensions.minimum_grid_blocks(),
    )];
    let transfers = [
        TransferActivity::new(
            TransferDirection::HostToDevice,
            0,
            1,
            dimensions.minimum_host_to_device_bytes(),
        ),
        TransferActivity::new(
            TransferDirection::DeviceToHost,
            5_000_000,
            5_000_001,
            dimensions.minimum_device_to_host_bytes(),
        ),
    ];

    let errors = validate_cuda_evidence(
        "bayes_logit_bayesian_ovr_cuda[gpu:0]",
        dimensions,
        &kernels,
        &transfers,
    )
    .expect_err("one disguised launch may satisfy at most one semantic stage");
    let joined = errors.join("\n");
    for missing in ["MAP update", "Hessian", "Cholesky", "inference"] {
        assert!(
            joined.contains(missing),
            "disguised one-kernel rejection omitted {missing}:\n{joined}"
        );
    }
}

#[test]
fn fixture_and_label_hashes_bind_shape_order_and_f64_bits() {
    let cases = fixture_cases();
    for case in &cases {
        let train_hash = hash_f64_matrix(&case.train_features);
        let oos_hash = hash_f64_matrix(&case.oos_features);
        let label_hash = hash_labels(&case.train_labels);
        assert_ne!(train_hash, oos_hash, "{} train/OOS collision", case.name);
        assert_eq!(train_hash.len(), 64);
        assert_eq!(oos_hash.len(), 64);
        assert_eq!(label_hash.len(), 64);
    }

    let mut bit_drift = cases[0].train_features.clone();
    bit_drift[(0, 0)] = f64::from_bits(bit_drift[(0, 0)].to_bits() ^ 1);
    assert_ne!(
        hash_f64_matrix(&cases[0].train_features),
        hash_f64_matrix(&bit_drift),
        "one f64 bit must change the fixture identity"
    );
}

#[test]
fn timing_receipt_preserves_every_raw_duration_and_rejects_wrong_sample_count() {
    let receipt = TimingReceipt::new(
        Duration::from_nanos(17),
        [
            Duration::from_nanos(101),
            Duration::from_nanos(97),
            Duration::from_nanos(103),
        ],
    )
    .expect("one warm-up and exactly three samples are valid");
    assert_eq!(receipt.warmup_ns, 17);
    assert_eq!(receipt.raw_sample_ns, vec![101, 97, 103]);
    assert_eq!(receipt.median_ns, 101);

    let errors = TimingReceipt::from_slice(
        Duration::from_nanos(1),
        &[Duration::from_nanos(2), Duration::from_nanos(3)],
    )
    .expect_err("two timed samples must not satisfy the bounded contract");
    assert!(errors.contains("exactly three"));
}

#[test]
fn dynamic_git_identity_rejects_dirty_or_non_object_output() {
    let identity = GitIdentity::parse(
        "0123456789abcdef0123456789abcdef01234567\n",
        "89abcdef0123456789abcdef0123456789abcdef\n",
        "",
    )
    .expect("clean live Git outputs are valid");
    assert_eq!(identity.commit, "0123456789abcdef0123456789abcdef01234567");
    assert_eq!(identity.tree, "89abcdef0123456789abcdef0123456789abcdef");

    assert!(
        GitIdentity::parse(
            "0123456789abcdef0123456789abcdef01234567",
            "89abcdef0123456789abcdef0123456789abcdef",
            " M crates/neoethos-models/src/statistical/bayesian_impl.rs\n",
        )
        .expect_err("dirty implementation trees are forbidden")
        .contains("dirty")
    );
    assert!(
        GitIdentity::parse("preimplementation-authority", "tree", "")
            .expect_err("hard-coded labels are not Git object identities")
            .contains("hex")
    );
}

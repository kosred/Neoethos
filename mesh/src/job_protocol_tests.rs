use super::{FedJob, MeshReq, MeshResp, validate_job};
use serde_json::json;

fn received_job(value: serde_json::Value) -> FedJob {
    let response = serde_json::to_vec(&json!({ "Job": value })).unwrap();
    match serde_json::from_slice::<MeshResp>(&response).unwrap() {
        MeshResp::Job(Some(job)) => job,
        other => panic!("expected a received job, got {other:?}"),
    }
}

#[test]
fn legacy_remote_training_archive_is_not_a_request_variant() {
    // The same decoder used by MeshProto::accept rejects the request tag. The
    // payload is deliberately not an archive or valid base64; neither is decoded.
    let request = serde_json::to_vec(&json!({
        "SubmitTraining": {
            "symbol": "EURUSD",
            "base_tf": "M5",
            "model_tar_b64": "not base64 and not an archive"
        }
    }))
    .unwrap();
    let error = serde_json::from_slice::<MeshReq>(&request).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unknown variant `SubmitTraining`"),
        "{error}"
    );
}

#[test]
fn canonical_and_legacy_training_job_fields_are_preserved_then_refused() {
    for key in ["workType", "work_type"] {
        let mut value = json!({ "symbol": "EURUSD", "baseTf": "M5" });
        value[key] = json!("training");
        let job = received_job(value);
        assert_eq!(
            job.work_type, "training",
            "{key} must not default to discovery"
        );
        let error = validate_job(&job).unwrap_err().to_string();
        assert!(error.contains("training jobs are unsupported"), "{error}");
        assert!(
            error.contains("exact candidate training handoff"),
            "{error}"
        );
    }
}

#[test]
fn discovery_job_roundtrip_uses_the_app_canonical_work_type_field() {
    for key in ["workType", "work_type"] {
        let mut value = json!({ "symbol": "EURUSD", "baseTf": "M5" });
        value[key] = json!("discovery");
        let job = received_job(value);
        validate_job(&job).unwrap();
        let serialized = serde_json::to_value(&job).unwrap();
        assert_eq!(
            serialized,
            json!({
                "symbol": "EURUSD", "baseTf": "M5", "workType": "discovery"
            })
        );
        assert!(serialized.get("work_type").is_none());
    }
}

#[test]
fn omitted_legacy_work_type_remains_discovery() {
    let job = received_job(json!({ "symbol": "EURUSD", "baseTf": "M5" }));
    assert_eq!(job.work_type, "discovery");
    validate_job(&job).unwrap();
}

#[test]
fn conflicting_or_duplicate_work_type_aliases_are_rejected() {
    for legacy in ["discovery", "training"] {
        let wire = serde_json::to_vec(&json!({ "Job": {
            "symbol": "EURUSD", "baseTf": "M5",
            "workType": "discovery", "work_type": legacy
        } }))
        .unwrap();
        let error = serde_json::from_slice::<MeshResp>(&wire).unwrap_err();
        assert!(
            error.to_string().contains("duplicate field `workType`"),
            "{error}"
        );
    }
}

#[test]
fn unknown_job_fields_or_work_types_never_silently_become_discovery() {
    for key in ["worktype", "work_typ", "model_tar_b64", "training_handoff"] {
        let mut job = json!({ "symbol": "EURUSD", "baseTf": "M5" });
        job[key] = json!("training");
        let wire = serde_json::to_vec(&json!({ "Job": job })).unwrap();
        let error = serde_json::from_slice::<MeshResp>(&wire).unwrap_err();
        assert!(
            error.to_string().contains("unknown field"),
            "{key}: {error}"
        );
    }
    for work_type in [
        "",
        "Training",
        "DISCOVERY",
        "train",
        "discovery/training",
        "other",
    ] {
        let job = received_job(json!({
            "symbol": "EURUSD", "baseTf": "M5", "workType": work_type
        }));
        assert_eq!(job.work_type, work_type);
        let error = validate_job(&job).unwrap_err();
        assert!(
            error.to_string().contains("unknown workType"),
            "{work_type}: {error}"
        );
    }
}

#[test]
fn unsafe_received_symbol_or_timeframe_is_refused_before_local_work() {
    let too_long = "A".repeat(65);
    for unsafe_component in [
        "",
        "..",
        ".",
        "../EURUSD",
        "EURUSD/M5",
        "EURUSD\\M5",
        "C:",
        "C:\\models",
        "EUR USD",
        "EURUSD\n",
        "ΕURUSD",
        too_long.as_str(),
    ] {
        for key in ["symbol", "baseTf"] {
            let mut value = json!({
                "symbol": "EURUSD", "baseTf": "M5", "workType": "discovery"
            });
            value[key] = json!(unsafe_component);
            let job = received_job(value);
            let error = validate_job(&job).unwrap_err();
            assert!(
                error.to_string().contains("unsafe symbol/baseTf"),
                "{key}: {error}"
            );
        }
    }
}

#[test]
fn safe_components_do_not_duplicate_the_apps_canonical_dataset_admission() {
    for symbol in ["EURUSD", "eurusd", "EUR_USD", "EURUSD-demo"] {
        let job = received_job(json!({
            "symbol": symbol, "baseTf": "M5", "workType": "discovery"
        }));
        validate_job(&job).unwrap();
    }
}

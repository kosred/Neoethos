use std::cell::Cell;

use neoethos_autoresearch::journal::OosWindow;
use neoethos_autoresearch::runner::{
    OosEvidence, PromotionPortfolio, RunArgs, SearchOutcome, SearchRequest, SweepExecutor,
    run_with_executor,
};
use neoethos_autoresearch::session::SweepId;
use neoethos_core::config::Settings;
use neoethos_search::DiscoveryConfig;

#[derive(Default)]
struct InvocationProbe {
    calls: Cell<usize>,
}

impl InvocationProbe {
    fn called<T>(&self, method: &str) -> anyhow::Result<T> {
        self.calls.set(self.calls.get() + 1);
        anyhow::bail!("broker gate invoked executor method {method}")
    }
}

impl SweepExecutor for InvocationProbe {
    fn describe(&self) -> String {
        self.calls.set(self.calls.get() + 1);
        "broker-gate invocation probe".to_string()
    }

    fn streaming_requested(&self) -> bool {
        self.calls.set(self.calls.get() + 1);
        false
    }

    fn windows(&self) -> anyhow::Result<((i64, i64), OosWindow, usize, f64)> {
        self.called("windows")
    }

    fn execute(&mut self, _request: &SearchRequest<'_>) -> anyhow::Result<SearchOutcome> {
        self.called("execute")
    }

    fn oos_preflight(&self, _portfolio: &PromotionPortfolio) -> anyhow::Result<()> {
        self.called("oos_preflight")
    }

    fn evaluate_oos(
        &mut self,
        _sweep: SweepId,
        _slot: usize,
        _config: &DiscoveryConfig,
        _portfolio: &PromotionPortfolio,
    ) -> anyhow::Result<OosEvidence> {
        self.called("evaluate_oos")
    }
}

#[test]
fn public_runner_refuses_before_executor_or_artifact_work() {
    let root = std::env::temp_dir().join(format!(
        "neoethos-autoresearch-broker-gate-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create isolated autoresearch root");

    // SAFETY: this integration binary contains one test and sets the process
    // variable before the runner can create a thread or access the store.
    unsafe {
        std::env::set_var("NEOETHOS_USER_DATA_DIR", &root);
    }

    let settings = Settings::default();
    let mut executor = InvocationProbe::default();
    let result = run_with_executor(
        RunArgs::new("EURUSD"),
        &settings,
        DiscoveryConfig::default(),
        &mut executor,
    );

    let error = result.expect_err("autoresearch ran without broker financial truth");
    let message = format!("{error:#}");
    assert!(
        message.contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1")
            && message.contains("operation=historical_evaluation"),
        "unexpected refusal: {message}"
    );
    assert_eq!(
        executor.calls.get(),
        0,
        "the broker gate must precede every executor method"
    );
    assert_eq!(
        std::fs::read_dir(&root)
            .expect("read isolated autoresearch root")
            .count(),
        0,
        "the broker gate must precede session or artifact publication"
    );
}

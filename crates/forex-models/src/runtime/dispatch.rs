use anyhow::{Result, bail};

use super::capabilities::{CapabilityState, ModelFamily};
use crate::registry::get_model_capability;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchEntry {
    pub name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchPlan {
    pub entries: Vec<DispatchEntry>,
}

pub fn build_dispatch_plan(model_names: &[String]) -> Result<DispatchPlan> {
    let mut names: Vec<String> = model_names
        .iter()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect();
    names.sort();
    names.dedup();

    if names.is_empty() {
        bail!("no model names provided for dispatch planning");
    }

    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let capability = get_model_capability(&name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' is not registered with a capability"))?;
        entries.push(DispatchEntry {
            name: capability.name,
            family: capability.family,
            state: capability.state,
        });
    }

    Ok(DispatchPlan { entries })
}

#[cfg(test)]
mod tests {
    use super::super::capabilities::{CapabilityState, ModelFamily};
    use super::build_dispatch_plan;

    #[test]
    fn dispatch_plan_is_deterministic_and_carries_runtime_metadata() {
        let models = vec![
            "mlp".to_string(),
            "lightgbm".to_string(),
            "patchtst".to_string(),
            "lightgbm".to_string(),
        ];

        let plan = build_dispatch_plan(&models).expect("dispatch plan should build");
        let names: Vec<_> = plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();

        assert_eq!(names, vec!["lightgbm", "mlp", "patchtst"]);
        assert_eq!(plan.entries[0].family, ModelFamily::Tree);
        assert_eq!(plan.entries[0].state, CapabilityState::Implemented);
        assert_eq!(plan.entries[1].family, ModelFamily::Deep);
        assert_eq!(plan.entries[1].state, CapabilityState::Implemented);
        assert_eq!(plan.entries[2].family, ModelFamily::Deep);
        assert_eq!(plan.entries[2].state, CapabilityState::Implemented);
    }

    #[test]
    fn dispatch_plan_rejects_empty_model_list() {
        let err = build_dispatch_plan(&[]).expect_err("empty model list should fail");
        assert!(err.to_string().contains("no model names"));
    }
}

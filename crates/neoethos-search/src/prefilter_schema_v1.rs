//! Canonical host-side schema classification shared by the CPU prefilter and
//! the one compact metadata upload for the resident CUDA prefilter.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) const PREFILTER_STATE_FAMILIES_V1: [&str; 4] = ["regime_", "smc_", "session_", "fp_"];
pub(crate) const COLUMN_CLASS_STATE_V1: u8 = 1 << 0;
pub(crate) const COLUMN_CLASS_TEMPLATE_V1: u8 = 1 << 1;

const PREFILTER_SCHEMA_CLASSIFICATION_SEMANTICS_V1: &str = concat!(
    "neoethos.prefilter-schema-classification.v1;",
    "base-state-prefixes-regime-smc-session-fp;",
    "timeframe-head-M-H-D-W-MN-plus-digits-length2or3;",
    "timeframe-group-id-first-ordered-schema-occurrence-one-based;",
    "template-force-keep-seed-template-role-resolution;",
    "ordered-feature-name-length-prefix-sha256"
);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SealedPrefilterColumnClassificationV1 {
    column_class_flags: Vec<u8>,
    timeframe_group_ids: Vec<u32>,
    template_force_keep_flags: Vec<u8>,
    timeframe_group_count: u64,
    ordered_feature_schema_sha256: [u8; 32],
    column_classification_content_sha256: [u8; 32],
}

impl SealedPrefilterColumnClassificationV1 {
    pub(crate) fn column_class_flags(&self) -> &[u8] {
        &self.column_class_flags
    }

    pub(crate) fn timeframe_group_ids(&self) -> &[u32] {
        &self.timeframe_group_ids
    }

    pub(crate) fn template_force_keep_flags(&self) -> &[u8] {
        &self.template_force_keep_flags
    }

    pub(crate) const fn timeframe_group_count(&self) -> u64 {
        self.timeframe_group_count
    }

    pub(crate) const fn ordered_feature_schema_sha256(&self) -> [u8; 32] {
        self.ordered_feature_schema_sha256
    }

    pub(crate) const fn column_classification_content_sha256(&self) -> [u8; 32] {
        self.column_classification_content_sha256
    }
}

pub(crate) fn is_prefilter_state_column_v1(name: &str) -> bool {
    PREFILTER_STATE_FAMILIES_V1
        .iter()
        .any(|family| name.starts_with(family))
}

pub(crate) fn timeframe_group_v1(name: &str) -> Option<&str> {
    let head = name.split('_').next()?;
    if head.len() < 2 || head.len() > 3 {
        return None;
    }
    let digits = if let Some(rest) = head.strip_prefix("MN") {
        rest
    } else if head.starts_with(['M', 'H', 'D', 'W']) {
        &head[1..]
    } else {
        return None;
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(head)
}

pub(crate) fn seal_prefilter_column_classification_v1(
    ordered_feature_names: &[String],
) -> Option<SealedPrefilterColumnClassificationV1> {
    if ordered_feature_names.is_empty() {
        return None;
    }
    let mut group_ids = BTreeMap::<String, u32>::new();
    let mut next_group_id = 1_u32;
    let mut column_class_flags = vec![0_u8; ordered_feature_names.len()];
    let mut timeframe_group_ids = vec![0_u32; ordered_feature_names.len()];
    let mut template_force_keep_flags = vec![0_u8; ordered_feature_names.len()];

    for (column, name) in ordered_feature_names.iter().enumerate() {
        if is_prefilter_state_column_v1(name) {
            column_class_flags[column] |= COLUMN_CLASS_STATE_V1;
        }
        if let Some(group) = timeframe_group_v1(name) {
            let group_id = if let Some(group_id) = group_ids.get(group) {
                *group_id
            } else {
                let group_id = next_group_id;
                next_group_id = next_group_id.checked_add(1)?;
                group_ids.insert(group.to_owned(), group_id);
                group_id
            };
            timeframe_group_ids[column] = group_id;
        }
    }
    for column in crate::genetic::seed_templates::template_feature_indices(ordered_feature_names) {
        let class = column_class_flags.get_mut(column)?;
        let force_keep = template_force_keep_flags.get_mut(column)?;
        *class |= COLUMN_CLASS_TEMPLATE_V1;
        *force_keep = 1;
    }

    let ordered_feature_schema_sha256 = hash_ordered_feature_schema_v1(ordered_feature_names);
    let mut hasher = Sha256::new();
    hasher.update(PREFILTER_SCHEMA_CLASSIFICATION_SEMANTICS_V1.as_bytes());
    hasher.update(ordered_feature_schema_sha256);
    hasher.update((ordered_feature_names.len() as u64).to_le_bytes());
    hasher.update((group_ids.len() as u64).to_le_bytes());
    hasher.update(&column_class_flags);
    for group_id in &timeframe_group_ids {
        hasher.update(group_id.to_le_bytes());
    }
    hasher.update(&template_force_keep_flags);
    let column_classification_content_sha256 = hasher.finalize().into();

    Some(SealedPrefilterColumnClassificationV1 {
        column_class_flags,
        timeframe_group_ids,
        template_force_keep_flags,
        timeframe_group_count: group_ids.len() as u64,
        ordered_feature_schema_sha256,
        column_classification_content_sha256,
    })
}

fn hash_ordered_feature_schema_v1(ordered_feature_names: &[String]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.ordered-prefilter-feature-schema.v1");
    hasher.update((ordered_feature_names.len() as u64).to_le_bytes());
    for name in ordered_feature_names {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_reuses_state_timeframe_and_template_authorities() {
        let names = [
            "regime_vol_state",
            "H1_smc_ob",
            "M15_rsi",
            "h4_macd",
            "unrelated",
        ]
        .map(str::to_owned);
        let sealed = seal_prefilter_column_classification_v1(&names).expect("valid schema");
        assert_eq!(sealed.column_class_flags()[0] & COLUMN_CLASS_STATE_V1, 1);
        assert_eq!(sealed.column_class_flags()[1] & COLUMN_CLASS_STATE_V1, 0);
        assert_eq!(sealed.timeframe_group_ids()[0], 0);
        assert_eq!(sealed.timeframe_group_ids()[1], 1);
        assert_eq!(sealed.timeframe_group_ids()[2], 2);
        assert!(sealed.timeframe_group_count() >= 2);
        assert_eq!(sealed.column_class_flags().len(), names.len());
        assert_eq!(sealed.template_force_keep_flags().len(), names.len());
        assert_ne!(sealed.ordered_feature_schema_sha256(), [0; 32]);
        assert_ne!(sealed.column_classification_content_sha256(), [0; 32]);
    }
}

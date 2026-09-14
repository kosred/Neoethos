//! Editable form state for Discover / Train pages.
//!
//! A form is a list of typed fields the operator can navigate (Up/Down),
//! activate (Enter to start editing), edit (type / Backspace), commit
//! (Enter again), and cancel (Esc). Each field stores its value as a
//! `String` for keyboard editability — page code converts to int / Vec
//! at launch time.
//!
//! The Symbols page inventories exact manifest-backed dataset identities;
//! this module does not maintain a second filesystem scanner.

#[derive(Debug, Clone)]
pub struct Field {
    /// Short label shown to the operator. Always upper-case to fit the
    /// dense Bloomberg-style aesthetic.
    pub label: &'static str,
    /// Free-text value. Page launch handlers validate numeric fields and
    /// report invalid input instead of silently substituting a default.
    pub value: String,
    /// Default value used when `value` is empty, not when it is invalid.
    pub default_value: String,
    /// Hint shown beneath the value in muted text.
    ///
    /// Owned, not `&'static str`: a hint that has to REPORT something — which
    /// of two knobs is in force, what the on-disk value maps to — cannot be a
    /// literal. Making it owned is what lets the Config page name the twin
    /// that beats the field the operator is looking at.
    pub hint: String,
}

impl Field {
    pub fn new(
        label: &'static str,
        default_value: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        let default_value: String = default_value.into();
        Self {
            label,
            value: default_value.clone(),
            default_value,
            hint: hint.into(),
        }
    }

    /// Read the value, falling back to the default if blank.
    pub fn effective(&self) -> &str {
        if self.value.trim().is_empty() {
            &self.default_value
        } else {
            &self.value
        }
    }
}

#[derive(Debug, Default)]
pub struct FormState {
    pub fields: Vec<Field>,
    /// Index of the currently focused field. Wraps on
    /// `focus_next` / `focus_prev`.
    pub focused: usize,
    /// True when the operator has hit Enter on a field — keystrokes
    /// modify `fields[focused].value`.
    pub editing: bool,
    /// Last status / validation message. Cleared when the operator
    /// switches focus.
    pub message: Option<String>,
    /// Exact value before the current edit, including an intentionally blank
    /// override. Esc restores this value, not a factory default.
    edit_snapshot: Option<(usize, String)>,
}

impl FormState {
    pub fn new(fields: Vec<Field>) -> Self {
        Self {
            fields,
            focused: 0,
            editing: false,
            message: None,
            edit_snapshot: None,
        }
    }

    pub fn focus_next(&mut self) {
        if self.fields.is_empty() {
            return;
        }
        self.stop_editing(true);
        self.focused = (self.focused + 1) % self.fields.len();
        self.message = None;
    }

    pub fn focus_prev(&mut self) {
        if self.fields.is_empty() {
            return;
        }
        self.stop_editing(true);
        self.focused = (self.focused + self.fields.len() - 1) % self.fields.len();
        self.message = None;
    }

    pub fn focus(&mut self, idx: usize) {
        if idx < self.fields.len() {
            self.stop_editing(true);
            self.focused = idx;
            self.message = None;
        }
    }

    pub fn start_editing(&mut self) {
        if !self.editing && self.focused < self.fields.len() {
            self.edit_snapshot = Some((self.focused, self.fields[self.focused].value.clone()));
            self.editing = true;
        }
    }

    pub fn stop_editing(&mut self, commit: bool) {
        if let Some((index, original)) = self.edit_snapshot.take() {
            if !commit {
                if let Some(field) = self.fields.get_mut(index) {
                    field.value = original;
                }
            }
        }
        self.editing = false;
    }

    pub fn type_char(&mut self, c: char) {
        if !self.editing {
            return;
        }
        if let Some(field) = self.fields.get_mut(self.focused) {
            field.value.push(c);
        }
    }

    pub fn backspace(&mut self) {
        if !self.editing {
            return;
        }
        if let Some(field) = self.fields.get_mut(self.focused) {
            field.value.pop();
        }
    }

    pub fn clear_focused(&mut self) {
        if let Some(field) = self.fields.get_mut(self.focused) {
            field.value.clear();
        }
    }

    pub fn value_for(&self, label: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.label == label)
            .map(|f| f.effective())
    }

    /// Set initial output directories from the same immutable startup settings
    /// as the data root. Later operator edits remain ordinary form overrides.
    pub fn with_cache_defaults(mut self, cache_root: &std::path::Path) -> Self {
        for field in &mut self.fields {
            let child = match field.label {
                "Out dir" => "discovery",
                "Models dir" => "models",
                _ => continue,
            };
            let value = cache_root.join(child).to_string_lossy().into_owned();
            field.default_value = value.clone();
            field.value = value;
        }
        self
    }
}

// ─── Discover form ─────────────────────────────────────────────────────

pub fn make_discover_form(default_root: &str) -> FormState {
    FormState::new(vec![
        Field::new(
            "Symbols",
            "",
            "Comma-separated. Empty = auto-detect from data root.",
        ),
        Field::new(
            "Timeframes",
            "M30,H1,H4,D1",
            "Comma-separated. Default: M30,H1,H4,D1",
        ),
        Field::new(
            "Population",
            "",
            "Positive override. Blank inherits configured population/adaptive budget.",
        ),
        Field::new(
            "Population auto",
            "",
            "true/false. Blank inherits the configured setting.",
        ),
        Field::new(
            "Generations",
            "",
            "Positive override. Blank inherits configured generations.",
        ),
        Field::new(
            "Portfolio size",
            "",
            "Positive override. Blank inherits configured portfolio size.",
        ),
        Field::new(
            "Data root",
            default_root,
            "Root containing canonical manifest-backed Vortex generations",
        ),
        Field::new(
            "Out dir",
            "cache/discovery",
            "Where portfolio JSONs are written",
        ),
    ])
}

// ─── Train form ────────────────────────────────────────────────────────

pub fn make_train_form(default_root: &str) -> FormState {
    // **F-641 / F-CORE2 closure (2026-05-25)**: the Symbol field used
    // to default to "EURUSD" — a synthetic per-symbol assumption that
    // the no-synthetic-data directive forbids. Now empty so the
    // operator has to pick from the Symbols page (which enumerates
    // what's actually on disk via `collect_inventory`). The TUI form's
    // submit handler rejects empty Symbol with a clear validation
    // error rather than silently training on the default.
    FormState::new(vec![
        Field::new(
            "Symbol",
            "",
            "Single symbol to train. Pick from Symbols page (required).",
        ),
        Field::new("Base TF", "M30", "Base timeframe. Default: M30"),
        Field::new("Data root", default_root, "Path to data/ directory"),
        Field::new(
            "Models dir",
            "cache/models",
            "Where trained model artifacts are written",
        ),
    ])
}

// `discover_symbols_in_root` was a half-wired Symbol-field browser. The
// Symbols page now owns the single manifest-only identity inventory, so a
// parallel directory scanner would violate the canonical runtime contract.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_cache_defaults_support_blank_fallback_and_operator_override() {
        let cache = std::path::Path::new("owned-cache");
        for (mut form, label, child) in [
            (
                make_discover_form("owned-data").with_cache_defaults(cache),
                "Out dir",
                "discovery",
            ),
            (
                make_train_form("owned-data").with_cache_defaults(cache),
                "Models dir",
                "models",
            ),
        ] {
            let expected = cache.join(child).to_string_lossy().into_owned();
            assert_eq!(form.value_for(label), Some(expected.as_str()));
            let field = form
                .fields
                .iter_mut()
                .find(|field| field.label == label)
                .unwrap();
            field.value.clear();
            assert_eq!(field.effective(), expected);
            field.value = "operator-output".to_owned();
            assert_eq!(field.effective(), "operator-output");
        }
    }

    #[test]
    fn cancel_restores_the_exact_previous_value_not_the_default() {
        let mut form = FormState::new(vec![Field::new("Symbol", "EURUSD", "")]);
        form.fields[0].value = "δοκιμή".to_string();
        form.start_editing();
        form.backspace();
        form.type_char('X');
        form.start_editing(); // A repeated mouse click must not replace the snapshot.
        form.stop_editing(false);
        assert_eq!(form.fields[0].value, "δοκιμή");
        assert!(!form.editing);
    }

    #[test]
    fn cancel_preserves_a_blank_override() {
        let mut form = make_discover_form("data");
        form.focus(2);
        form.start_editing();
        form.type_char('4');
        form.stop_editing(false);
        assert_eq!(form.value_for("Population"), Some(""));
    }

    #[test]
    fn commit_and_focus_change_discard_the_old_snapshot() {
        let mut form = FormState::new(vec![
            Field::new("First", "1", ""),
            Field::new("Second", "2", ""),
        ]);
        form.start_editing();
        form.type_char('0');
        form.stop_editing(true);
        form.stop_editing(false); // No active edit: do not undo the accepted value.
        assert_eq!(form.fields[0].value, "10");
        form.start_editing();
        form.type_char('1');
        form.focus_next(); // Existing blur behavior commits the current field.
        form.start_editing();
        form.type_char('9');
        form.stop_editing(false);
        assert_eq!(form.fields[0].value, "101");
        assert_eq!(form.fields[1].value, "2");
    }

    #[test]
    fn discovery_budget_fields_inherit_settings_until_explicitly_overridden() {
        let form = make_discover_form("data");
        for label in [
            "Population",
            "Population auto",
            "Generations",
            "Portfolio size",
        ] {
            assert_eq!(form.value_for(label), Some(""), "{label}");
        }
    }
}

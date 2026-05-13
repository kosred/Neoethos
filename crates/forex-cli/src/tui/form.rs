//! Editable form state for Discover / Train pages.
//!
//! A form is a list of typed fields the operator can navigate (Up/Down),
//! activate (Enter to start editing), edit (type / Backspace), commit
//! (Enter again), and cancel (Esc). Each field stores its value as a
//! `String` for keyboard editability — page code converts to int / Vec
//! at launch time.
//!
//! There is also a special "browse" action that, when invoked on the
//! data-root field, scans the current value's directory and offers the
//! list of `symbol=*/` children as picker rows.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Field {
    /// Short label shown to the operator. Always upper-case to fit the
    /// dense Bloomberg-style aesthetic.
    pub label: &'static str,
    /// Free-text value. Numeric fields use parse-on-launch; bad values
    /// fall back to the default in `default_value` and surface in the
    /// status line.
    pub value: String,
    /// Default value used when `value` is empty or invalid.
    pub default_value: String,
    /// Hint shown beneath the value in muted text.
    pub hint: &'static str,
}

impl Field {
    pub fn new(label: &'static str, default_value: impl Into<String>, hint: &'static str) -> Self {
        let default_value: String = default_value.into();
        Self {
            label,
            value: default_value.clone(),
            default_value,
            hint,
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
}

impl FormState {
    pub fn new(fields: Vec<Field>) -> Self {
        Self {
            fields,
            focused: 0,
            editing: false,
            message: None,
        }
    }

    pub fn focus_next(&mut self) {
        if self.fields.is_empty() {
            return;
        }
        self.editing = false;
        self.focused = (self.focused + 1) % self.fields.len();
        self.message = None;
    }

    pub fn focus_prev(&mut self) {
        if self.fields.is_empty() {
            return;
        }
        self.editing = false;
        self.focused = (self.focused + self.fields.len() - 1) % self.fields.len();
        self.message = None;
    }

    pub fn focus(&mut self, idx: usize) {
        if idx < self.fields.len() {
            self.editing = false;
            self.focused = idx;
            self.message = None;
        }
    }

    pub fn start_editing(&mut self) {
        if self.focused < self.fields.len() {
            self.editing = true;
        }
    }

    pub fn stop_editing(&mut self, commit: bool) {
        if !commit {
            // Esc — restore the field to its default if the operator
            // had cleared it; otherwise leave the partially-typed value
            // alone. We don't snapshot a "before edit" value because
            // operators usually want to keep what they typed.
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

    pub fn get(&self, idx: usize) -> Option<&Field> {
        self.fields.get(idx)
    }

    pub fn value_for(&self, label: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.label == label)
            .map(|f| f.effective())
    }
}

// ─── Discover form ─────────────────────────────────────────────────────

pub fn make_discover_form(default_root: &str) -> FormState {
    FormState::new(vec![
        Field::new("Symbols", "", "Comma-separated. Empty = auto-detect from data root."),
        Field::new("Timeframes", "M30,H1,H4,D1", "Comma-separated. Default: M30,H1,H4,D1"),
        Field::new("Population", "1000", "GA population per generation. Default: 1000"),
        Field::new("Generations", "10", "GA generations. Default: 10"),
        Field::new("Portfolio size", "2000", "Max portfolio size per work-unit. Default: 2000"),
        Field::new("Data root", default_root, "Path to data/ directory containing symbol=*/timeframe=*/"),
        Field::new("Out dir", "cache/discovery", "Where portfolio JSONs are written"),
    ])
}

// ─── Train form ────────────────────────────────────────────────────────

pub fn make_train_form(default_root: &str) -> FormState {
    FormState::new(vec![
        Field::new("Symbol", "EURUSD", "Single symbol to train. Pick from Symbols page."),
        Field::new("Base TF", "M30", "Base timeframe. Default: M30"),
        Field::new("Data root", default_root, "Path to data/ directory"),
        Field::new("Models dir", "cache/models", "Where trained model artifacts are written"),
    ])
}

// ─── Symbol-from-data-root helper (used by Symbol field's browse) ──────

/// Scan a data root and return the list of `symbol=*` directory names
/// (with the prefix stripped). Empty if path missing or unreadable.
pub fn discover_symbols_in_root(data_root: &str) -> Vec<String> {
    let p = PathBuf::from(data_root);
    let mut out = Vec::new();
    let Ok(read) = std::fs::read_dir(&p) else {
        return out;
    };
    for entry in read.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if let Some(rest) = s.strip_prefix("symbol=") {
            out.push(rest.to_string());
        }
    }
    out.sort();
    out
}

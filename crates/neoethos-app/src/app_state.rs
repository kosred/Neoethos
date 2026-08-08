use neoethos_core::Settings;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppRuntimeConfig {
    pub config_path: String,
    pub data_dir: PathBuf,
    pub start_local: bool,
    /// Auto-start discovery on headless launch (VPS/WSL2 use-case).
    /// The UI start/stop controls are one of several interfaces to this subsystem.
    pub auto_discovery: bool,
    /// Auto-start training on headless launch (VPS/WSL2 use-case).
    pub auto_training: bool,
}

impl AppRuntimeConfig {
    pub fn from_settings(
        config_path: String,
        start_local: bool,
        auto_discovery: bool,
        auto_training: bool,
        settings: &Settings,
    ) -> Self {
        Self {
            config_path,
            data_dir: settings.system.data_dir.clone(),
            start_local,
            auto_discovery,
            auto_training,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn app_runtime_config_uses_settings_data_dir() {
        let mut settings = Settings::default();
        settings.system.data_dir = PathBuf::from("custom-data-root");

        let runtime = AppRuntimeConfig::from_settings(
            "config.yaml".to_string(),
            true,
            false,
            false,
            &settings,
        );

        assert_eq!(runtime.data_dir, PathBuf::from("custom-data-root"));
        assert!(runtime.start_local);
        assert!(!runtime.auto_discovery);
        assert!(!runtime.auto_training);
    }
}

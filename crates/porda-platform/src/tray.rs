use std::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    OpenSettings,
    ToggleDetection,
    TakeScreenshot,
    RefreshHotkeys,
    RefreshOverlay,
    Exit,
}

pub struct TrayManager {
    action_tx: mpsc::Sender<TrayAction>,
}

impl TrayManager {
    pub fn new(action_tx: mpsc::Sender<TrayAction>) -> Self {
        Self { action_tx }
    }

    pub fn send_action(&self, action: TrayAction) -> Result<(), String> {
        self.action_tx
            .send(action)
            .map_err(|e| e.to_string())
    }

    pub fn show_notification(&self, title: &str, message: &str) {
        tracing::info!("Tray notification: {} - {}", title, message);
    }
}

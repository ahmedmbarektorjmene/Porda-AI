use std::sync::mpsc;

use porda_platform::tray::{TrayAction, TrayManager};

pub struct PordaTray {
    manager: TrayManager,
}

impl PordaTray {
    pub fn new(action_tx: mpsc::Sender<TrayAction>) -> Self {
        let manager = TrayManager::new(action_tx);
        Self { manager }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("System tray initialized");
        Ok(())
    }

    pub fn show_notification(&self, title: &str, message: &str) {
        self.manager.show_notification(title, message);
    }

    pub fn send_action(&self, action: TrayAction) -> Result<(), String> {
        self.manager.send_action(action)
    }
}

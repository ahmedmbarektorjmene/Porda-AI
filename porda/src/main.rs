use std::sync::{mpsc, Arc, Mutex};

use porda_core::app_state::AppState;
use porda_core::commands::{CoreEvent, UiCommand};
use porda_core::pipeline::Pipeline;
use porda_platform::tray::TrayAction;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Porda AI starting...");

    porda_platform::ensure_app_directories().map_err(|e| {
        tracing::error!("Failed to create directories: {}", e);
        e
    })?;

    if porda_platform::check_duplicate_instances() {
        tracing::warn!("Another instance is already running");
        porda_platform::show_message("Porda AI", "Another instance is already running.");
        return Ok(());
    }

    let config = porda_config::defaults::load_config().unwrap_or_default();

    let (cmd_tx, cmd_rx) = mpsc::channel::<UiCommand>();
    let (event_tx, event_rx) = mpsc::channel::<CoreEvent>();
    let (tray_tx, tray_rx) = mpsc::channel::<TrayAction>();

    let ui_state = porda_ui::create_shared_state();
    let core_state = Arc::new(Mutex::new(AppState::new(config.clone())));

    let pipeline = Pipeline::new(Arc::clone(&core_state), event_tx.clone());
    pipeline.start();

    let tray = porda_tray::PordaTray::new(tray_tx);
    tray.run()?;

    let ui_state_for_ui = Arc::clone(&ui_state);
    let cmd_tx_for_ui = cmd_tx.clone();
    let ui_handle = std::thread::Builder::new()
        .name("porda-ui".to_string())
        .spawn(move || {
            let app = porda_ui::PordaApp::new(ui_state_for_ui, cmd_tx_for_ui);
            if let Err(e) = app.run() {
                tracing::error!("UI error: {}", e);
            }
        })?;

    let ui_state_for_events = Arc::clone(&ui_state);
    let event_handle = std::thread::Builder::new()
        .name("porda-event-handler".to_string())
        .spawn(move || {
            while let Ok(event) = event_rx.recv() {
                match event {
                    CoreEvent::DetectionStateChange(state) => {
                        let mut ui = ui_state_for_events.lock().unwrap();
                        ui.detection_state = format!("{:?}", state);
                        tracing::info!("Detection state: {:?}", state);
                    }
                    CoreEvent::CpuUsageUpdate(usage) => {
                        let mut ui = ui_state_for_events.lock().unwrap();
                        ui.cpu_usage = usage;
                    }
                    CoreEvent::CoversUpdated(_covers) => {}
                    CoreEvent::ScreenshotTaken(path) => {
                        tracing::info!("Screenshot saved: {:?}", path);
                    }
                    CoreEvent::Error(msg) => {
                        tracing::error!("Core error: {}", msg);
                    }
                    CoreEvent::ConfigSaved => {
                        tracing::info!("Configuration saved");
                    }
                    CoreEvent::ConfigLoaded(_config) => {}
                    CoreEvent::Terminated => {
                        tracing::info!("Terminated");
                        break;
                    }
                }
            }
        })?;

    let cmd_tx_for_commands = cmd_tx.clone();
    let core_state_for_commands = Arc::clone(&core_state);
    let ui_state_for_commands = Arc::clone(&ui_state);
    let command_handle = std::thread::Builder::new()
        .name("porda-command-handler".to_string())
        .spawn(move || {
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    UiCommand::SaveSettings => {
                        let config = {
                            let ui = ui_state_for_commands.lock().unwrap();
                            ui.to_config()
                        };
                        if let Err(e) = porda_config::defaults::save_config(&config) {
                            tracing::error!("Failed to save config: {}", e);
                        }
                        let _ = cmd_tx_for_commands.send(UiCommand::SaveSettings);
                    }
                    UiCommand::LoadSettings(config) => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(config);
                    }
                    UiCommand::RestoreDefaults => {
                        let default_config = porda_config::settings::PordaConfig::default();
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(default_config);
                    }
                    UiCommand::Activate => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = true;
                        tracing::info!("Detection activated");
                    }
                    UiCommand::Deactivate => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = false;
                        tracing::info!("Detection deactivated");
                    }
                    UiCommand::ToggleActivation => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = !state.is_active;
                        tracing::info!("Detection toggled: {}", state.is_active);
                    }
                    UiCommand::ApplySettings(config) => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(config);
                        tracing::info!("Settings applied");
                    }
                    UiCommand::Terminate => {
                        tracing::info!("Terminate requested");
                        break;
                    }
                    UiCommand::TakeScreenshot => match porda_platform::capture_screenshot() {
                        Some(_frame) => {
                            let dataset_dir = porda_config::defaults::dataset_dir();
                            let filename = format!("screenshot_{}.jpg", chrono_now());
                            let path = dataset_dir.join(filename);
                            tracing::info!("Screenshot captured: {:?}", path);
                        }
                        None => {
                            tracing::warn!("Failed to capture screenshot");
                        }
                    },
                    UiCommand::RefreshHotkeys => {
                        tracing::info!("Hotkeys refreshed");
                    }
                    UiCommand::RefreshOverlay => {
                        tracing::info!("Overlay refreshed");
                    }
                    _ => {}
                }
            }
        })?;

    let tray_handle = std::thread::Builder::new()
        .name("porda-tray-handler".to_string())
        .spawn(move || {
            while let Ok(action) = tray_rx.recv() {
                match action {
                    TrayAction::OpenSettings => {
                        tracing::info!("Open settings requested");
                    }
                    TrayAction::ToggleDetection => {
                        let _ = cmd_tx.send(UiCommand::ToggleActivation);
                    }
                    TrayAction::TakeScreenshot => {
                        let _ = cmd_tx.send(UiCommand::TakeScreenshot);
                    }
                    TrayAction::RefreshHotkeys => {
                        let _ = cmd_tx.send(UiCommand::RefreshHotkeys);
                    }
                    TrayAction::RefreshOverlay => {
                        let _ = cmd_tx.send(UiCommand::RefreshOverlay);
                    }
                    TrayAction::Exit => {
                        let _ = cmd_tx.send(UiCommand::Terminate);
                        break;
                    }
                }
            }
        })?;

    ui_handle.join().unwrap_or_else(|e| {
        tracing::error!("UI thread panicked: {:?}", e);
    });

    pipeline.stop();

    event_handle.join().unwrap_or_else(|e| {
        tracing::error!("Event handler thread panicked: {:?}", e);
    });

    command_handle.join().unwrap_or_else(|e| {
        tracing::error!("Command handler thread panicked: {:?}", e);
    });

    tray_handle.join().unwrap_or_else(|e| {
        tracing::error!("Tray handler thread panicked: {:?}", e);
    });

    tracing::info!("Porda AI stopped");
    Ok(())
}

fn chrono_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

use std::sync::Arc;

use crate::commands::UiCommandHandler;
use crate::state::SharedUiState;
use porda_core::commands::UiCommand;

slint::include_modules!();

pub struct PordaApp {
    ui_state: SharedUiState,
    command_tx: std::sync::mpsc::Sender<UiCommand>,
}

impl PordaApp {
    pub fn new(ui_state: SharedUiState, command_tx: std::sync::mpsc::Sender<UiCommand>) -> Self {
        Self {
            ui_state,
            command_tx,
        }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let app = AppWindow::new()?;

        let handler = UiCommandHandler::new(Arc::clone(&self.ui_state), self.command_tx.clone());

        {
            let state = self.ui_state.lock().unwrap();
            app.set_is_active(state.is_active);
            app.set_is_blur(state.is_blur);
            app.set_is_bg_color(state.is_bg_color);
            app.set_is_solid_color(state.is_solid_color);
            app.set_accuracy(state.accuracy as i32);
            app.set_network_width(state.network_width as i32);
            app.set_network_height(state.network_height as i32);
            app.set_active_timeout(state.active_timeout_ms as i32);
            app.set_sleep_timeout(state.sleep_timeout_ms as i32);
            app.set_keep_running_seconds(state.keep_running_seconds as i32);
            app.set_is_detect_male(state.is_detect_male);
            app.set_is_detect_female(state.is_detect_female);
            app.set_is_all_windows(state.is_all_windows);
            app.set_is_include_window(state.is_include_window);
            app.set_is_exclude_window(state.is_exclude_window);
            app.set_auto_startup(state.auto_startup);
            app.set_is_priority_realtime(state.is_priority_realtime);
            app.set_is_allow_max_cpu_limit(state.is_allow_max_cpu_limit);
            app.set_max_cpu_limit(state.max_cpu_limit as i32);
            app.set_cpu_usage(state.cpu_usage);
            app.set_current_page(state.current_page as i32);
        }

        let weak = app.as_weak();

        {
            let h = handler.clone();
            app.on_activate(move || {
                h.activate();
            });
        }

        {
            let h = handler.clone();
            app.on_deactivate(move || {
                h.deactivate();
            });
        }

        {
            let h = handler.clone();
            app.on_toggle_activation(move || {
                h.toggle_activation();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_save(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                }
                h.save_settings();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_ok_and_close(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                }
                h.save_settings();
                slint::quit_event_loop().ok();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_apply(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                }
                h.apply_settings();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_restore_defaults(move || {
                h.restore_defaults();
                if let Some(app) = weak.upgrade() {
                    let state = ui_state.lock().unwrap();
                    app.set_accuracy(state.accuracy as i32);
                    app.set_network_width(state.network_width as i32);
                    app.set_network_height(state.network_height as i32);
                    app.set_active_timeout(state.active_timeout_ms as i32);
                    app.set_sleep_timeout(state.sleep_timeout_ms as i32);
                    app.set_keep_running_seconds(state.keep_running_seconds as i32);
                    app.set_is_detect_male(state.is_detect_male);
                    app.set_is_detect_female(state.is_detect_female);
                    app.set_is_blur(state.is_blur);
                    app.set_is_bg_color(state.is_bg_color);
                    app.set_is_solid_color(state.is_solid_color);
                    app.set_auto_startup(state.auto_startup);
                    app.set_is_priority_realtime(state.is_priority_realtime);
                    app.set_is_allow_max_cpu_limit(state.is_allow_max_cpu_limit);
                    app.set_max_cpu_limit(state.max_cpu_limit as i32);
                }
            });
        }

        {
            let h = handler.clone();
            app.on_take_screenshot(move || {
                h.take_screenshot();
            });
        }

        {
            let h = handler.clone();
            app.on_refresh_hotkeys(move || {
                h.refresh_hotkeys();
            });
        }

        {
            let h = handler.clone();
            app.on_terminate(move || {
                h.terminate();
                slint::quit_event_loop().ok();
            });
        }

        {
            let ui_state = Arc::clone(&self.ui_state);
            app.on_navigate(move |page| {
                let mut state = ui_state.lock().unwrap();
                state.current_page = page as usize;
            });
        }

        app.run()?;

        Ok(())
    }
}

fn sync_ui_to_state(state: &SharedUiState, app: &AppWindow) {
    let mut s = state.lock().unwrap();
    s.is_blur = app.get_is_blur();
    s.is_bg_color = app.get_is_bg_color();
    s.is_solid_color = app.get_is_solid_color();
    s.accuracy = app.get_accuracy() as u8;
    s.network_width = app.get_network_width() as u32;
    s.network_height = app.get_network_height() as u32;
    s.active_timeout_ms = app.get_active_timeout() as u64;
    s.sleep_timeout_ms = app.get_sleep_timeout() as u64;
    s.keep_running_seconds = app.get_keep_running_seconds() as u64;
    s.is_detect_male = app.get_is_detect_male();
    s.is_detect_female = app.get_is_detect_female();
    s.is_all_windows = app.get_is_all_windows();
    s.is_include_window = app.get_is_include_window();
    s.is_exclude_window = app.get_is_exclude_window();
    s.auto_startup = app.get_auto_startup();
    s.is_priority_realtime = app.get_is_priority_realtime();
    s.is_allow_max_cpu_limit = app.get_is_allow_max_cpu_limit();
    s.max_cpu_limit = app.get_max_cpu_limit() as u8;
}

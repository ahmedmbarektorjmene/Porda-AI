use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::app_state::AppState;
use crate::commands::CoreEvent;
use porda_capture::capturer::{PlatformCapturer, ScreenCapturer};
use porda_inference::detector::{Detector, MockDetector, OpenCvDetector};
use porda_overlay::compositor::{CpuOverlayRenderer, OverlayRenderer};
use porda_vision::cover::covers_for_detections;

pub struct Pipeline {
    state: Arc<Mutex<AppState>>,
    event_tx: std::sync::mpsc::Sender<CoreEvent>,
    running: Arc<Mutex<bool>>,
}

impl Pipeline {
    pub fn new(
        state: Arc<Mutex<AppState>>,
        event_tx: std::sync::mpsc::Sender<CoreEvent>,
    ) -> Self {
        Self {
            state,
            event_tx,
            running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start(&self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = true;
        }

        let state = Arc::clone(&self.state);
        let event_tx = self.event_tx.clone();
        let running = Arc::clone(&self.running);

        std::thread::Builder::new()
            .name("porda-pipeline".to_string())
            .spawn(move || {
                tracing::info!("Pipeline thread started");

                let capturer = PlatformCapturer::new();
                let (config_path, weights_path) = porda_platform::model_paths();
                let detector: Box<dyn Detector> = if config_path.exists() && weights_path.exists() {
                    Box::new(OpenCvDetector::new(config_path, weights_path))
                } else {
                    Box::new(MockDetector)
                };
                let mut overlay = CpuOverlayRenderer::new(
                    porda_vision::geometry::ColorRgb::default(),
                );

                loop {
                    if !*running.lock().unwrap() {
                        break;
                    }

                    let interval_ms = {
                        let state = state.lock().unwrap();
                        if !state.should_run_detection() {
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        }
                        state.detection_interval_ms()
                    };

                    match capturer.capture_foreground(
                        &state.lock().unwrap().config.windows.include_windows,
                        &state.lock().unwrap().config.windows.exclude_windows,
                        &state.lock().unwrap().config.windows.always_skip_windows,
                    ) {
                        Ok(captured) => {
                            match detector.detect(
                                &captured.frame,
                                state.lock().unwrap().confidence_threshold(),
                                state.lock().unwrap().config.detection.nms_threshold,
                                &state.lock().unwrap().target_classes(),
                                state.lock().unwrap().config.detection.network_width,
                                state.lock().unwrap().config.detection.network_height,
                                &captured.window_rect,
                            ) {
                                Ok(detections) => {
                                    let has_detections = !detections.is_empty();

                                    {
                                        let mut s = state.lock().unwrap();
                                        s.last_detections = detections.clone();
                                        s.update_detection_state(has_detections);
                                    }

                                    {
                                        let covers = covers_for_detections(
                                            &detections,
                                            &captured.frame,
                                            state.lock().unwrap().cover_mode(),
                                            state.lock().unwrap().solid_color(),
                                            &state.lock().unwrap().window_rects,
                                        );

                                        let _ = overlay.update_covers(&covers, &captured.frame);

                                        {
                                            let mut s = state.lock().unwrap();
                                            s.covers = covers.clone();
                                        }

                                        let _ = event_tx.send(CoreEvent::CoversUpdated(covers));
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Detection failed: {}", e);
                                }
                            }
                        }
                        Err(porda_capture::capturer::CaptureError::NoForegroundWindow) => {
                            std::thread::sleep(Duration::from_millis(interval_ms));
                            continue;
                        }
                        Err(e) => {
                            tracing::debug!("Capture failed: {}", e);
                        }
                    }

                    std::thread::sleep(Duration::from_millis(interval_ms));
                }

                tracing::info!("Pipeline thread stopped");
            })
            .expect("Failed to spawn pipeline thread");
    }

    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }

    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

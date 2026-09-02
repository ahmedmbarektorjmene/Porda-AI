use porda_vision::detection::{Detection, FrameData};
use porda_vision::geometry::ScreenRect;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Inference failed: {0}")]
    Failed(String),
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
}

pub trait Detector: Send + Sync {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError>;

    fn backend_name(&self) -> &str;
}

pub struct MockDetector;

impl Detector for MockDetector {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        _nms_threshold: f32,
        target_classes: &[i32],
        _network_width: u32,
        _network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        tracing::info!(
            "MockDetector: frame {}x{} target_classes={:?} conf_thresh={:.2}",
            frame.width,
            frame.height,
            target_classes,
            confidence_threshold
        );
        if std::env::var("PORDA_MOCK_DETECTIONS").is_ok() {
            tracing::info!("MockDetector: PORDA_MOCK_DETECTIONS set, generating synthetic detection");
            if target_classes.contains(&1) && confidence_threshold <= 0.9 {
                let w = (frame.width / 4).min(300);
                let h = (frame.height / 4).min(200);
                let x = (frame.width as i32 / 2) - (w as i32 / 2);
                let y = (frame.height as i32 / 2) - (h as i32 / 2);
                tracing::info!(
                    "MockDetector: returning 1 detection at ({},{},{},{})",
                    x, y, w, h
                );
                return Ok(vec![Detection {
                    class: porda_vision::detection::ObjectClass::Female,
                    confidence: 0.91,
                    screen_rect: ScreenRect::new(x, y, w, h),
                }]);
            } else {
                tracing::info!(
                    "MockDetector: not generating (target_classes={:?}, thresh={})",
                    target_classes,
                    confidence_threshold
                );
            }
        }
        let _ = screen_rect;
        Ok(vec![])
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

#[cfg(feature = "opencv")]
pub struct OpenCvDetector {
    config_path: std::path::PathBuf,
    weights_path: std::path::PathBuf,
    model: std::sync::Arc<std::sync::Mutex<Option<opencv::dnn::DetectionModel>>>,
}

#[cfg(not(feature = "opencv"))]
pub struct OpenCvDetector {
    config_path: std::path::PathBuf,
    weights_path: std::path::PathBuf,
}

#[cfg(feature = "opencv")]
impl OpenCvDetector {
    pub fn new(config_path: std::path::PathBuf, weights_path: std::path::PathBuf) -> Self {
        use opencv::prelude::*;
        tracing::info!("OpenCvDetector: REAL OpenCV/DNN detector active");
        tracing::info!(
            "OpenCvDetector: loading Darknet model cfg={:?} weights={:?}",
            config_path,
            weights_path
        );
        let cfg_exists = config_path.exists();
        let w_exists = weights_path.exists();
        tracing::info!(
            "OpenCvDetector: config {:?} exists={}, weights {:?} exists={}",
            config_path,
            cfg_exists,
            weights_path,
            w_exists
        );
        if !cfg_exists || !w_exists {
            tracing::warn!(
                "OpenCvDetector: model files not found at {:?} / {:?} — \
                 Python reference model is at /home/torchi/Desktop/Porda-AI/Porda-AI/model/ \
                 (pordav4x3.cfg, porda-19200-lr-0005-909.weights).",
                config_path,
                weights_path
            );
            return Self {
                config_path,
                weights_path,
                model: std::sync::Arc::new(std::sync::Mutex::new(None)),
            };
        }

        let model = match opencv::dnn::DetectionModel::new(
            weights_path.to_str().unwrap_or(""),
            config_path.to_str().unwrap_or(""),
        ) {
            Ok(mut m) => {
                if let Err(e) = m.set_input_params(
                    1.0 / 255.0,
                    opencv::core::Size::new(544, 320),
                    opencv::core::Scalar::default(),
                    true,
                    false,
                ) {
                    tracing::error!("OpenCvDetector: set_input_params failed: {}", e);
                    None
                } else {
                    tracing::info!("OpenCvDetector: Darknet model loaded successfully");
                    tracing::info!("OpenCvDetector: model loaded and configured (544x320, 1/255, swapRB=true)");
                    Some(m)
                }
            }
            Err(e) => {
                tracing::error!("OpenCvDetector: failed to load model: {}", e);
                tracing::error!("OpenCvDetector: model initialization error: {}", e);
                None
            }
        };

        Self {
            config_path,
            weights_path,
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
        }
    }
}

#[cfg(not(feature = "opencv"))]
impl OpenCvDetector {
    pub fn new(config_path: std::path::PathBuf, weights_path: std::path::PathBuf) -> Self {
        let cfg_exists = config_path.exists();
        let w_exists = weights_path.exists();
        tracing::info!(
            "OpenCvDetector: config {:?} exists={}, weights {:?} exists={}",
            config_path,
            cfg_exists,
            weights_path,
            w_exists
        );
        if !cfg_exists || !w_exists {
            tracing::warn!(
                "OpenCvDetector: model files not found at {:?} / {:?} — \
                 Python reference model is at /home/torchi/Desktop/Porda-AI/Porda-AI/model/ \
                 (pordav4x3.cfg, porda-19200-lr-0005-909.weights).",
                config_path,
                weights_path
            );
        }
        Self {
            config_path,
            weights_path,
        }
    }
}

#[cfg(feature = "opencv")]
impl Detector for OpenCvDetector {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        use opencv::prelude::*;
        let mut guard = self
            .model
            .lock()
            .map_err(|e| InferenceError::Failed(format!("Model mutex poisoned: {}", e)))?;

        let model = guard.as_mut().ok_or(InferenceError::ModelNotLoaded)?;

        let (padded_data, x_ratio, y_ratio) = porda_vision::preprocessing::resize_and_pad(
            &frame.data,
            frame.width,
            frame.height,
            network_width,
            network_height,
        );

        let (padded_w, padded_h) = if (x_ratio - 1.0).abs() < 0.01 && (y_ratio - 1.0).abs() < 0.01 {
            (frame.width, frame.height)
        } else {
            (network_width, network_height)
        };

        tracing::info!(
            "OpenCvDetector: preprocessing {}x{} -> padded {}x{} ratios {:.2},{:.2} network {}x{}",
            frame.width,
            frame.height,
            padded_w,
            padded_h,
            x_ratio,
            y_ratio,
            network_width,
            network_height
        );

        let vec3b_slice: &[opencv::core::Vec3b] = unsafe {
            std::slice::from_raw_parts(
                padded_data.as_ptr() as *const opencv::core::Vec3b,
                (padded_w * padded_h) as usize,
            )
        };
        let mat = opencv::core::Mat::new_rows_cols_with_data(
            padded_h as i32,
            padded_w as i32,
            vec3b_slice,
        )
        .map_err(|e| InferenceError::Failed(format!("Mat creation failed: {}", e)))?;

        let mut classes = opencv::core::Vector::<i32>::new();
        let mut confidences = opencv::core::Vector::<f32>::new();
        let mut boxes = opencv::core::Vector::<opencv::core::Rect>::new();

        tracing::info!(
            "OpenCvDetector: inference started (conf={}, nms={})",
            confidence_threshold,
            nms_threshold
        );

        model
            .detect(
                &mat,
                &mut classes,
                &mut confidences,
                &mut boxes,
                confidence_threshold,
                nms_threshold,
            )
            .map_err(|e| InferenceError::Failed(format!("detect failed: {}", e)))?;

        tracing::info!(
            "OpenCvDetector: inference completed: {} raw detections",
            classes.len()
        );

        let mut detections = Vec::new();
        for i in 0..classes.len() {
            let class_id = classes.get(i).map_err(|e| {
                InferenceError::Failed(format!("classes.get failed: {}", e))
            })?;
            let conf = confidences.get(i).map_err(|e| {
                InferenceError::Failed(format!("confidences.get failed: {}", e))
            })?;
            let rect = boxes.get(i).map_err(|e| {
                InferenceError::Failed(format!("boxes.get failed: {}", e))
            })?;

            if !target_classes.contains(&class_id) {
                tracing::debug!(
                    "OpenCvDetector: filtering class {} not in target {:?}",
                    class_id,
                    target_classes
                );
                continue;
            }

            let x = (rect.x as f32 * x_ratio) as i32 + screen_rect.x;
            let y = (rect.y as f32 * y_ratio) as i32 + screen_rect.y;
            let w = (rect.width as f32 * x_ratio) as u32;
            let h = (rect.height as f32 * y_ratio) as u32;

            let class =
                porda_vision::detection::ObjectClass::from_id(class_id).unwrap_or(
                    porda_vision::detection::ObjectClass::Female,
                );

            tracing::info!(
                "OpenCvDetector: detection class={:?} conf={:.2} bbox=({},{},{},{})",
                class,
                conf,
                x,
                y,
                w,
                h
            );

            detections.push(Detection {
                class,
                confidence: conf,
                screen_rect: ScreenRect::new(x, y, w, h),
            });
        }

        Ok(detections)
    }

    fn backend_name(&self) -> &str {
        "opencv-dnn"
    }
}

#[cfg(not(feature = "opencv"))]
impl Detector for OpenCvDetector {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        let cfg_ok = self.config_path.exists();
        let w_ok = self.weights_path.exists();
        if !cfg_ok || !w_ok {
            return Err(InferenceError::ModelNotLoaded);
        }
        tracing::warn!(
            "OpenCvDetector: model found at {:?} / {:?} ({}x{}), \
             but inference requires `opencv` feature (Cargo `porda-inference` with `features=[\"opencv\"]`). \
             System has opencv5 (pkg-config opencv5), but Rust was built without `opencv` feature. \
             Frame {}x{} would be preprocessed via resize_and_pad to {}x{} \
             with scale 1/255, then YOLO decode (anchors, objectness×class, NMS 0.1) \
             and x_ratio/y_ratio scaling to {:?}. Returning BackendNotAvailable \
             to distinguish from 0 detections.",
            self.config_path,
            self.weights_path,
            network_width,
            network_height,
            frame.width,
            frame.height,
            network_width,
            network_height,
            screen_rect
        );
        let _ = (confidence_threshold, nms_threshold, target_classes);
        Err(InferenceError::BackendNotAvailable(
            "opencv feature not enabled (need `cargo build -p porda-inference --features opencv` \
             with OPENCV_PKGCONFIG_NAME=opencv5 and libclang)".to_string(),
        ))
    }

    fn backend_name(&self) -> &str {
        "opencv-dnn"
    }
}

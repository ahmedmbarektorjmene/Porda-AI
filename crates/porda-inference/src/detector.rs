use porda_vision::detection::Detection;
use porda_vision::detection::FrameData;
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
        _frame: &FrameData,
        _confidence_threshold: f32,
        _nms_threshold: f32,
        _target_classes: &[i32],
        _network_width: u32,
        _network_height: u32,
        _screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        Ok(vec![])
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

pub struct OpenCvDetector {
    config_path: std::path::PathBuf,
    weights_path: std::path::PathBuf,
}

impl OpenCvDetector {
    pub fn new(config_path: std::path::PathBuf, weights_path: std::path::PathBuf) -> Self {
        Self {
            config_path,
            weights_path,
        }
    }
}

impl Detector for OpenCvDetector {
    fn detect(
        &self,
        _frame: &FrameData,
        _confidence_threshold: f32,
        _nms_threshold: f32,
        _target_classes: &[i32],
        _network_width: u32,
        _network_height: u32,
        _screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        tracing::warn!(
            "OpenCV DNN inference not yet implemented in Rust. Config: {:?}, Weights: {:?}",
            self.config_path,
            self.weights_path
        );
        Ok(vec![])
    }

    fn backend_name(&self) -> &str {
        "opencv-dnn"
    }
}

use porda_vision::detection::{CoverRect, FrameData};
use porda_vision::geometry::{ColorRgb, ScreenRect};

#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("Overlay creation failed: {0}")]
    CreationFailed(String),
    #[error("Rendering failed: {0}")]
    RenderFailed(String),
}

pub trait OverlayRenderer: Send + Sync {
    fn update_covers(&mut self, covers: &[CoverRect], frame: &FrameData) -> Result<(), OverlayError>;
    fn clear(&mut self) -> Result<(), OverlayError>;
    fn set_geometry(&mut self, monitors: &[ScreenRect]) -> Result<(), OverlayError>;
}

pub struct CpuOverlayRenderer {
    covers: Vec<CoverRect>,
    #[allow(dead_code)]
    solid_color: ColorRgb,
}

impl CpuOverlayRenderer {
    pub fn new(solid_color: ColorRgb) -> Self {
        Self {
            covers: Vec::new(),
            solid_color,
        }
    }
}

impl OverlayRenderer for CpuOverlayRenderer {
    fn update_covers(&mut self, covers: &[CoverRect], _frame: &FrameData) -> Result<(), OverlayError> {
        self.covers = covers.to_vec();
        Ok(())
    }

    fn clear(&mut self) -> Result<(), OverlayError> {
        self.covers.clear();
        Ok(())
    }

    fn set_geometry(&mut self, _monitors: &[ScreenRect]) -> Result<(), OverlayError> {
        Ok(())
    }
}

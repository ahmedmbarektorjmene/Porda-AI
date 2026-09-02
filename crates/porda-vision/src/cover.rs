use crate::detection::{CoverMode, CoverRect, FrameData, extract_dominant_color};
use crate::geometry::{ColorRgb, ScreenRect};

pub fn generate_cover(
    frame: &FrameData,
    rect: &ScreenRect,
    mode: CoverMode,
    _solid_color: ColorRgb,
) -> CoverRect {
    match mode {
        CoverMode::SolidColor => CoverRect {
            screen_rect: *rect,
            mode,
        },
        CoverMode::BackgroundColor => {
            let _ = extract_dominant_color(frame, rect);
            CoverRect {
                screen_rect: *rect,
                mode,
            }
        }
        CoverMode::Blur => CoverRect {
            screen_rect: *rect,
            mode,
        },
    }
}

pub fn generate_blur_data(frame: &FrameData, rect: &ScreenRect) -> Option<Vec<u8>> {
    let region = frame.region(rect)?;
    Some(crate::preprocessing::blur_region(
        &region.data,
        region.width,
        region.height,
    ))
}

pub fn covers_for_detections(
    detections: &[crate::detection::Detection],
    frame: &FrameData,
    mode: CoverMode,
    solid_color: ColorRgb,
    window_rects: &[ScreenRect],
) -> Vec<CoverRect> {
    let mut covers = Vec::new();

    for det in detections {
        let cover = generate_cover(frame, &det.screen_rect, mode, solid_color);

        let mut final_parts = Vec::new();
        let mut current_parts = vec![cover.screen_rect];

        for win_rect in window_rects {
            let mut next_parts = Vec::new();
            for part in &current_parts {
                if let Some(inter) = part.intersection(win_rect) {
                    if inter == *part {
                        continue;
                    }
                    let subtracted = part.subtract(win_rect);
                    next_parts.extend(subtracted);
                } else {
                    next_parts.push(*part);
                }
            }
            current_parts = next_parts;
        }

        for part in current_parts {
            final_parts.push(CoverRect {
                screen_rect: part,
                mode: cover.mode,
            });
        }

        covers.append(&mut final_parts);
    }

    covers
}

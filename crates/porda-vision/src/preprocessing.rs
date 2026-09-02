use crate::geometry::ScreenRect;

pub fn resize_and_pad(
    data: &[u8],
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> (Vec<u8>, f32, f32) {
    let scale_x = target_width as f32 / src_width as f32;
    let scale_y = target_height as f32 / src_height as f32;
    let scale = scale_x.min(scale_y);

    let new_w = (src_width as f32 * scale) as u32;
    let new_h = (src_height as f32 * scale) as u32;

    let mut resized = vec![0u8; (new_w * new_h * 3) as usize];

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x as f32 / scale) as u32;
            let src_y = (y as f32 / scale) as u32;
            let src_x = src_x.min(src_width - 1);
            let src_y = src_y.min(src_height - 1);

            let src_idx = ((src_y * src_width + src_x) * 3) as usize;
            let dst_idx = ((y * new_w + x) * 3) as usize;

            if src_idx + 2 < data.len() && dst_idx + 2 < resized.len() {
                resized[dst_idx] = data[src_idx];
                resized[dst_idx + 1] = data[src_idx + 1];
                resized[dst_idx + 2] = data[src_idx + 2];
            }
        }
    }

    let mut padded = vec![0u8; (target_width * target_height * 3) as usize];
    for y in 0..new_h {
        let src_start = (y * new_w * 3) as usize;
        let src_end = src_start + (new_w * 3) as usize;
        let dst_start = (y * target_width * 3) as usize;
        let dst_end = dst_start + (new_w * 3).min(target_width * 3) as usize;
        if src_end <= resized.len() && dst_end <= padded.len() {
            padded[dst_start..dst_end].copy_from_slice(&resized[src_start..src_end]);
        }
    }

    (padded, scale, scale)
}

pub fn nms(
    boxes: &mut Vec<ScreenRect>,
    scores: &mut Vec<f32>,
    threshold: f32,
) {
    if boxes.is_empty() {
        return;
    }

    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; indices.len()];

    for i in 0..indices.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..indices.len() {
            if suppressed[j] {
                continue;
            }
            let a = boxes[indices[i]];
            let b = boxes[indices[j]];

            if let Some(inter) = a.intersection(&b) {
                let inter_area = inter.area() as f32;
                let union_area = a.area() as f32 + b.area() as f32 - inter_area;
                let iou = if union_area > 0.0 {
                    inter_area / union_area
                } else {
                    0.0
                };

                if iou > threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    let mut write_idx = 0;
    for i in 0..indices.len() {
        if !suppressed[i] {
            boxes[write_idx] = boxes[indices[i]];
            scores[write_idx] = scores[indices[i]];
            write_idx += 1;
        }
    }
    boxes.truncate(write_idx);
    scores.truncate(write_idx);
}

pub fn blur_region(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![];
    }

    let kernel = (width / 2).max(1);
    let mut output = vec![0u8; data.len()];

    for y in 0..height {
        for x in 0..width {
            let mut r_sum: u32 = 0;
            let mut g_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut count: u32 = 0;

            let ky_start = y.saturating_sub(kernel);
            let ky_end = (y + kernel + 1).min(height);
            let kx_start = x.saturating_sub(kernel);
            let kx_end = (x + kernel + 1).min(width);

            for ky in ky_start..ky_end {
                for kx in kx_start..kx_end {
                    let idx = ((ky * width + kx) * 3) as usize;
                    if idx + 2 < data.len() {
                        b_sum += data[idx] as u32;
                        g_sum += data[idx + 1] as u32;
                        r_sum += data[idx + 2] as u32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < output.len() {
                    output[idx] = (b_sum / count) as u8;
                    output[idx + 1] = (g_sum / count) as u8;
                    output[idx + 2] = (r_sum / count) as u8;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_and_pad() {
        let data = vec![128u8; 100 * 100 * 3];
        let (padded, sx, sy) = resize_and_pad(&data, 100, 100, 320, 320);
        assert_eq!(padded.len(), 320 * 320 * 3);
        assert!((sx - 3.2).abs() < 0.01);
    }
}

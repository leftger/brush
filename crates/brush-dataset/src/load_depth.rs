use brush_vfs::BrushVfs;
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncReadExt;

/// Ground-truth depth map loader for a single scene view.
///
/// Supports 16-bit grayscale PNGs (defaulting to millimetre units, scale `0.001`)
/// and 32-bit float EXRs (defaulting to metres, scale `1.0`), unless an explicit
/// `unit_scale` (`depth_unit_scale_factor`) is supplied.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    unit_scale: Option<f32>,
    scene_scale: f32,
}

impl LoadDepth {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf, unit_scale: Option<f32>) -> Self {
        Self {
            vfs,
            path,
            unit_scale,
            scene_scale: 1.0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Multiply the loaded depth values by `factor` (used when `units_per_meter != 1.0`).
    pub fn scale_unit(&mut self, factor: f32) {
        self.scene_scale *= factor;
    }

    /// Load and decode the depth map, nearest-neighbor resized to `target_size` `(w, h)`.
    ///
    /// Returns row-major `f32` depths in scene units (metres), with non-positive
    /// or non-finite pixels normalised to `0.0`.
    pub async fn load(&self, target_size: (u32, u32)) -> image::ImageResult<Vec<f32>> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;
        let img = image::load_from_memory(&bytes)?;
        Ok(decode_depth_image(
            &img,
            target_size,
            self.unit_scale,
            self.scene_scale,
        ))
    }
}

fn decode_depth_image(
    img: &DynamicImage,
    target_size: (u32, u32),
    unit_scale: Option<f32>,
    scene_scale: f32,
) -> Vec<f32> {
    let (src_w, src_h) = (img.width(), img.height());
    let (dst_w, dst_h) = target_size;
    assert!(dst_w > 0 && dst_h > 0, "depth target size must be non-zero");

    let default_scale = match img {
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => 1.0,
        _ => 0.001,
    };
    let scale = unit_scale.unwrap_or(default_scale) * scene_scale;

    let sample = |idx: usize| -> f32 {
        let raw = match img {
            DynamicImage::ImageLuma16(buf) => buf.as_raw()[idx] as f32,
            DynamicImage::ImageLuma8(buf) => buf.as_raw()[idx] as f32,
            DynamicImage::ImageLumaA16(buf) => buf.as_raw()[idx * 2] as f32,
            DynamicImage::ImageLumaA8(buf) => buf.as_raw()[idx * 2] as f32,
            DynamicImage::ImageRgb16(buf) => buf.as_raw()[idx * 3] as f32,
            DynamicImage::ImageRgba16(buf) => buf.as_raw()[idx * 4] as f32,
            DynamicImage::ImageRgb8(buf) => buf.as_raw()[idx * 3] as f32,
            DynamicImage::ImageRgba8(buf) => buf.as_raw()[idx * 4] as f32,
            DynamicImage::ImageRgb32F(buf) => buf.as_raw()[idx * 3],
            DynamicImage::ImageRgba32F(buf) => buf.as_raw()[idx * 4],
            other => other.to_luma16().as_raw()[idx] as f32,
        };
        let d = raw * scale;
        if d.is_finite() && d > 0.0 { d } else { 0.0 }
    };

    let n = (dst_w as usize) * (dst_h as usize);
    let mut out = Vec::with_capacity(n);

    if src_w == dst_w && src_h == dst_h {
        for idx in 0..n {
            out.push(sample(idx));
        }
    } else {
        // Nearest-neighbor avoids interpolating across depth edges (flying pixels).
        for dy in 0..dst_h {
            let sy = ((dy as u64 * src_h as u64) / dst_h as u64).min(src_h as u64 - 1) as usize;
            for dx in 0..dst_w {
                let sx =
                    ((dx as u64 * src_w as u64) / dst_w as u64).min(src_w as u64 - 1) as usize;
                out.push(sample(sy * src_w as usize + sx));
            }
        }
    }

    out
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb, Rgb32FImage};

    #[tokio::test]
    async fn loads_16bit_png_in_millimetres_and_resizes_nearest() {
        let dir = tempfile::tempdir().expect("temp dir");
        let depth_img: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(4, 2, vec![0, 1500, 2000, 4500, 500, 0, 3000, 1000])
                .expect("valid luma16");
        depth_img
            .save(dir.path().join("depth.png"))
            .expect("save depth");

        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.expect("vfs"));
        let loader = LoadDepth::new(vfs, "depth.png".into(), None);

        let full = loader.load((4, 2)).await.expect("load full");
        let expected_full = [0.0f32, 1.5, 2.0, 4.5, 0.5, 0.0, 3.0, 1.0];
        assert_eq!(full.len(), expected_full.len());
        for (a, b) in full.iter().zip(expected_full.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }

        let half = loader.load((2, 1)).await.expect("load half");
        // Nearest-neighbor samples (0,0) -> 0.0 and (2,0) -> 2.0, never averaging 0 and 1.5.
        assert_eq!(half, vec![0.0, 2.0]);
    }

    #[tokio::test]
    async fn loads_exr_float_depth_and_sanitizes_invalid_values() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut exr = Rgb32FImage::new(2, 1);
        exr.put_pixel(0, 0, Rgb([2.25, 0.0, 0.0]));
        exr.put_pixel(1, 0, Rgb([f32::NAN, 0.0, 0.0]));
        exr.save(dir.path().join("depth.exr")).expect("save exr");

        let vfs = Arc::new(BrushVfs::from_path(dir.path()).await.expect("vfs"));
        let mut loader = LoadDepth::new(vfs, "depth.exr".into(), None);
        loader.scale_unit(0.5);

        let loaded = loader.load((2, 1)).await.expect("load exr");
        assert_eq!(loaded, vec![1.125, 0.0]);
    }
}

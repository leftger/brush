//! Smoke + invariant tests for the loss kernels.
//!
//! GT lives as `[H, W]` u32 packing `[r g b a]` u8. We feed deterministic u8
//! data through `image_loss` and check structural properties (`SSIM(x, x) ≈ 1`,
//! output range, backward produces finite gradients). Bit-exact reference
//! matching is covered by the integration training tests in `brush-bench-test`.

use brush_loss::{
    ImageLossConfig, TILE_SIZE, image_loss, image_loss_eval, image_loss_partials, psnr,
    psnr_from_mse,
};
use burn::tensor::{Device, Int, Tensor, TensorData};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn pack_rgba(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|p| {
            u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16 | u32::from(p[3]) << 24
        })
        .collect()
}

/// Deterministic u8 pattern (avoids RNG so the test is reproducible across
/// machines). Returns `H*W*4` RGBA bytes.
fn make_pattern(h: usize, w: usize, scale: u32, offset: u32) -> Vec<u8> {
    (0..h * w * 4)
        .map(|i| ((i as u32 * scale + offset) % 251) as u8)
        .collect()
}

fn pred_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<3> {
    let rgb: Vec<f32> = bytes
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|b| b as f32 / 255.0))
        .collect();
    Tensor::<1>::from_floats(rgb.as_slice(), device).reshape([h, w, 3])
}

fn gt_packed_from_bytes(bytes: &[u8], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    // Bit-reinterpret the u32 packing as i32 so the dispatch int_from_data
    // path doesn't reject magnitudes > i32::MAX.
    let packed: Vec<i32> = pack_rgba(bytes).into_iter().map(|x| x as i32).collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

/// Read a tensor back as a flat `f32` vec.
async fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("vec")
}

fn ssim_only_cfg() -> ImageLossConfig {
    ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 0.0,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_identical_inputs_is_one() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes = make_pattern(h, w, 11, 13);
    let pred = pred_from_bytes(&bytes, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let map = to_vec(image_loss_eval(pred, gt, ssim_only_cfg())).await;
    let mean: f32 = map.iter().sum::<f32>() / (h * w * 3) as f32;
    // Identical inputs SSIM saturates at 1; allow a sub-ULP roundoff.
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "SSIM(x, x) should be 1, got {mean}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_in_clamp_range() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (40, 56);
    let bytes_a = make_pattern(h, w, 7, 19);
    let bytes_b = make_pattern(h, w, 13, 7);
    let pred = pred_from_bytes(&bytes_a, h, w, &device);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let data = to_vec(image_loss_eval(pred, gt, ssim_only_cfg())).await;
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (-1.0..=1.0).contains(&min) && (-1.0..=1.0).contains(&max),
        "SSIM out of [-1, 1]: min={min} max={max}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_loss_backward_runs() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (32, 48);
    let bytes_a = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let pred = pred_from_bytes(&bytes_a, h, w, &device).require_grad();
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);

    let loss = image_loss(
        pred.clone(),
        gt,
        ImageLossConfig {
            l1_weight: 0.8,
            ssim_weight: -0.2,
            composite_bg: None,
            mask: false,
            alpha_weight: 0.0,
        },
    );
    let grads = loss.backward();
    let data = to_vec(pred.grad(&grads).expect("pred should have a gradient")).await;
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > 0.0,
        "backward should produce non-zero gradients, got all zeros"
    );
    assert!(
        data.iter().all(|v| v.is_finite()),
        "gradients should be finite"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alpha_match_via_4ch_pred() {
    // Feeding 4-channel `pred` makes the kernel emit `|pred.a - gt.a|`
    // into the alpha channel of the loss map.
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (16, 24);
    let bytes = make_pattern(h, w, 17, 5);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
        .reshape([h, w, 4])
        .require_grad();
    let gt = gt_packed_from_bytes(&bytes, h, w, &device);

    let cfg = ImageLossConfig {
        l1_weight: 1.0,
        ssim_weight: 0.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let map = image_loss_eval(pred.clone(), gt.clone(), cfg);
    assert_eq!(map.dims(), [h, w, 4]);
    let partials = image_loss_partials(pred, gt, cfg);
    assert_eq!(partials.dims()[0], 4);
    let _grads = partials.sum().backward();
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn psnr_matches_known_values() {
    let device = Device::from(brush_cube::test_helpers::test_device().await);
    let scalar = async |t: Tensor<1>| t.into_scalar_async::<f32>().await.expect("readback");

    // Plain formula: MSE 0.01 -> 20 dB, MSE 1 -> 0 dB.
    let db = scalar(psnr_from_mse(Tensor::from_floats([0.01], &device))).await;
    assert!((db - 20.0).abs() < 1e-3, "got {db}");
    let db = scalar(psnr_from_mse(Tensor::from_floats([1.0], &device))).await;
    assert!(db.abs() < 1e-3, "got {db}");

    // Identical images floor at 100 dB instead of going infinite.
    let db = scalar(psnr_from_mse(Tensor::from_floats([0.0], &device))).await;
    assert!((db - 100.0).abs() < 1e-3, "got {db}");

    // Image wrapper: a constant 0.1 offset on every channel is MSE 0.01.
    let a = Tensor::<3>::zeros([4, 6, 3], &device);
    let b = Tensor::<3>::full([4, 6, 3], 0.1, &device);
    let db = scalar(psnr(a.clone(), b)).await;
    assert!((db - 20.0).abs() < 1e-3, "got {db}");
    let db = scalar(psnr(a.clone(), a)).await;
    assert!((db - 100.0).abs() < 1e-3, "got {db}");
}

/// The weighted per-tile partial sums must agree with the per-pixel map's
/// channel means.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn partials_match_map_sums() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (37, 53); // not a multiple of the 16x16 tile
    let bytes = make_pattern(h, w, 17, 5);
    let bytes_b = make_pattern(h, w, 3, 9);
    let rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device).reshape([h, w, 4]);
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.7,
        ssim_weight: 0.3,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let map = image_loss_eval(pred.clone(), gt.clone(), cfg);
    let map_sums = to_vec(map.sum_dims(&[0, 1]).reshape([4])).await;
    let partial_sums = to_vec(image_loss_partials(pred, gt, cfg).sum_dim(1).reshape([4])).await;
    let pixels = (h * w) as f32;
    for c in 0..4 {
        let weight = if c < 3 {
            1.0 / (3.0 * pixels)
        } else {
            cfg.alpha_weight / pixels
        };
        let (a, b) = (map_sums[c] * weight, partial_sums[c]);
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1e-3),
            "channel {c}: weighted map sum {a} vs partial sum {b}"
        );
    }
}

/// Finite-difference check of the loss gradient through the tile partials.
/// Each probe weights only its own channel, and gives every tile a slightly
/// different weight so the backward's per-tile upstream gradient path has
/// something non-uniform to pick up. Keeping it to one channel also keeps the
/// summed loss small enough for f32 finite differences to resolve (weighting
/// all four quantises them to ~0.03).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn partials_gradient_matches_finite_difference() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let (h, w) = (24, 40);
    let bytes = make_pattern(h, w, 5, 1);
    let bytes_b = make_pattern(h, w, 7, 11);
    let mut rgba: Vec<f32> = bytes.iter().map(|b| *b as f32 / 255.0).collect();
    // Keep pred away from exact equality with gt so the L1 sign is defined.
    for (i, v) in rgba.iter_mut().enumerate() {
        *v = (*v + 0.013 * ((i % 7) as f32 + 1.0)).min(0.97);
    }
    let gt = gt_packed_from_bytes(&bytes_b, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.6,
        ssim_weight: 0.4,
        composite_bg: None,
        mask: false,
        alpha_weight: 1.0,
    };
    let tiles = w.div_ceil(TILE_SIZE) * h.div_ceil(TILE_SIZE);

    // A few pixels: interior, tile edges, and the alpha channel.
    let probes = [(5, 7, 0), (15, 16, 1), (16, 33, 2), (23, 39, 0), (9, 20, 3)];
    let eps = 2e-3_f32;
    let mut failed = Vec::new();
    for (y, x, c) in probes {
        let mut wts = vec![0.0f32; 4 * tiles];
        for (tile, wt) in wts[c * tiles..(c + 1) * tiles].iter_mut().enumerate() {
            *wt = 1.0 + tile as f32 / tiles as f32;
        }
        let weights = Tensor::<1>::from_floats(wts.as_slice(), &device).reshape([4, tiles]);
        let loss_of = |data: &[f32]| {
            let pred = Tensor::<1>::from_floats(data, &device).reshape([h, w, 4]);
            (image_loss_partials(pred, gt.clone(), cfg) * weights.clone()).sum()
        };

        let pred = Tensor::<1>::from_floats(rgba.as_slice(), &device)
            .reshape([h, w, 4])
            .require_grad();
        let loss = (image_loss_partials(pred.clone(), gt.clone(), cfg) * weights.clone()).sum();
        let grads = loss.backward();
        let grad = to_vec(pred.grad(&grads).expect("grad")).await;

        let i = (y * w + x) * 4 + c;
        let mut plus = rgba.clone();
        plus[i] += eps;
        let mut minus = rgba.clone();
        minus[i] -= eps;
        let lp = loss_of(&plus).into_scalar_async::<f32>().await.expect("rb");
        let lm = loss_of(&minus)
            .into_scalar_async::<f32>()
            .await
            .expect("rb");
        let numerical = (lp - lm) / (2.0 * eps);
        let analytical = grad[i];
        let tol = 5e-3 + 0.02 * numerical.abs().max(analytical.abs());
        if (numerical - analytical).abs() > tol {
            failed.push(format!(
                "pixel ({y},{x}) channel {c}: numerical {numerical:.4} vs analytical {analytical:.4}"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "loss gradient mismatches:\n  {}",
        failed.join("\n  ")
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn depth_loss_masks_invalid_pixels_and_backprops() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // 2x2 grid:
    // (0,0): valid gt=2.0, alpha=0.5, acc_depth=1.0 -> expected=2.0, err=0.0
    // (0,1): valid gt=1.5, alpha=0.8, acc_depth=2.0 -> expected=2.5, err=1.0
    // (1,0): invalid gt=0.0, alpha=0.9, acc_depth=9.0 -> masked out
    // (1,1): low alpha=0.005, gt=3.0, acc_depth=0.5 -> masked out
    let pred_depth =
        Tensor::<1>::from_floats([1.0, 2.0, 9.0, 0.5], &device)
            .reshape([2, 2])
            .require_grad();
    let pred_alpha =
        Tensor::<1>::from_floats([0.5, 0.8, 0.9, 0.005], &device)
            .reshape([2, 2])
            .require_grad();
    let gt_depth = Tensor::<1>::from_floats([2.0, 1.5, 0.0, 3.0], &device).reshape([2, 2]);

    let loss = brush_loss::depth_loss(pred_depth.clone(), pred_alpha.clone(), gt_depth);
    let loss_val = loss
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    // 2 valid pixels with errors 0.0 and 1.0 -> mean = 0.5
    assert!(
        (loss_val - 0.5).abs() < 1e-5,
        "expected masked mean loss 0.5, got {loss_val}"
    );

    let grads = loss.backward();
    let g_depth = to_vec(pred_depth.grad(&grads).expect("depth grad")).await;
    let g_alpha = to_vec(pred_alpha.grad(&grads).expect("alpha grad")).await;

    assert!(g_depth.iter().all(|v| v.is_finite()));
    assert!(g_alpha.iter().all(|v| v.is_finite()));
    // Only pixel (0, 1) has non-zero gradient: d/d(pred_depth) = +0.5 / 0.8 = 0.625.
    assert!((g_depth[1] - 0.625).abs() < 1e-4, "g_depth[1]={}", g_depth[1]);
    assert!(g_depth[2].abs() < 1e-6 && g_depth[3].abs() < 1e-6);
    assert!(g_alpha[2].abs() < 1e-6 && g_alpha[3].abs() < 1e-6);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn robust_depth_loss_suppresses_edges_ranges_and_uses_huber() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Pixel 0 is flat interior (neighbor also 1.0); pixels 1 & 2 straddle a 1.0 -> 3.0 depth edge.
    let pred_depth = Tensor::<1>::from_floats([1.5, 1.5, 3.5], &device)
        .reshape([1, 3])
        .require_grad();
    let pred_alpha = Tensor::<1>::from_floats([1.0, 1.0, 1.0], &device).reshape([1, 3]);
    let gt_depth = Tensor::<1>::from_floats([1.0, 1.0, 3.0], &device).reshape([1, 3]);

    let cfg = brush_loss::DepthLossConfig {
        huber_delta: 0.05,
        edge_threshold: 0.05,
        range_weight: true,
    };
    let loss =
        brush_loss::depth_loss_with_config(pred_depth.clone(), pred_alpha.clone(), gt_depth.clone(), cfg);
    let grads = loss.backward();
    let g_depth = to_vec(pred_depth.grad(&grads).expect("depth grad")).await;
    assert!(g_depth.iter().all(|v| v.is_finite()));
    // Edge pixels (1 & 2) must be heavily attenuated relative to flat interior pixel 0.
    assert!(
        g_depth[1].abs() < 0.05 * g_depth[0].abs() && g_depth[2].abs() < 0.05 * g_depth[0].abs(),
        "g_depth={g_depth:?}"
    );

    // Sub-delta error gets quadratic (smaller) Huber gradient vs saturated gradient above delta.
    let pred_small = Tensor::<1>::from_floats([1.01, 1.01, 3.0], &device)
        .reshape([1, 3])
        .require_grad();
    let loss_small = brush_loss::depth_loss_with_config(pred_small.clone(), pred_alpha, gt_depth, cfg);
    let grads_small = loss_small.backward();
    let g_small = to_vec(pred_small.grad(&grads_small).expect("small depth grad")).await;
    assert!(
        (g_small[0] / g_depth[0] - 0.2).abs() < 0.02,
        "expected Huber quadratic slope 0.01/0.05=0.2, got {}",
        g_small[0] / g_depth[0]
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn opacity_entropy_loss_peaks_at_half_and_vanishes_at_extremes() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let active = Tensor::<1>::from_floats([1.0, 1.0, 0.0], &device);
    let mid_raw = Tensor::<1>::from_floats([0.0, 0.0, 0.0], &device).require_grad();
    let sharp_raw = Tensor::<1>::from_floats([-10.0, 10.0, 0.0], &device);

    let l_mid = brush_loss::opacity_entropy_loss(mid_raw.clone(), active.clone())
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    let l_sharp = brush_loss::opacity_entropy_loss(sharp_raw, active.clone())
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    assert!((l_mid - std::f32::consts::LN_2).abs() < 1e-3, "l_mid={l_mid}");
    assert!(l_sharp < 2e-3, "l_sharp={l_sharp}");

    let grads = brush_loss::opacity_entropy_loss(mid_raw.clone(), active).backward();
    let g = to_vec(mid_raw.grad(&grads).expect("opac grad")).await;
    // Masked-out splat receives zero gradient; p=0.5 is a stationary maximum (grad ~ 0).
    assert!(g[2].abs() < 1e-6);
    assert!(g[0].abs() < 1e-5);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn scale_reg_penalizes_elongated_and_thick_splats() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Splat 0: well-behaved thin disk (0.1, 0.1, 0.01) -> aniso 0, flatness 0.1
    // Splat 1: needle (0.8, 0.04, 0.04) -> aniso 20-10 = 10, flatness 1.0
    let mut raw = vec![0.0f32; 2 * 10];
    raw[7..10].copy_from_slice(&[0.1f32.ln(), 0.1f32.ln(), 0.01f32.ln()]);
    raw[17..20].copy_from_slice(&[0.8f32.ln(), 0.04f32.ln(), 0.04f32.ln()]);
    let transforms = Tensor::<1>::from_floats(raw.as_slice(), &device)
        .reshape([2, 10])
        .require_grad();
    let active_disk = Tensor::<1>::from_floats([1.0, 0.0], &device);
    let active_needle = Tensor::<1>::from_floats([0.0, 1.0], &device);

    let l_disk = brush_loss::scale_regularization_loss(transforms.clone(), active_disk, 10.0, 0.5)
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    let l_needle =
        brush_loss::scale_regularization_loss(transforms.clone(), active_needle, 10.0, 0.5)
            .into_scalar_async::<f32>()
            .await
            .expect("readback");
    assert!(
        l_needle > l_disk * 50.0,
        "needle ({l_needle}) should be penalized much more than thin disk ({l_disk})"
    );
    let grads = brush_loss::scale_regularization_loss(
        transforms.clone(),
        Tensor::<1>::from_floats([1.0, 1.0], &device),
        10.0,
        0.5,
    )
    .backward();
    let g = to_vec(transforms.grad(&grads).expect("transform grad")).await;
    assert!(g.iter().all(|v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn normal_consistency_zero_for_fronto_parallel_and_positive_for_tilted() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Constant depth z=2.0 -> depth normals are (0, 0, -1), encoded RGB = (0.5, 0.5, 0.0).
    let depth = Tensor::<1>::from_floats([2.0f32; 16], &device).reshape([4, 4]);
    let alpha = Tensor::<1>::from_floats([1.0f32; 16], &device).reshape([4, 4]);

    let aligned_rgb: Vec<f32> = (0..16).flat_map(|_| [0.5f32, 0.5, 0.0]).collect();
    let tilted_rgb: Vec<f32> = (0..16).flat_map(|_| [1.0f32, 0.5, 0.5]).collect();
    let pred_aligned = Tensor::<1>::from_floats(aligned_rgb.as_slice(), &device)
        .reshape([4, 4, 3])
        .require_grad();
    let pred_tilted = Tensor::<1>::from_floats(tilted_rgb.as_slice(), &device)
        .reshape([4, 4, 3])
        .require_grad();

    let l_aligned = brush_loss::normal_consistency_loss(
        pred_aligned,
        alpha.clone(),
        depth.clone(),
        300.0,
        300.0,
        2.0,
        2.0,
        0.05,
    )
    .into_scalar_async::<f32>()
    .await
    .expect("readback");
    let loss_tilted = brush_loss::normal_consistency_loss(
        pred_tilted.clone(),
        alpha,
        depth,
        300.0,
        300.0,
        2.0,
        2.0,
        0.05,
    );
    let l_tilted = loss_tilted
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    assert!(l_aligned < 1e-4, "expected ~0 aligned normal loss, got {l_aligned}");
    assert!((l_tilted - 1.0).abs() < 1e-3, "expected ~1.0 orthogonal normal loss, got {l_tilted}");
    let grads = loss_tilted.backward();
    let g = to_vec(pred_tilted.grad(&grads).expect("normal grad")).await;
    assert!(g.iter().all(|v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn splat_sh0_camera_normals_faces_camera_and_differentiates_quat() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Identity quat -> local +Z = (0, 0, +1). Camera at (0, 0, -2) sees view_dir = (0, 0, -2),
    // so facing normal flips to (0, 0, -1). Identity camera rotation leaves n_cam = (0, 0, -1).
    let transforms = Tensor::<1>::from_floats([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -3.0, -3.0, -3.0], &device)
        .reshape([1, 10])
        .require_grad();
    let sh0 = brush_loss::splat_sh0_camera_normals(
        transforms.clone(),
        glam::Vec3::new(0.0, 0.0, -2.0),
        glam::Quat::IDENTITY,
        0.282_094_8,
    );
    let vals = to_vec(sh0.clone()).await;
    // Rendered color = sh0 * SH_C0 + 0.5 -> expect (0.5, 0.5, 0.0) so sh0.z = -0.5 / SH_C0.
    assert!(vals[0].abs() < 1e-5 && vals[1].abs() < 1e-5);
    assert!((vals[2] * 0.282_094_8 + 0.5).abs() < 1e-5, "vals={vals:?}");
    let grads = sh0.powi_scalar(2).sum().backward();
    let g = to_vec(transforms.grad(&grads).expect("quat grad")).await;
    assert!(g.iter().all(|v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn image_gradient_loss_tracks_edges() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // Left half black, right half white in GT.
    let black = 255u32 << 24;
    let white = 0xff_ff_ff_ffu32;
    let packed = Tensor::<1, burn::tensor::Int>::from_ints(
        [black as i32, white as i32, black as i32, white as i32],
        &device.clone().inner(),
    )
    .reshape([2, 2]);
    let flat_pred = Tensor::<1>::from_floats([0.5; 12], &device)
        .reshape([2, 2, 3])
        .require_grad();
    let loss = brush_loss::image_gradient_loss(flat_pred.clone(), packed, None);
    let l = loss
        .clone()
        .into_scalar_async::<f32>()
        .await
        .expect("readback");
    assert!(l > 0.5, "expected positive edge gradient loss, got {l}");
    let grads = loss.backward();
    let g = to_vec(flat_pred.grad(&grads).expect("rgb grad")).await;
    assert!(g.iter().all(|v| v.is_finite()));
}

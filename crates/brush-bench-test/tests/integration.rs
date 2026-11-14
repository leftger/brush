//! Integration tests for the benchmark functions
//!
//! These tests verify that the benchmark data generation and core operations work correctly.

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::bwd::render_splats;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::module::Module;
use burn::tensor::{Device, TensorData};
use glam::{Quat, Vec3};
use rand::{RngExt, SeedableRng};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const TEST_SEED: u64 = 12345;

/// Generate small realistic splats for testing
fn generate_test_splats(device: &Device, count: usize) -> Splats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);

    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();

    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base = rng.random_range(0.01..0.1_f32).ln();
            [base, base, base]
        })
        .collect();

    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let u1 = rng.random::<f32>();
            let u2 = rng.random::<f32>();
            let u3 = rng.random::<f32>();

            let sqrt1_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let theta1 = 2.0 * std::f32::consts::PI * u2;
            let theta2 = 2.0 * std::f32::consts::PI * u3;

            [
                sqrt1_u1 * theta1.sin(),
                sqrt1_u1 * theta1.cos(),
                sqrt_u1 * theta2.sin(),
                sqrt_u1 * theta2.cos(),
            ]
        })
        .collect();

    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
            ]
        })
        .collect();

    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.6..1.0)).collect();

    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn generate_test_batch(resolution: (u32, u32)) -> SceneBatch {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let (width, height) = resolution;
    let pixel_count = (width * height) as usize;

    let mut byte = |v: f32| -> u32 {
        let v = (v + (rng.random::<f32>() - 0.5) * 0.05).clamp(0.0, 1.0);
        (v * 255.0).round() as u32
    };
    let img_packed_data: Vec<i32> = (0..pixel_count)
        .map(|i| {
            let x = (i as u32) % width;
            let y = (i as u32) / width;
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let r = byte(nx * 0.5 + 0.25);
            let g = byte(ny * 0.5 + 0.25);
            let b = byte((nx + ny) * 0.25 + 0.5);
            (r | g << 8 | b << 16 | 255 << 24) as i32
        })
        .collect();

    let img_packed = TensorData::new(img_packed_data, [height as usize, width as usize]);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    SceneBatch {
        img_packed,
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
        depth: None,
        view_index: 0,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_splat_generation() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 1000);

    assert_eq!(splats.num_splats(), 1000);

    // Check that means are reasonable
    let means_data = splats
        .means()
        .into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .unwrap();
    assert_eq!(means_data.len(), 3000);

    for chunk in means_data.chunks(3) {
        assert!(chunk.iter().all(|&x| x.is_finite()));
        assert!(chunk[0].abs() < 10.0 && chunk[1].abs() < 10.0 && chunk[2].abs() < 20.0);
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_forward_rendering() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 1000);
    assert_eq!(splats.num_splats(), 1000);

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -8.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let img_size = glam::uvec2(64, 64);
    let result = render_splats(splats, &camera, img_size, Vec3::ZERO).await;
    assert!(result.num_visible > 0, "no splats rendered");
    let data = result
        .img
        .into_data_async()
        .await
        .expect("readback")
        .try_into_vec::<f32>()
        .expect("Wrong type");
    assert!(data.iter().all(|&v| v.is_finite()));
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_training_step() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let splats = generate_test_splats(&device, 500);
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    let (final_splats, _stats) = trainer.step(batch, splats).await;

    assert!(final_splats.num_splats() > 0);
}

#[wasm_bindgen_test(unsupported = test)]
fn test_batch_generation() {
    let batch = generate_test_batch((256, 128));
    let img_dims = batch.img_packed.shape.as_slice();
    assert_eq!(img_dims, &[128, 256]);
    let img_data = batch.img_packed.try_into_vec::<i32>().unwrap();
    assert_eq!(img_data.len(), 128 * 256);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_multi_step_training() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig::default();
    let mut splats = generate_test_splats(&device, 100);
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    for _ in 0..10 {
        let (new_splats, _) = trainer.step(batch.clone(), splats).await;
        splats = new_splats;
    }
    assert!(splats.num_splats() > 0);
}

// Training with a camera pointing away from every splat — num_visible == 0
// every step. The training loop must not crash on this; all gradients should
// be zero (or at least finite) and the optimizer step should be a no-op.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_with_zero_visible_does_not_crash() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 200);

    // Camera pointing away from the scene (looking along +Z, scene is at ±5).
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 1000.0),
        Quat::IDENTITY, // looks along -Z in local space → away from origin
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );

    let pixel = 0x80808080u32 as i32; // mid-grey, opaque; bit-cast to i32 for the dispatch backend
    let batch = SceneBatch {
        img_packed: TensorData::new(vec![pixel; 64 * 64], [64usize, 64]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera,
        depth: None,
        view_index: 0,
    };

    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (new_splats, _stats) = trainer.step(batch, splats).await;
    // Should succeed; nothing visible means num_visible ≈ 0.
    assert!(new_splats.num_splats() > 0);
}

// Training with a deliberately degenerate bounding box (NaN center) used to
// crash in `median_size`. After the fix, training must still proceed without
// panicking — the per-step `lr_mean * median_size` just ends up NaN, which is
// ultimately harmless because any NaN optimizer update is caught by
// `validate_gradient` under the debug-validation feature.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn trainer_tolerates_nan_bounds() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 100);
    let config = TrainConfig::default();

    // A degenerate bounds with one NaN axis. Before the `total_cmp` fix, this
    // panicked inside `median_size()` on the first `step` call.
    let bounds = BoundingBox {
        center: Vec3::ZERO,
        extent: Vec3::new(f32::NAN, 1.0, 1.0),
    };
    let mut trainer = SplatTrainer::new(&config, &device, bounds);
    let batch = generate_test_batch((64, 64));
    let (_splats, _stats) = trainer.step(batch, splats).await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn test_gradient_validation() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = generate_test_splats(&device, 100);

    // Create a simple loss by rendering and taking the mean
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let img_size = glam::uvec2(64, 64);

    // Clone splats since render_splats takes ownership and we need splats for gradient validation
    let result = render_splats(splats.clone(), &camera, img_size, Vec3::ZERO).await;
    splats.bwd_validate(result.img.mean()).await;
}

// One trainer + many parallel viewers.
// Test whether multithreading produces any issues.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "multi_thread")]
async fn stress_concurrent_train_and_view() {
    use brush_async::Actor;
    use brush_render::TextureMode;
    use brush_render::gaussian_splats::render_splats as render_splats_fwd;
    use tokio::sync::watch;

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let img_size = glam::uvec2(64, 64);

    let viewer_count = 6;
    let train_steps = 100;
    let viewer_iters_per_task = 10;

    let initial = generate_test_splats(&device, 500);
    let (tx, rx) = watch::channel::<Splats>(initial.clone().valid());

    let trainer_actor = Actor::new("test-trainer");
    let device_c = device.clone();
    let mut splats = initial;
    let trainer_done = trainer_actor.run(move || async move {
        let batch = generate_test_batch((64, 64));
        let config = TrainConfig::default();
        let mut trainer = SplatTrainer::new(
            &config,
            &device_c,
            BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
        );
        for _ in 0..train_steps {
            let (new_splats, _) = trainer.step(batch.clone(), splats).await;
            splats = new_splats;
            let _ = tx.send(splats.valid());
        }
    });

    let mut viewer_actors = Vec::with_capacity(viewer_count);
    let mut viewer_dones = Vec::with_capacity(viewer_count);
    for v in 0..viewer_count {
        let actor = Actor::new(&format!("test-viewer-{v}"));
        let mut rx = rx.clone();
        let done = actor.run(move || async move {
            let camera = Camera::new(
                Vec3::new(0.0, 0.0, -5.0 - v as f32 * 0.3),
                Quat::IDENTITY,
                45.0,
                45.0,
                glam::vec2(0.5, 0.5),
                Pinhole,
            );
            for _ in 0..viewer_iters_per_task {
                let snap = rx.borrow_and_update().clone();
                render_splats_fwd(
                    snap,
                    &camera,
                    img_size,
                    Vec3::ZERO,
                    None,
                    TextureMode::Float,
                )
                .await;
            }
        });
        viewer_actors.push(actor);
        viewer_dones.push(done);
    }

    trainer_done.await;
    for d in viewer_dones {
        d.await;
    }
    drop(viewer_actors);
}

async fn all_finite(splats: &Splats) -> bool {
    for t in [
        splats.means().into_data_async().await.unwrap(),
        splats.log_scales().into_data_async().await.unwrap(),
    ] {
        if !t.iter::<f32>().all(f32::is_finite) {
            return false;
        }
    }
    splats
        .opacities()
        .into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .all(f32::is_finite)
}

// A run of one iteration (or a degenerate zero) must not poison the LR decay
// (exponent 1 / iters) or the refine schedule (iter / iters) with inf or NaN.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn tiny_training_runs_stay_finite() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    for total_train_iters in [0, 1] {
        let config = TrainConfig {
            total_train_iters,
            ..TrainConfig::default()
        };
        let mut trainer = SplatTrainer::new(
            &config,
            &device,
            BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
        );
        // Like the training loop: splats live on the inner device and are
        // lifted to autodiff for each step, so refine sees plain leaves.
        let mut splats = generate_test_splats(&device, 100).valid();
        for iter in 0..3 {
            let (new_splats, _) = trainer.step(batch.clone(), splats.train()).await;
            let (new_splats, _) = trainer.refine(iter, new_splats.valid()).await;
            splats = new_splats;
        }
        assert!(
            all_finite(&splats).await,
            "non-finite params with total_train_iters = {total_train_iters}"
        );
    }
}

// Gradient-driven growth only runs inside [growth_start_iter, growth_stop_iter).
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn growth_waits_for_start_iter() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let batch = generate_test_batch((64, 64));
    // Every visible splat with any gradient qualifies, so growth is only
    // absent when the gate says so.
    let config = TrainConfig {
        growth_grad_threshold: 0.0,
        growth_select_fraction: 1.0,
        growth_start_iter: 1000,
        growth_stop_iter: 2000,
        split_at_screen_size: 0.0,
        ..TrainConfig::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );

    let mut splats = generate_test_splats(&device, 100).valid();
    for _ in 0..5 {
        let (new_splats, _) = trainer.step(batch.clone(), splats.train()).await;
        splats = new_splats.valid();
    }
    let (splats, before) = trainer.refine(500, splats).await;
    assert_eq!(before.num_split_high_grad, 0);

    let mut splats = splats;
    for _ in 0..5 {
        let (new_splats, _) = trainer.step(batch.clone(), splats.train()).await;
        splats = new_splats.valid();
    }
    let (splats, inside) = trainer.refine(1500, splats).await;
    assert!(inside.num_split_high_grad > 0);

    let mut splats = splats;
    for _ in 0..5 {
        let (new_splats, _) = trainer.step(batch.clone(), splats.train()).await;
        splats = new_splats.valid();
    }
    let (_, after) = trainer.refine(2500, splats).await;
    assert_eq!(after.num_split_high_grad, 0);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn train_step_with_depth_supervision_is_finite() {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut batch = generate_test_batch((64, 64));
    let mut depth_vals = vec![2.5f32; 64 * 64];
    for (i, v) in depth_vals.iter_mut().enumerate() {
        if i % 5 == 0 {
            *v = 0.0;
        }
    }
    batch.depth = Some(TensorData::new(depth_vals, [64, 64]));

    let config = TrainConfig {
        depth_loss_weight: 0.25,
        ..TrainConfig::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE),
    );
    let mut splats = generate_test_splats(&device, 100).valid();
    for iter in 0..5 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats.train()).await;
        let loss_scalar = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(loss_scalar.is_finite(), "non-finite loss at iter {iter}");
        let (refined, _) = trainer.refine(iter, new_splats.valid()).await;
        splats = refined;
    }
    assert!(all_finite(&splats).await);
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn pose_refinement_recovers_perturbed_camera_pose() {
    use brush_render::bwd::render_splats_with_options;
    use brush_render::gaussian_splats::RasterPass;

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut splats = generate_test_splats(&device, 128).valid();
    // Make splats visible and high-contrast at z ~ 0 inside the frustum.
    let true_cam = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        0.8,
        0.8,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let gt_out = render_splats_with_options(
        splats.clone().train(),
        &true_cam,
        glam::uvec2(64, 64),
        Vec3::ZERO,
        RasterPass::Backward,
        true,
        false,
    )
    .await;
    assert!(gt_out.num_visible > 0);

    let gt_floats: Vec<f32> = gt_out
        .img
        .inner()
        .into_data_async()
        .await
        .expect("gt render readback")
        .try_into_vec()
        .expect("f32 gt image");
    let gt_alpha: Vec<f32> = gt_floats.chunks_exact(4).map(|px| px[3]).collect();
    let gt_depth_raw: Vec<f32> = gt_out
        .depth
        .expect("gt depth")
        .inner()
        .into_data_async()
        .await
        .expect("gt depth readback")
        .try_into_vec()
        .expect("f32 gt depth");
    // Ground-truth metric depth is expected_depth = pred_depth / pred_alpha.
    let gt_depth: Vec<f32> = gt_depth_raw
        .into_iter()
        .zip(gt_alpha.into_iter())
        .map(|(d, a)| if a > 0.05 { (d / a).max(0.0) } else { 0.0 })
        .collect();
    let packed: Vec<i32> = gt_floats
        .chunks_exact(4)
        .map(|px| {
            let r = (px[0].clamp(0.0, 1.0) * 255.0).round() as u32;
            let g = (px[1].clamp(0.0, 1.0) * 255.0).round() as u32;
            let b = (px[2].clamp(0.0, 1.0) * 255.0).round() as u32;
            (r | (g << 8) | (b << 16) | (255 << 24)) as i32
        })
        .collect();

    let perturbed_cam = Camera::new(
        true_cam.position + Vec3::new(0.06, -0.04, 0.06),
        (Quat::from_rotation_z(0.05) * true_cam.rotation).normalize(),
        0.8,
        0.8,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let batch = SceneBatch {
        img_packed: TensorData::new(packed, [64usize, 64]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera: perturbed_cam,
        depth: Some(TensorData::new(gt_depth, [64usize, 64])),
        view_index: 7,
    };

    // Freeze splats (all splat LRs = 0) so only per-view SE(3) pose refinement optimizes.
    let config = TrainConfig {
        lr_mean: 0.0,
        lr_mean_end: 0.0,
        lr_rotation: 0.0,
        lr_scale: 0.0,
        lr_coeffs_dc: 0.0,
        lr_opac: 0.0,
        opac_entropy_weight: 0.0,
        mean_noise_weight: 0.0,
        background_noise_strength: 0.0,
        depth_loss_weight: 0.25,
        lr_pose: 5e-3,
        total_train_iters: 150,
        ..TrainConfig::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );

    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for _ in 0..90 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats.train()).await;
        let l = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        if first_loss.is_none() {
            first_loss = Some(l);
        }
        last_loss = l;
        splats = new_splats.valid();
    }

    let first_loss = first_loss.unwrap();
    assert!(
        last_loss < first_loss * 0.5,
        "pose refinement did not strongly reduce photometric loss: {first_loss:.5} -> {last_loss:.5}"
    );
    let (dp, dq) = trainer
        .view_pose_delta(7)
        .expect("pose delta should be recorded for view 7");
    let init_pos_err = (perturbed_cam.position - true_cam.position).length();
    let final_pos_err = ((perturbed_cam.position + dp) - true_cam.position).length();
    assert!(
        final_pos_err < init_pos_err * 0.6,
        "pose translation did not converge toward GT: init_err={init_pos_err:.4}, final_err={final_pos_err:.4}, dp={dp:?}"
    );
    let init_ang_err = perturbed_cam.rotation.angle_between(true_cam.rotation);
    let final_ang_err = (dq * perturbed_cam.rotation).angle_between(true_cam.rotation);
    assert!(
        final_ang_err < init_ang_err * 0.6,
        "pose rotation did not converge toward GT: init_ang={init_ang_err:.4}, final_ang={final_ang_err:.4}"
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn appearance_and_normal_consistency_train_cleanly() {
    use brush_render::bwd::render_splats_with_options;
    use brush_render::gaussian_splats::RasterPass;

    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut splats = generate_test_splats(&device, 64).with_sh_degree(2).valid();
    let cam = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        0.8,
        0.8,
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let gt_out = render_splats_with_options(
        splats.clone().train(),
        &cam,
        glam::uvec2(64, 64),
        Vec3::ZERO,
        RasterPass::Backward,
        true,
        false,
    )
    .await;
    let gt_floats: Vec<f32> = gt_out
        .img
        .inner()
        .into_data_async()
        .await
        .expect("gt render readback")
        .try_into_vec()
        .expect("f32 gt image");
    let gt_alpha: Vec<f32> = gt_floats.chunks_exact(4).map(|px| px[3]).collect();
    let gt_depth_raw: Vec<f32> = gt_out
        .depth
        .expect("gt depth")
        .inner()
        .into_data_async()
        .await
        .expect("gt depth readback")
        .try_into_vec()
        .expect("f32 gt depth");
    let gt_depth: Vec<f32> = gt_depth_raw
        .into_iter()
        .zip(gt_alpha)
        .map(|(d, a)| if a > 0.05 { (d / a).max(0.0) } else { 0.0 })
        .collect();
    // Darken view RGB by 0.65x to simulate auto-exposure shift.
    let packed: Vec<i32> = gt_floats
        .chunks_exact(4)
        .map(|px| {
            let r = ((px[0] * 0.65).clamp(0.0, 1.0) * 255.0).round() as u32;
            let g = ((px[1] * 0.65).clamp(0.0, 1.0) * 255.0).round() as u32;
            let b = ((px[2] * 0.65).clamp(0.0, 1.0) * 255.0).round() as u32;
            (r | (g << 8) | (b << 16) | (255 << 24)) as i32
        })
        .collect();

    let batch = SceneBatch {
        img_packed: TensorData::new(packed, [64usize, 64]),
        has_alpha: false,
        alpha_mode: AlphaMode::Transparent,
        camera: cam,
        depth: Some(TensorData::new(gt_depth, [64usize, 64])),
        view_index: 9,
    };

    let config = TrainConfig {
        background_noise_strength: 0.0,
        mean_noise_weight: 0.0,
        lr_pose: 0.0,
        lr_appearance: 4e-3,
        appearance_reg_weight: 0.0,
        normal_loss_weight: 0.05,
        scale_reg_weight: 0.01,
        grad_loss_weight: 0.05,
        sh_warmup_every: 5,
        total_train_iters: 40,
        ..TrainConfig::default()
    };
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );

    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for _ in 0..20 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats.train()).await;
        let l = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(l.is_finite(), "non-finite loss {l}");
        if first_loss.is_none() {
            first_loss = Some(l);
        }
        last_loss = l;
        splats = new_splats.valid();
    }
    assert!(
        last_loss < first_loss.unwrap(),
        "loss did not decrease: {} -> {last_loss}",
        first_loss.unwrap()
    );
    let (mat, _bias) = trainer
        .view_appearance(9)
        .expect("appearance state recorded for view 9");
    let diag_mean = (mat[0] + mat[4] + mat[8]) / 3.0;
    assert!(
        diag_mean < 0.98,
        "expected affine appearance diagonal to dim below 1.0, got {diag_mean:.4}"
    );
}

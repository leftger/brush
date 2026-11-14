use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-7")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50.0")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is only an upper bound, the actual final number of splats is NOT determined by this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.0025")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    pub growth_select_fraction: f32,

    /// Iteration at which splat growth starts. Clamped to `growth_stop_iter`.
    #[arg(long, help_heading = "Refine options", default_value = "0")]
    pub growth_start_iter: u32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Split any splat whose max screen-space extent exceeds this fraction of
    /// the image dimension, shrinking the children so they land at (at most)
    /// this size on screen. 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_at_screen_size: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Mip-Splatting 3D-filter strength (the paper's `s`): each splat gets a
    /// per-splat world-space scale floor `sqrt(min_scale_factor) · pixel size
    /// at the nearest observing camera`, i.e. a ~0.32px std-dev floor at the
    /// default. Folded into scales/opacity at render (and baked at export),
    /// never optimized. 0 disables the filter.
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub min_scale_factor: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    /// Weight of L1/Huber depth loss when ground-truth depth maps are present in the dataset.
    /// Set to 0 to disable depth supervision.
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub depth_loss_weight: f32,

    /// Cosine-decay multiplier reached by `depth_loss_weight` at `total_train_iters` (1.0 keeps constant).
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub depth_loss_min_scale: f32,

    /// Huber transition delta (metres) for depth loss; 0 uses pure L1.
    #[arg(long, help_heading = "Training options", default_value = "0.05")]
    pub depth_huber_delta: f32,

    /// Relative GT depth step threshold for downweighting depth silhouettes; 0 disables edge masking.
    #[arg(long, help_heading = "Training options", default_value = "0.05")]
    pub depth_edge_threshold: f32,

    /// Disable `1 / (1 + depth)` inverse-range weighting on depth loss.
    #[arg(long, help_heading = "Training options", default_value = "false")]
    pub no_depth_range_weight: bool,

    /// Weight of surface-normal consistency loss aligning splat +Z normals with rendered depth normals.
    #[arg(long, help_heading = "Refine options", default_value = "0.05")]
    pub normal_loss_weight: f32,

    /// Also run surface-normal consistency on RGB-only views without ground-truth depth.
    #[arg(long, help_heading = "Refine options", default_value = "false")]
    pub normal_loss_rgb: bool,

    /// Weight of splat flatness and max-anisotropy regularization; 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.01")]
    pub scale_reg_weight: f32,

    /// Maximum allowed ratio between a splat's largest and middle axis before anisotropy penalty applies.
    #[arg(long, help_heading = "Refine options", default_value = "10.0")]
    pub scale_max_ratio: f32,

    /// Weight of high-frequency finite-difference image gradient L1 loss; 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "0.05")]
    pub grad_loss_weight: f32,

    /// Weight of binary-opacity entropy regularization pushing splats toward 0 or 1 opacity.
    #[arg(long, help_heading = "Refine options", default_value = "0.0001")]
    pub opac_entropy_weight: f32,

    /// Learning rate for per-view `se(3)` camera pose refinement; 0 disables pose optimization.
    #[arg(long, help_heading = "Refine options", default_value = "1e-4")]
    pub lr_pose: f64,

    /// Learning rate for per-view 3x3 affine appearance correction; 0 disables appearance optimization.
    #[arg(long, help_heading = "Refine options", default_value = "5e-4")]
    pub lr_appearance: f64,

    /// L2 regularization pulling per-view affine appearance toward identity.
    #[arg(long, help_heading = "Refine options", default_value = "0.01")]
    pub appearance_reg_weight: f32,

    /// Iterations per unlocked SH degree band during warmup; 0 unlocks all bands immediately.
    #[arg(long, help_heading = "Training options", default_value = "1000")]
    pub sh_warmup_every: u32,

    /// Iterations for coarse-to-fine image resolution warmup (>= 256px views only); 0 disables.
    #[arg(long, help_heading = "Training options", default_value = "3000")]
    pub coarse_warmup_iters: u32,

    /// Starting resolution scale factor during coarse-to-fine warmup (in (0.1, 1.0]).
    #[arg(long, help_heading = "Training options", default_value = "0.5")]
    pub coarse_scale_start: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Iteration after which LPIPS loss activates when `lpips_loss_weight > 0` (0 starts immediately).
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub lpips_start_iter: u32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }
}

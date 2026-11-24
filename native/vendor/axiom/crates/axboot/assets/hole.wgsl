enable ax_volume_sampling;
// The welcome's black hole: the first of its two passes. This one draws the
// hole alone, at the lens map's resolution on displays larger than it; the
// runtime upsamples it to the display and `compose.wgsl` lays the overlay
// over it at the display's own resolution, so text stays sharp while the
// hole costs the same on any panel.
//
// The geodesics are traced once, at build time (`build/kerr.rs`), from a
// camera at azimuth 0, so every pixel's path through the metric is fixed.
// Each pixel arrives with that trace's result — where its escaping rays meet
// the sky (a point in the baked sky atlas, at the tier prefiltered to its
// footprint) and the first two disk crossings along its ray (coverage, ln r,
// φ, g) — and this pass composes them. The camera swings by `orbit` around the
// spin axis; metric and disk are symmetric about it, so the swing is the
// scene turned by `orbit`: the sky lookup moves along the atlas row and the
// disk's azimuth turns with it. The disk's emission comes from r and its
// redshift g, its turbulence co-rotating with the gas (a lookup at
// φ − Ω(r)·t in the baked polar table), front to back over the sky, then
// tone mapping and the scrim the overlay's text sits on. Constants the build
// derives (ISCO, ln r range, the atlas's tiers) are prepended to this source
// by build.rs.
@group(0) @binding(0) var<uniform> viewport: vec2<f32>;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> orbit: f32;
@group(1) @binding(0) var<storage, read> turbulence: AxRgbVolume;
@group(1) @binding(1) var<storage, read> sky_atlas: AxRgbVolume;

const TAU: f32 = 6.28318530717959;

// Blackbody-ish tint for a normalised temperature (cool red → white-blue).
fn thermal(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(1.00, 0.30, 0.08);
    let c1 = vec3<f32>(1.00, 0.72, 0.40);
    let c2 = vec3<f32>(1.00, 0.96, 0.90);
    let c3 = vec3<f32>(0.80, 0.88, 1.00);
    let low = mix(c0, c1, clamp(t / 0.33, 0.0, 1.0));
    let mid = mix(c1, c2, clamp((t - 0.33) / 0.33, 0.0, 1.0));
    let high = mix(c2, c3, clamp((t - 0.66) / 0.34, 0.0, 1.0));
    return select(select(high, mid, t < 0.66), low, t < 0.33);
}

// Disk light at radius r seen with redshift g: a Novikov–Thorne-shaped
// temperature normalised to its peak, I_obs = g⁴ I_emit. Fractional powers
// are square roots and products, never `pow`: the CPU plan runs `sqrt` as a
// vector kernel and `pow` one element at a time.
fn emission(r: f32, g: f32) -> vec3<f32> {
    let root = sqrt(r);
    let inner = sqrt(max(1.0 - sqrt(DISK_INNER) / root, 0.0));
    let tn = sqrt(inner) / (root * sqrt(root) * 0.17);
    let g2 = g * g;
    let tint = thermal(clamp(0.42 * tn * g, 0.0, 1.0)) * vec3<f32>(1.0, 0.80, 0.58);
    return tint * (tn * tn * tn) * (g2 * g2) * DISK_GAIN;
}

// Radial falloff at both edges of the disk.
fn edge(r: f32) -> f32 {
    return smoothstep(DISK_INNER, DISK_INNER + 0.6, r) * (1.0 - smoothstep(DISK_OUTER * 0.5, DISK_OUTER, r));
}

// Where the turbulence table holds the gas now at (r, φ), φ already turned
// by the camera's swing: the pattern co-rotates at the local Keplerian Ω, so
// each radius shears past the next. The table's rows run in ln r, which the
// crossing already carries: no logarithm here.
fn turbulence_point(r: f32, ln_r: f32, phi: f32) -> vec3<f32> {
    let omega = 1.0 / (r * sqrt(r) + SPIN);
    let u = fract((phi - omega * time * 0.6) / TAU);
    let row = (ln_r - LOG_R_MIN) / (LOG_R_MAX - LOG_R_MIN) * (TURBULENCE_ROWS - 1.0);
    return vec3<f32>(u * TURBULENCE_COLUMNS, row, 0.0);
}

fn density(r: f32, n: vec3<f32>) -> f32 {
    return edge(r) * (0.45 + 1.1 * n.x * n.x + 0.5 * n.y);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Where the pixel's sky is with the camera swung by `orbit`: its atlas point
// moves along the row by the swing in its tier's texels. The bake padded
// every tier by the full swing, so the lookup never leaves its tier.
fn swung(point: vec3<f32>) -> vec3<f32> {
    let small = select(SKY_ULTRA_PER_RADIAN, SKY_COARSE_PER_RADIAN, point.x < SKY_COARSE_END);
    let upper = select(small, SKY_FINE_PER_RADIAN, point.x < SKY_FINE_END);
    let per_radian = select(SKY_MEDIUM_PER_RADIAN, upper, point.y < SKY_FINE_ROWS);
    return vec3<f32>(point.x + orbit * per_radian, point.y, 0.0);
}

// sRGB encoding without `pow`: x^(1/2.4) over the power segment as a fit in
// √x, ⁴√x, ⁸√x and x — three vector square roots — within 0.13 of an 8-bit
// step of the exact curve.
fn encode_srgb(x: vec3<f32>) -> vec3<f32> {
    let c = max(x, vec3<f32>(0.0031308));
    let s = sqrt(c);
    let q = sqrt(s);
    let e = sqrt(q);
    let root = 0.738108353 * s + 0.396679043 * q - 0.042969377 * c - 0.092308645 * e;
    return select(1.055 * root - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

@fragment
fn main(@builtin(position) position: vec4<f32>,
        @location(0) sky: vec3<f32>,
        @location(1) near: vec4<f32>,
        @location(2) far: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = min(viewport.x / 1920.0, viewport.y / 1200.0);
    let size = viewport / scale;
    let p = position.xy / scale;

    // ── The hole ─────────────────────────────────────────────────────────
    // Crossings arrive as ln r (decoded by table, not per pixel); `exp` runs
    // on the vector path.
    let near_ln_r = max(near.y, LOG_R_MIN);
    let far_ln_r = max(far.y, LOG_R_MIN);
    let near_r = exp(near_ln_r);
    let far_r = exp(far_ln_r);
    let backdrop = ax_sample_volume_rgb(&sky_atlas, swung(sky)) * sky.z;
    let near_noise = ax_sample_volume_rgb(&turbulence, turbulence_point(near_r, near_ln_r, near.z + orbit));
    let far_noise = ax_sample_volume_rgb(&turbulence, turbulence_point(far_r, far_ln_r, far.z + orbit));
    let near_density = density(near_r, near_noise);
    let far_density = density(far_r, far_noise);
    let near_alpha = clamp(near_density * 0.75, 0.0, 1.0) * near.x;
    let far_alpha = clamp(far_density * 0.75, 0.0, 1.0) * far.x;
    let near_light = emission(near_r, near.w) * near_density * near.x;
    let far_light = emission(far_r, far.w) * far_density * far.x;
    let through_near = 1.0 - near_alpha;
    let light = near_light + through_near * far_light + through_near * (1.0 - far_alpha) * backdrop;
    let hole = encode_srgb(aces(light));
    // A soft scrim under the overlay's lower left keeps its text legible over
    // the brightest lensed sky.
    let corner = clamp(1.0 - length((p - vec2<f32>(0.0, size.y)) / vec2<f32>(1150.0, 760.0)), 0.0, 1.0);
    let c0 = hole * (1.0 - 0.38 * corner * corner);
    return vec4<f32>(clamp(c0, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// Kore shell study, adapted from the user-supplied kore-shell-wgsl.html.
// The host specializes layout/palette constants and unrolls the SDF paints.
// No pixels are evaluated by the host. Coverage is one *physical* pixel wide.
// This v0 study uses sRGB paint values and translucent tints, not the original
// HDR / sampled-backdrop blur / dither pipeline.
@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let viewport = vec2<f32>(__WIDTH__, __HEIGHT__);
    let scale = min(viewport.x / 1200.0, viewport.y / 800.0);
    let p = (position.xy - (viewport - vec2<f32>(1200.0, 800.0) * scale) * 0.5) / scale;
    let warm = 1.0 - smoothstep(0.0, 1.0, length((p - vec2<f32>(1050.0, 60.0)) / vec2<f32>(1150.0, 850.0)));
    let sage = 1.0 - smoothstep(0.0, 1.0, length((p - vec2<f32>(50.0, 790.0)) / vec2<f32>(1050.0, 740.0)));
    let pale = 1.0 - smoothstep(0.0, 1.0, length((p - vec2<f32>(520.0, 330.0)) / vec2<f32>(550.0, 550.0)));
    let ground_a = mix(__GROUND__, __WARM__, warm * 0.62);
    let ground_b = mix(ground_a, __SAGE__, sage * 0.55);
    let ground_c = mix(ground_b, __PALE__, pale * 0.38);
    let disc = clamp(0.5 - (length(p - vec2<f32>(180.0, 970.0)) - 400.0) * scale, 0.0, 1.0);
    let ring = clamp(0.5 - (abs(length(p - vec2<f32>(1005.0, 660.0)) - 174.0) - 9.0) * scale, 0.0, 1.0);
    let ground_d = mix(ground_c, __SAGE__, disc * 0.28);
    let c0 = mix(ground_d, __PALE__, ring * 0.45);
    __PAINTS__
    return vec4<f32>(__FINAL__, 1.0);
}

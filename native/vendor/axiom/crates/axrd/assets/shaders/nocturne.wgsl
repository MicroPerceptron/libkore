// A branchless ax-shader-v0 scene. The Rust host replaces the two viewport
// tokens before compiling this embedded source with concrete tensor shapes.
@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let viewport = vec2<f32>(__VIEWPORT_WIDTH__, __VIEWPORT_HEIGHT__);
    let design = vec2<f32>(960.0, 600.0);
    let scale = min(viewport.x / design.x, viewport.y / design.y);
    let origin = (viewport - design * scale) * 0.5;
    let p = (position.xy - origin) / scale;
    let uv = position.xy / viewport;

    // Deep olive room light with two soft sources.
    let background = vec3<f32>(0.035, 0.052, 0.044)
        + vec3<f32>(0.020, 0.032, 0.026) * (1.0 - uv.y);
    let coral_glow = clamp(1.0 - length(p - vec2<f32>(790.0, 95.0)) / 520.0, 0.0, 1.0);
    let sage_glow = clamp(1.0 - length(p - vec2<f32>(165.0, 510.0)) / 480.0, 0.0, 1.0);
    let room = background
        + vec3<f32>(0.105, 0.030, 0.016) * coral_glow * coral_glow
        + vec3<f32>(0.026, 0.085, 0.047) * sage_glow * sage_glow;

    // Retained glass surfaces. Linear one-pixel coverage keeps this demo on
    // the representative arithmetic subset already executable by AxHetero.
    let shell = clamp(
        min(408.0 - abs(p.x - 480.0), 248.0 - abs(p.y - 300.0)),
        0.0,
        1.0,
    );
    let after_shell = room
        + (vec3<f32>(0.081, 0.100, 0.084) - room) * shell * 0.96;

    let header = clamp(
        min(408.0 - abs(p.x - 480.0), 43.0 - abs(p.y - 101.0)),
        0.0,
        1.0,
    ) * shell;
    let rail = clamp(
        min(82.0 - abs(p.x - 194.0), 164.0 - abs(p.y - 337.0)),
        0.0,
        1.0,
    ) * shell;
    let chrome = clamp(header + rail, 0.0, 1.0);
    let after_chrome = after_shell
        + (vec3<f32>(0.056, 0.073, 0.061) - after_shell) * chrome * 0.88;

    let primary = clamp(
        min(174.0 - abs(p.x - 487.0), 143.0 - abs(p.y - 318.0)),
        0.0,
        1.0,
    ) * shell;
    let secondary = clamp(
        min(94.0 - abs(p.x - 735.0), 143.0 - abs(p.y - 318.0)),
        0.0,
        1.0,
    ) * shell;
    let after_primary = after_chrome
        + (vec3<f32>(0.108, 0.130, 0.108) - after_chrome) * primary * 0.91;
    let after_cards = after_primary
        + (vec3<f32>(0.091, 0.113, 0.094) - after_primary) * secondary * 0.94;

    // Sage status geometry: selected destination, online light, and activity
    // bars all share one analytical paint.
    let selected = clamp(
        min(58.0 - abs(p.x - 194.0), 17.0 - abs(p.y - 258.0)),
        0.0,
        1.0,
    );
    let status = clamp(8.0 - length(p - vec2<f32>(766.0, 101.0)), 0.0, 1.0);
    let bar_a = clamp(min(7.0 - abs(p.x - 695.0), 29.0 - abs(p.y - 370.0)), 0.0, 1.0);
    let bar_b = clamp(min(7.0 - abs(p.x - 720.0), 47.0 - abs(p.y - 352.0)), 0.0, 1.0);
    let bar_c = clamp(min(7.0 - abs(p.x - 745.0), 65.0 - abs(p.y - 334.0)), 0.0, 1.0);
    let bar_d = clamp(min(7.0 - abs(p.x - 770.0), 38.0 - abs(p.y - 361.0)), 0.0, 1.0);
    let sage = clamp(selected + status + (bar_a + bar_b + bar_c + bar_d) * secondary, 0.0, 1.0);
    let after_sage = after_cards
        + (vec3<f32>(0.337, 0.491, 0.277) - after_cards) * sage * 0.92;

    // Coral mark plus a live signal trace. `sin` is the only transcendental
    // beyond geometric length in this intentionally conservative program.
    let logo = clamp(14.0 - length(p - vec2<f32>(150.0, 101.0)), 0.0, 1.0);
    let wave_y = 350.0 + sin((p.x - 487.0) * 0.036) * 19.0;
    let wave_window = clamp(146.0 - abs(p.x - 487.0), 0.0, 1.0);
    let wave = clamp(3.0 - abs(p.y - wave_y), 0.0, 1.0) * wave_window * primary;
    let coral = clamp(logo + wave, 0.0, 1.0);
    let after_coral = after_sage
        + (vec3<f32>(0.946, 0.404, 0.238) - after_sage) * coral * 0.95;

    // Abstract typography stays analytical: crisp coverage bars, not a host
    // raster layer. Their differing lengths imply the dashboard hierarchy.
    let title = clamp(min(62.0 - abs(p.x - 260.0), 3.0 - abs(p.y - 101.0)), 0.0, 1.0);
    let subtitle = clamp(min(44.0 - abs(p.x - 242.0), 1.5 - abs(p.y - 114.0)), 0.0, 1.0);
    let copy_a = clamp(min(92.0 - abs(p.x - 443.0), 3.0 - abs(p.y - 221.0)), 0.0, 1.0) * primary;
    let copy_b = clamp(min(61.0 - abs(p.x - 412.0), 2.0 - abs(p.y - 237.0)), 0.0, 1.0) * primary;
    let copy = clamp(title * 0.86 + subtitle * 0.36 + copy_a * 0.66 + copy_b * 0.30, 0.0, 1.0);
    let final_color = after_coral
        + (vec3<f32>(0.902, 0.892, 0.818) - after_coral) * copy;

    return vec4<f32>(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

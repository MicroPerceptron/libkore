enable ax_buffer_access;

@group(1) @binding(0) var<storage, read> cells: AxU8Buffer;
@group(1) @binding(1) var<storage, read> atlas: AxU8Buffer;

@group(0) @binding(0) var<uniform> view: vec4<f32>;    // tile xy, viewport wh
@group(0) @binding(1) var<uniform> grid: vec4<f32>;    // columns, rows, (free)
@group(0) @binding(2) var<uniform> state: vec4<f32>;   // (free), paper rgb
@group(0) @binding(3) var<uniform> accent: vec4<f32>;  // accent rgb, native density
@group(0) @binding(4) var<uniform> divider: vec4<f32>; // divider rgb, title kind
@group(0) @binding(5) var<uniform> thumb: vec4<f32>;   // scrollbar thumb x0 y0 x1 y1
@group(0) @binding(6) var<uniform> thumb_ink: vec4<f32>; // thumb rgb, shown

fn terminal_title_glyph(index: i32) -> i32 {
    let g0 = select(0i, 94i, index == 0i);  // ~
    let g1 = select(g0, 15i, index == 2i);  // /
    let g2 = select(g1, 75i, index == 4i);  // k
    let g3 = select(g2, 79i, index == 5i);  // o
    let g4 = select(g3, 82i, index == 6i);  // r
    let g5 = select(g4, 69i, index == 7i);  // e
    let g6 = select(g5, 84i, index == 9i);  // t
    let g7 = select(g6, 69i, index == 10i);
    let g8 = select(g7, 82i, index == 11i);
    let g9 = select(g8, 77i, index == 12i);
    let g10 = select(g9, 73i, index == 13i);
    let g11 = select(g10, 78i, index == 14i);
    let g12 = select(g11, 65i, index == 15i);
    return select(g12, 76i, index == 16i);
}

fn editor_title_glyph(index: i32) -> i32 {
    let g0 = select(0i, 94i, index == 0i);  // ~
    let g1 = select(g0, 15i, index == 2i);  // /
    let g2 = select(g1, 75i, index == 4i);  // k
    let g3 = select(g2, 79i, index == 5i);  // o
    let g4 = select(g3, 82i, index == 6i);  // r
    let g5 = select(g4, 69i, index == 7i);  // e
    let g6 = select(g5, 69i, index == 9i);  // e
    let g7 = select(g6, 68i, index == 10i); // d
    let g8 = select(g7, 73i, index == 11i); // i
    let g9 = select(g8, 84i, index == 12i); // t
    let g10 = select(g9, 79i, index == 13i);// o
    return select(g10, 82i, index == 14i);  // r
}

fn title_glyph(index: i32) -> i32 {
    return select(terminal_title_glyph(index), editor_title_glyph(index), divider.w == 1.0);
}

struct Pixel { @location(0) @interpolate(flat) rgb: i32, }

@fragment
fn main(@builtin(position) position: vec4<f32>) -> Pixel {
    let px = i32(position.x + view.x);
    let py = i32(position.y + view.y);
    let density = accent.w;
    let cw = i32(12.0 * density + 0.5);
    let ch = i32(24.0 * density + 0.5);
    let tx = i32(16.0 * density + 0.5);
    let ty = i32(64.0 * density + 0.5);
    let column = (px - tx) / cw;
    let row = (py - ty) / ch;
    let main_valid = px >= tx && py >= ty && column >= 0i && row >= 0i
        && column < i32(grid.x) && row < i32(grid.y);
    let raw_record = (row * 640i + column) * 10i;
    let raw_width = ax_load_u8(&cells, raw_record + 1i);
    let raw_flags = ax_load_u8(&cells, raw_record + 2i);
    let resolved_column = select(column - 1i, column, raw_width != 0i);
    let record = (row * 640i + resolved_column) * 10i;
    let cell_glyph = ax_load_u8(&cells, record);
    let cell_flags = ax_load_u8(&cells, record + 2i);
    let foreground = vec3<f32>(
        f32(ax_load_u8(&cells, record + 3i)),
        f32(ax_load_u8(&cells, record + 4i)),
        f32(ax_load_u8(&cells, record + 5i)));
    let background = vec3<f32>(
        f32(ax_load_u8(&cells, record + 6i)),
        f32(ax_load_u8(&cells, record + 7i)),
        f32(ax_load_u8(&cells, record + 8i)));

    let title_index = (px - tx) / cw;
    let title_valid = px >= tx && py >= tx && py < i32(40.0 * density + 0.5) && title_index >= 0i && title_index < 17i;
    let glyph_index = select(cell_glyph, title_glyph(title_index), title_valid);
    let origin_x = select(tx + resolved_column * cw, tx + title_index * cw, title_valid);
    let origin_y = select(ty + row * ch, tx, title_valid);
    let lx = px - origin_x;
    let ly = py - origin_y;
    let glyph_valid = (main_valid || title_valid) && lx >= 0i && ly >= 0i && lx < cw && ly < ch;
    // Sum preceding quarter-tier cell areas (q=2 through density*4-1).
    let n = i32(density * 4.0) - 1i;
    let atlas_base = 224i * 36i * (n * (n + 1i) * (2i * n + 1i) / 6i - 1i);
    let atlas_index = select(-2i, atlas_base + ((glyph_index * ch + ly) * cw + lx) * 2i, glyph_valid);
    let coverage = f32(ax_load_u8(&atlas, atlas_index)
        + 256i * ax_load_u8(&atlas, atlas_index + 1i)) / 65535.0;

    let paper = state.yzw;
    let base = select(paper, background, main_valid);
    let ink = select(foreground, accent.xyz, title_valid);
    let glyph_color = mix(base, ink, coverage);
    let underlined = main_valid && (cell_flags & 1i) != 0i
        && py == ty + (row + 1i) * ch - i32(3.0 * density + 0.5);
    let underline_color = select(glyph_color, foreground, underlined);
    // The cursor is a flag of the cell's own record (TERMINAL_CELL_CURSOR):
    // the cells say where it is, so a view scrolled back carries it with
    // its line.
    let cursor = main_valid && (raw_flags & 2i) != 0i
        && py >= ty + (row + 1i) * ch - i32(2.0 * density + 0.5);
    let cursor_color = select(underline_color, accent.xyz, cursor);
    let on_divider = py == i32(48.0 * density + 0.5) && px >= tx && px < i32(view.z) - tx;
    let divided = select(cursor_color, divider.xyz, on_divider);
    let on_thumb = thumb_ink.w != 0.0
        && f32(px) >= thumb.x && f32(px) < thumb.z
        && f32(py) >= thumb.y && f32(py) < thumb.w;
    let color = select(divided, thumb_ink.xyz, on_thumb);
    let rgb = vec3<i32>(clamp(color, vec3<f32>(0.0), vec3<f32>(255.0)) + vec3<f32>(0.5));
    return Pixel((rgb.x << 16u) | (rgb.y << 8u) | rgb.z);
}

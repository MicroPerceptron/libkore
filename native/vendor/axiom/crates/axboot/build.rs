#![feature(allocator_api)]

use axrd::{FontAtlas, FontFace};
use std::{
    alloc::Global,
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

#[path = "src/kerr_params.rs"]
mod kerr_params;
#[path = "build/kerr.rs"]
mod kerr;
// Shared with tools/sky-prep, which uses the half that cuts plates
// (orientation, encoding); the bake uses the half that reads them.
#[allow(dead_code, reason = "shared with tools/sky-prep; each side uses its half")]
#[path = "build/sky.rs"]
mod sky;

use kerr_params::*;

/// One baked text run: its string, face (0 Fraunces 900, 1 DM Sans 450,
/// 2 DM Mono) and size in design pixels at the 1920x1200 reference.
struct Spec {
    text: String,
    face: usize,
    px: f32,
}

fn spec(text: impl Into<String>, face: usize, px: f32) -> Spec {
    Spec {
        text: text.into(),
        face,
        px,
    }
}

fn main() {
    for path in ["assets/hole.wgsl", "assets/compose.wgsl", "build/kerr.rs", "build/sky.rs", "src/kerr_params.rs", "assets/sky/near.bin", "assets/sky/far.bin"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // ── The Kerr scene, traced once ──────────────────────────────────────
    // The trace and the sky atlas (a couple of minutes on every core) depend
    // only on the bake's own source, its parameters and the sky plates. A
    // finished bake is kept once per cargo target directory, under its root
    // (marked by CACHEDIR.TAG) at axboot-bake/<key>, and hard-linked into
    // each build script's OUT_DIR: every target triple and profile built in
    // that directory shares one trace, where a per-OUT_DIR cache re-traced
    // for each (eleven times in one Kore image build). A bake is written to
    // a private directory and renamed into place, so concurrent builds never
    // read a partial one. It holds the three tables, the runtime's constants
    // (kerr.rs), the shader's (kerr_prelude.wgsl), and the atlas grid in the
    // stamp.
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let plate = |name: &str| fs::read(manifest.join("assets/sky").join(name)).unwrap_or_else(|e| panic!("assets/sky/{name}: {e}"));
    let (near, far) = (plate("near.bin"), plate("far.bin"));
    let key = {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        // This script too: it writes the constants the bake's outputs carry.
        let sources = [include_str!("build.rs"), include_str!("build/kerr.rs"), include_str!("build/sky.rs"), include_str!("src/kerr_params.rs")];
        for byte in sources.iter().flat_map(|s| s.bytes()).chain(near.iter().copied()).chain(far.iter().copied()) {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}")
    };
    let outputs = ["kerr_lens.bin", "kerr_turbulence.bin", "kerr_sky.bin", "kerr.rs", "kerr_prelude.wgsl", "kerr_bake.txt"];
    let shelf = out.ancestors().find(|dir| dir.join("CACHEDIR.TAG").is_file()).unwrap_or(&out).join("axboot-bake");
    let bake = shelf.join(&key);
    let read_bake = |dir: &Path| -> Option<([u32; 2], String)> {
        let text = fs::read_to_string(dir.join("kerr_bake.txt")).ok()?;
        let dims: Vec<u32> = text.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        let prelude = fs::read_to_string(dir.join("kerr_prelude.wgsl")).ok()?;
        (dims.len() == 2 && outputs.iter().all(|f| dir.join(f).is_file())).then(|| ([dims[0], dims[1]], prelude))
    };
    let cached = read_bake(&bake);
    let (sky_dims, scene_prelude) = match cached {
        Some(cached) => cached,
        None => {
            let private = shelf.join(format!("{key}.{}", std::process::id()));
            fs::create_dir_all(&private).unwrap();
            let baked = kerr::bake(&near, &far);
            let [fine, medium, coarse, ultra] = &baked.atlas.tiers;
            let dims = baked.atlas.size.map(|v| v as u32);
            fs::write(private.join("kerr_lens.bin"), &baked.lens).unwrap();
            fs::write(private.join("kerr_turbulence.bin"), &baked.turbulence).unwrap();
            fs::write(private.join("kerr_sky.bin"), &baked.sky).unwrap();
            fs::write(
                private.join("kerr.rs"),
                format!(
                    "/// The prograde ISCO for `SPIN`: the disk's inner edge.\n\
                     pub const DISK_INNER: f32 = {:?};\n\
                     /// The redshift factor's packed range over every crossing.\n\
                     pub const G_MIN: f32 = {:?};\n\
                     pub const G_MAX: f32 = {:?};\n\
                     /// The sky atlas's grid.\n\
                     pub const SKY_DIMS: [u32; 3] = [{}, {}, 1];\n\
                     /// Each tier's rectangle in the atlas (x0, y0, x1, y1, texels)\n\
                     /// and its texels per radian, finest first: the fine patch,\n\
                     /// then the medium, coarse and ultra spheres. A swung lookup\n\
                     /// stays inside its own rectangle.\n\
                     pub const SKY_TIERS: [([u32; 4], f32); 4] = {:?};\n\
                     /// Each tier's first texel centre in the hole's frame:\n\
                     /// azimuth and latitude, radians.\n\
                     pub const SKY_TIER_ORIGINS: [[f64; 2]; 4] = {:?};\n\
                     /// ln r for every packed radius code (code 0 is no crossing): the\n\
                     /// runtime decodes a crossing's ln r by lookup, so neither it nor\n\
                     /// the shader takes a logarithm per pixel.\n\
                     pub const LN_R: [f32; 2048] = {:?};\n",
                    baked.disk_inner as f32,
                    baked.g_min as f32,
                    baked.g_max as f32,
                    dims[0],
                    dims[1],
                    baked.atlas.tiers.map(|r| {
                        let [x, y] = r.origin.map(|v| v as u32);
                        ([x, y, x + r.size[0] as u32, y + r.size[1] as u32], (1.0 / r.step) as f32)
                    }),
                    baked.atlas.tiers.map(|r| [r.lon0, r.lat0]),
                    {
                        // The runtime's decode, in its own F32 arithmetic.
                        let inner = baked.disk_inner as f32;
                        core::array::from_fn::<f32, 2048, _>(|code| match code {
                            0 => 0.0,
                            _ => f64::from(inner + (code - 1) as f32 / 2046.0 * (DISK_OUTER - inner)).ln() as f32,
                        })
                    },
                ),
            )
            .unwrap();
            // The shader tells a lookup's tier from where it lands: rows below
            // the fine patch are the medium sphere; above, columns split the
            // fine patch from the small spheres. Half a texel off each edge,
            // since lookups stay two texels in.
            let prelude = format!(
                "const SPIN: f32 = {:?};\n\
                 const DISK_INNER: f32 = {:?};\n\
                 const DISK_OUTER: f32 = {:?};\n\
                 const DISK_GAIN: f32 = {:?};\n\
                 const LOG_R_MIN: f32 = {:?};\n\
                 const LOG_R_MAX: f32 = {:?};\n\
                 const TURBULENCE_COLUMNS: f32 = {:?};\n\
                 const TURBULENCE_ROWS: f32 = {:?};\n\
                 const SKY_FINE_ROWS: f32 = {:?};\n\
                 const SKY_FINE_END: f32 = {:?};\n\
                 const SKY_COARSE_END: f32 = {:?};\n\
                 const SKY_FINE_PER_RADIAN: f32 = {:?};\n\
                 const SKY_MEDIUM_PER_RADIAN: f32 = {:?};\n\
                 const SKY_COARSE_PER_RADIAN: f32 = {:?};\n\
                 const SKY_ULTRA_PER_RADIAN: f32 = {:?};\n",
                SPIN,
                baked.disk_inner as f32,
                DISK_OUTER,
                DISK_GAIN,
                (baked.disk_inner as f32).ln(),
                DISK_OUTER.ln(),
                TURBULENCE_COLUMNS as f32,
                TURBULENCE_ROWS as f32,
                fine.size[1] as f32 - 0.5,
                coarse.origin[0] as f32 - 0.5,
                ultra.origin[0] as f32 - 0.5,
                (1.0 / fine.step) as f32,
                (1.0 / medium.step) as f32,
                (1.0 / coarse.step) as f32,
                (1.0 / ultra.step) as f32,
            );
            fs::write(private.join("kerr_prelude.wgsl"), &prelude).unwrap();
            fs::write(private.join("kerr_bake.txt"), format!("{} {}\n", dims[0], dims[1])).unwrap();
            // Another build may have shelved the same bake meanwhile: keep
            // theirs. Then retire bakes of other keys.
            if fs::rename(&private, &bake).is_err() {
                let _ = fs::remove_dir_all(&private);
            }
            for old in fs::read_dir(&shelf).into_iter().flatten().flatten() {
                let name = old.file_name();
                if name.to_str().is_some_and(|n| n.len() == key.len() && n != key) {
                    let _ = fs::remove_dir_all(old.path());
                }
            }
            ([dims[0], dims[1]], prelude)
        }
    };
    for name in outputs {
        let target = out.join(name);
        let _ = fs::remove_file(&target);
        if fs::hard_link(bake.join(name), &target).is_err() {
            fs::copy(bake.join(name), &target).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    // ── Text ─────────────────────────────────────────────────────────────
    // Build-host only: the target links neither Naga nor a global-allocating
    // font parser. 2x design masks cover native rendering through 3840x2400.
    // Integer summed-area tables preserve coverage under arbitrary downscale
    // and subpixel translations, without repeated filtering of rendered text.
    let fonts = ["Fraunces.ttf", "DMSans.ttf", "DMMono-Regular.ttf"];
    let bytes: Vec<_> = fonts
        .iter()
        .map(|name| {
            let path = format!("../axrd/assets/fonts/{name}");
            println!("cargo:rerun-if-changed={path}");
            fs::read(path).unwrap()
        })
        .collect();
    let faces = [
        FontFace::parse_with_weight(&bytes[0], 900.0).unwrap(),
        FontFace::parse_with_weight(&bytes[1], 450.0).unwrap(),
        FontFace::parse(&bytes[2]).unwrap(),
    ];
    // Indices are the runtime's (`lib.rs` names them); append, never reorder.
    let mut specs: Vec<Spec> = "0123456789:-".chars().map(|c| spec(c, 0, 177.0)).collect();
    specs.extend([
        spec("Your workspace is ready when you are.", 1, 24.0),
        spec("Take a breath. Then make something yours.", 1, 24.0),
        spec("ON DEVICE", 2, 14.0),
        spec("LOCAL WORKSPACE", 2, 14.0),
        spec("READY", 2, 14.0),
        spec("PRESS A KEY OR CLICK TO ENTER", 2, 13.5),
        spec("K", 0, 18.0),
        spec("Kore", 1, 19.0),
        spec("Local session", 1, 14.0),
        spec("Enter", 1, 17.0),
        spec("UTC", 2, 15.0),
        spec("CLOCK UNAVAILABLE", 2, 15.0),
        spec("KORE \u{b7} LOCKED", 2, 15.0),
        spec(
            format!("KERR a = {SPIN:.2} \u{b7} i = {CAMERA_INCLINATION_DEG:.0}\u{b0}"),
            2,
            15.0,
        ),
    ]);
    let mut metadata = String::from("const MASKS: &[Mask] = &[\n");
    let mut data = Vec::new();
    // Each run's width in design pixels, for layout below.
    let mut widths = Vec::new();
    for Spec { text, face, px } in &specs {
        let mut atlas = FontAtlas::new(2048, 1024).unwrap();
        let run = faces[*face]
            .layout_line(&mut atlas, text, px * 2.0)
            .unwrap();
        let nonempty: Vec<_> = run
            .glyphs()
            .iter()
            .filter(|g| g.aw > 0 && g.ah > 0)
            .collect();
        let visible = text.chars().filter(|c| !c.is_whitespace()).count();
        assert_eq!(nonempty.len(), visible, "every glyph of {text:?} must be in its face");
        let x0 = nonempty.iter().map(|g| i32::from(g.dx)).min().unwrap();
        let y0 = nonempty.iter().map(|g| i32::from(g.dy)).min().unwrap();
        let width = (nonempty
            .iter()
            .map(|g| i32::from(g.dx) + i32::from(g.aw))
            .max()
            .unwrap()
            - x0) as usize;
        let height = (nonempty
            .iter()
            .map(|g| i32::from(g.dy) + i32::from(g.ah))
            .max()
            .unwrap()
            - y0) as usize;
        let mut coverage = vec![0u16; width * height];
        for g in nonempty {
            for y in 0..usize::from(g.ah) {
                for x in 0..usize::from(g.aw) {
                    let src = (usize::from(g.ay) + y) * usize::from(atlas.width())
                        + usize::from(g.ax)
                        + x;
                    let dst = (i32::from(g.dy) - y0) as usize * width
                        + y * width
                        + (i32::from(g.dx) - x0) as usize
                        + x;
                    coverage[dst] = coverage[dst].max(atlas.coverage()[src]);
                }
            }
        }
        let offset = data.len();
        let mut integral = vec![0u64; (width + 1) * (height + 1)];
        for y in 0..height {
            let mut row = 0u64;
            for x in 0..width {
                row += u64::from(coverage[y * width + x]);
                integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + row;
            }
        }
        data.extend(integral.iter().flat_map(|v| {
            u32::try_from(*v)
                .expect("coverage integral must fit u32")
                .to_le_bytes()
        }));
        widths.push(width as f32 / 2.0);
        writeln!(
            metadata,
            "Mask {{ width: {width}, height: {height}, top: {y0}, offset: {offset} }}, // {text}"
        )
        .unwrap();
    }
    metadata.push_str("];\n");

    // ── Layout, measured from the baked text ─────────────────────────────
    // The lock-screen kit's overlay at 1920x1200 (its 1280x800 composition
    // scaled by 1.5): chips padded 15 px with a 12 px gap; the seat pill
    // 96 px from the right and 90 px from the bottom, 69 px tall, holding a
    // 42 px avatar, the name and session, and a 42 px Enter button.
    const CHIP_LEFT: f32 = 99.0;
    const CHIP_PAD: f32 = 15.0;
    const CHIP_GAP: f32 = 12.0;
    let mut chips = [[0.0f32; 2]; 3];
    let mut x = CHIP_LEFT;
    for (chip, width) in chips.iter_mut().zip(&widths[14..17]) {
        let half_width = width / 2.0 + CHIP_PAD;
        *chip = [x + half_width, width / 2.0];
        x += 2.0 * half_width + CHIP_GAP;
    }
    const PILL_RADIUS: f32 = 34.5;
    const AVATAR_INSET: f32 = 15.0;
    const AVATAR_RADIUS: f32 = 21.0;
    const TEXT_GAP: f32 = 18.0;
    const BUTTON_GAP: f32 = 30.0;
    const BUTTON_INSET: f32 = 12.0;
    const BUTTON_RADIUS: f32 = 21.0;
    let label = widths[19].max(widths[20]);
    let button_half = widths[21] / 2.0;
    let pill_width = AVATAR_INSET
        + 2.0 * AVATAR_RADIUS
        + TEXT_GAP
        + label
        + BUTTON_GAP
        + 2.0 * (button_half + BUTTON_RADIUS)
        + BUTTON_INSET;
    let pill_half = pill_width / 2.0 - PILL_RADIUS;

    let mut layout = String::new();
    writeln!(layout, "const CHIPS: [[f32; 2]; 3] = {chips:?};").unwrap();
    writeln!(layout, "const PILL_HALF: f32 = {pill_half:?};").unwrap();
    writeln!(layout, "const PILL_RADIUS: f32 = {PILL_RADIUS:?};").unwrap();
    writeln!(layout, "const AVATAR_INSET: f32 = {AVATAR_INSET:?};").unwrap();
    writeln!(layout, "const AVATAR_RADIUS: f32 = {AVATAR_RADIUS:?};").unwrap();
    writeln!(layout, "const BUTTON_INSET: f32 = {BUTTON_INSET:?};").unwrap();
    writeln!(layout, "const BUTTON_RADIUS: f32 = {BUTTON_RADIUS:?};").unwrap();
    writeln!(layout, "const BUTTON_HALF: f32 = {button_half:?};").unwrap();
    metadata.push_str(&layout);
    fs::write(out.join("masks.rs"), metadata).unwrap();
    fs::write(out.join("coverage.bin"), data).unwrap();

    // ── The two passes ───────────────────────────────────────────────────
    // The hole, drawn at the lens map's resolution on larger displays, and
    // the overlay composed over it at the display's.
    let source = include_str!("assets/hole.wgsl");
    let (enable, body) = source
        .split_once('\n')
        .expect("hole.wgsl leads with its enable directive");
    let source = format!("{enable}\n{scene_prelude}{body}");
    fs::write(out.join("hole.wgsl"), &source).unwrap();
    let volume = |binding: u32, dims: [u32; 3]| {
        axshader::ShaderResource::RgbVolume(ax::sampling::VolumeRgbParams {
            resource: ax::sampling::ResourceKey { group: 1, binding },
            dims,
            storage: ax::sampling::VolumeRgbStorage::Uf16E5M11,
        })
    };
    let resources = [
        volume(0, [TURBULENCE_COLUMNS as u32 + 1, TURBULENCE_ROWS as u32, 1]),
        volume(1, [sky_dims[0], sky_dims[1], 1]),
    ];
    let hole = axshader::compile_wgsl_with_resource_specs_in(
        &source,
        axshader::CompileOptions::for_viewport("main", 128, 32),
        &resources,
        Global,
    )
    .expect("the welcome's hole WGSL must pass the ax profile");
    // sky (atlas x, y and coverage), near and far crossings (4 each);
    // viewport, time and the camera's swing.
    assert_eq!(hole.interface().varying_inputs, 11);
    assert_eq!(hole.interface().uniform_inputs, 4);
    fs::write(out.join("hole.ax"), hole.graph().as_bytes()).unwrap();

    let compose = axshader::compile_wgsl_in(
        include_str!("assets/compose.wgsl"),
        axshader::CompileOptions::for_viewport("main", 128, 32),
        Global,
    )
    .expect("the welcome's compose WGSL must pass the ax profile");
    // hole (3), ink (2), shape (3); no uniforms.
    assert_eq!(compose.interface().varying_inputs, 8);
    assert_eq!(compose.interface().uniform_inputs, 0);
    fs::write(out.join("compose.ax"), compose.graph().as_bytes()).unwrap();
}

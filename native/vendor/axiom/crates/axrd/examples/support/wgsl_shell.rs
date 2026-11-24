//! Host-side layout/source specialization for the shell study.
//! Geometry stays WGSL; prepared font coverage is composed into the same .ax
//! DAG, with no readback between the shader and text and no raster backend.

use std::{alloc::Allocator, fmt::Write};

use ax::{Graph, GraphBuf, GraphBuilder, TensorFlags, TensorId};
use axrd::{
    Anchor, Bind, FontAtlas, FontFace, LayoutParams, Offset, PlacedGlyph, Rect, Scene,
    graph::{AlphaAtlas, GraphBackground, GraphOutputFormat, compile_scene_graph_with_output_in},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub night: bool,
    pub floating: bool,
}

impl Options {
    pub fn from_env() -> Self {
        let night = match std::env::var("AXRD_WGSL_THEME").as_deref() {
            Err(_) | Ok("day") => false,
            Ok("night") => true,
            _ => panic!("AXRD_WGSL_THEME must be day or night"),
        };
        let floating = match std::env::var("AXRD_WGSL_LAYOUT").as_deref() {
            Err(_) | Ok("tiled") => false,
            Ok("floating") => true,
            _ => panic!("AXRD_WGSL_LAYOUT must be tiled or floating"),
        };
        Self { night, floating }
    }
}

type Box2 = [f32; 4];

#[derive(Clone, Copy)]
enum Face {
    Display,
    Body,
    Mono,
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

struct Label {
    text: String,
    rect: Box2,
    px: f32,
    face: Face,
    color: u32,
    align: Align,
}

pub struct Shell {
    options: Options,
    paints: String,
    count: usize,
    labels: Vec<Label>,
    ink: u32,
    dim: u32,
    paper: u32,
    panel: u32,
}

fn rgb(color: u32) -> String {
    format!(
        "vec3<f32>({:.8}, {:.8}, {:.8})",
        ((color >> 16) & 255) as f32 / 255.0,
        ((color >> 8) & 255) as f32 / 255.0,
        (color & 255) as f32 / 255.0,
    )
}

impl Shell {
    pub fn new(options: Options) -> Self {
        let mut s = Self {
            options,
            paints: String::new(),
            count: 0,
            labels: Vec::new(),
            ink: if options.night { 0xeee7db } else { 0x201e1d },
            dim: if options.night { 0xc0b6a5 } else { 0x82796a },
            paper: if options.night { 0x302e28 } else { 0xf9f4ed },
            panel: if options.night { 0x3c3931 } else { 0xeee7db },
        };
        let windows = if options.floating {
            // Overlap chrome, but keep readable content clear in this static
            // study. Arbitrary z-order text occlusion is not claimed here.
            [
                [124.0, 72.0, 548.0, 580.0],
                [635.0, 36.0, 535.0, 312.0],
                [650.0, 385.0, 495.0, 293.0],
            ]
        } else {
            [
                [104.0, 26.0, 570.0, 660.0],
                [688.0, 26.0, 490.0, 310.0],
                [688.0, 350.0, 490.0, 336.0],
            ]
        };
        s.inference(windows[0]);
        s.terminal(windows[1]);
        s.poster(windows[2]);
        s.chrome();
        s
    }

    /// Unroll one signed-distance paint into pure WGSL. The instance index is
    /// source-level naming only, never a backend/kernel choice.
    fn rect(&mut self, b: Box2, radius: f32, color: u32, alpha: f32, shadow: bool, border: bool) {
        let [x, y, w, h] = b;
        let n = self.count;
        let next = n + 1;
        let r = radius.min(w * 0.5).min(h * 0.5);
        writeln!(
            self.paints,
            "let q{n} = abs(p - vec2<f32>({:.4}, {:.4})) - vec2<f32>({:.4}, {:.4});",
            x + w * 0.5,
            y + h * 0.5,
            w * 0.5 - r,
            h * 0.5 - r
        )
        .unwrap();
        if r > 0.0 {
            writeln!(self.paints, "let d{n} = length(max(q{n}, vec2<f32>(0.0))) + min(max(q{n}.x, q{n}.y), 0.0) - {r:.4};").unwrap();
        } else {
            writeln!(self.paints, "let d{n} = max(q{n}.x, q{n}.y);").unwrap();
        }
        let mut base = format!("c{n}");
        if shadow {
            // SDF halo approximation, not the reference's quadrature blur.
            writeln!(self.paints, "let e{n} = max(d{n}, 0.0) / 14.0;\nlet s{n} = mix(c{n}, {}, exp(-e{n} * e{n}) * 0.16);", rgb(0x201e1d)).unwrap();
            base = format!("s{n}");
        }
        if border {
            let edge = if self.options.night {
                0x686052
            } else {
                0xc0b6a5
            };
            writeln!(
                self.paints,
                "let b{n} = mix({base}, {}, clamp(0.5 - d{n} * scale, 0.0, 1.0) * {alpha:.4});",
                rgb(edge)
            )
            .unwrap();
            base = format!("b{n}");
        }
        let inset = if border { 1.0 } else { 0.0 };
        writeln!(self.paints, "let c{next} = mix({base}, {}, clamp(0.5 - (d{n} + {inset:.1}) * scale, 0.0, 1.0) * {alpha:.4});", rgb(color)).unwrap();
        self.count = next;
    }

    fn text(&mut self, text: &str, rect: Box2, px: f32, face: Face, color: u32, align: Align) {
        self.labels.push(Label {
            text: text.to_owned(),
            rect,
            px,
            face,
            color,
            align,
        });
    }

    fn pill(&mut self, label: &str, x: f32, y: f32, width: f32, warm: bool) {
        let (bg, fg) = match (self.options.night, warm) {
            (false, true) => (0xffe1d0, 0x643312),
            (false, false) => (0xe1eecc, 0x3d472b),
            (true, true) => (0x65432d, 0xffc6a5),
            (true, false) => (0x414b34, 0xccdbb2),
        };
        self.rect([x, y, width, 21.0], 10.5, bg, 1.0, false, false);
        self.text(
            label,
            [x + 7.0, y, width - 14.0, 21.0],
            10.0,
            Face::Mono,
            fg,
            Align::Center,
        );
    }

    fn window(&mut self, b: Box2, title: &str, dark: bool, focused: bool) {
        let [x, y, w, _] = b;
        self.rect(
            b,
            17.0,
            if dark { 0x2e2b25 } else { self.paper },
            1.0,
            true,
            true,
        );
        self.rect(
            [x + 20.0, y + 20.0, 8.0, 8.0],
            4.0,
            if focused { 0xc67139 } else { 0xa19786 },
            1.0,
            false,
            false,
        );
        self.text(
            title,
            [x + 38.0, y + 12.0, 125.0, 25.0],
            17.0,
            Face::Display,
            if dark { 0xf9f4ed } else { self.ink },
            Align::Left,
        );
        for (i, color) in [0xc67139, 0xc0b6a5, 0xccdbb2].into_iter().enumerate() {
            self.rect(
                [x + w - 23.0, y + 18.0 + i as f32 * 19.0, 10.0, 10.0],
                5.0,
                color,
                1.0,
                false,
                false,
            );
        }
    }

    fn inference(&mut self, b: Box2) {
        let [x, y, w, h] = b;
        self.window(b, "Inference", false, true);
        self.pill("NPU", x + 170.0, y + 14.0, 48.0, true);
        self.pill("MEMLEASE", x + 224.0, y + 14.0, 84.0, false);
        let bx = x + 24.0;
        let bw = w - 68.0;
        self.text(
            "SESSION / ON DEVICE / lfm2.5-8b",
            [bx, y + 62.0, bw, 18.0],
            10.0,
            Face::Mono,
            self.dim,
            Align::Left,
        );
        self.text(
            "184",
            [bx, y + 92.0, 95.0, 56.0],
            48.0,
            Face::Display,
            self.ink,
            Align::Left,
        );
        self.text(
            "tok/s",
            [bx + 105.0, y + 110.0, 60.0, 26.0],
            14.0,
            Face::Mono,
            self.dim,
            Align::Left,
        );
        self.text(
            "4.2 GiB pinned / 12 IOVA windows",
            [bx, y + 160.0, bw, 20.0],
            12.0,
            Face::Mono,
            self.dim,
            Align::Left,
        );
        self.rect([bx, y + 195.0, bw, 7.0], 3.5, self.panel, 1.0, false, false);
        self.rect(
            [bx, y + 195.0, bw * 0.62, 7.0],
            3.5,
            0xc67139,
            1.0,
            false,
            false,
        );
        let top = y + 228.0;
        let pane_h = h - 324.0;
        self.rect([bx, top, bw, pane_h], 10.0, self.panel, 1.0, false, false);
        self.text(
            "YOU",
            [bx + 18.0, top + 17.0, bw - 36.0, 16.0],
            9.5,
            Face::Mono,
            self.dim,
            Align::Left,
        );
        let bubble = if self.options.night {
            0x65432d
        } else {
            0xffe1d0
        };
        self.rect(
            [bx + 16.0, top + 41.0, bw - 32.0, 38.0],
            10.0,
            bubble,
            1.0,
            false,
            false,
        );
        self.text(
            "How much is swap holding?",
            [bx + 30.0, top + 45.0, bw - 60.0, 30.0],
            14.0,
            Face::Body,
            self.ink,
            Align::Left,
        );
        self.text(
            "KORE",
            [bx + 18.0, top + 99.0, bw - 36.0, 16.0],
            9.5,
            Face::Mono,
            self.dim,
            Align::Left,
        );
        for (i, line) in [
            "41 regions in zram. All cold experts.",
            "1.94 GiB, ready when you need them.",
        ]
        .iter()
        .enumerate()
        {
            self.text(
                line,
                [bx + 18.0, top + 126.0 + i as f32 * 24.0, bw - 36.0, 22.0],
                14.0,
                Face::Body,
                self.ink,
                Align::Left,
            );
        }
        if pane_h > 285.0 {
            self.text(
                "LOCAL BY DESIGN",
                [bx + 18.0, top + pane_h - 46.0, bw - 36.0, 20.0],
                10.0,
                Face::Mono,
                if self.options.night {
                    0xaebf92
                } else {
                    0x728157
                },
                Align::Left,
            );
        }
        self.rect(
            [bx, y + h - 66.0, bw, 40.0],
            20.0,
            self.panel,
            1.0,
            false,
            false,
        );
        self.text(
            "Ask on device...",
            [bx + 16.0, y + h - 66.0, bw - 100.0, 40.0],
            13.0,
            Face::Body,
            self.dim,
            Align::Left,
        );
        self.rect(
            [bx + bw - 72.0, y + h - 60.0, 64.0, 28.0],
            14.0,
            0xc67139,
            1.0,
            false,
            false,
        );
        self.text(
            "Send",
            [bx + bw - 72.0, y + h - 60.0, 64.0, 28.0],
            12.0,
            Face::Body,
            0xfff2eb,
            Align::Center,
        );
    }

    fn terminal(&mut self, b: Box2) {
        let [x, y, w, _] = b;
        self.window(b, "Terminal", true, false);
        self.pill("CHANSEND", x + 170.0, y + 14.0, 84.0, false);
        let lines = [
            ("kore % status", 0xaebf92),
            ("input     attached   overrun 0", 0xdcd3c4),
            ("output    attached   flush 1_284_912", 0xdcd3c4),
            ("services  founded 7/7   beacons 7/7", 0xdcd3c4),
            ("kore % caps --held", 0xaebf92),
            ("Task      role::SELF       MODIFY", 0xdcd3c4),
            ("Window    fb0:0x1e0000     ACCESS", 0xdcd3c4),
            ("DmaPool   npu0/pool@0      DERIVE", 0xdcd3c4),
            ("kore % _", 0xf6a06b),
        ];
        for (i, (line, color)) in lines.into_iter().enumerate() {
            self.text(
                line,
                [x + 24.0, y + 57.0 + i as f32 * 25.0, w - 65.0, 20.0],
                12.0,
                Face::Mono,
                color,
                Align::Left,
            );
        }
    }

    fn poster(&mut self, b: Box2) {
        let [x, y, w, h] = b;
        self.window(b, "Studio", false, false);
        self.pill("DISPLAY", x + 142.0, y + 14.0, 72.0, false);
        let bx = x + 24.0;
        let by = y + 54.0;
        let bw = w - 65.0;
        let bh = h - 76.0;
        self.rect([bx, by, bw, bh], 10.0, 0xf5ead8, 1.0, false, false);
        self.rect(
            [bx + bw - 144.0, by + 15.0, 122.0, 122.0],
            61.0,
            0xaebf92,
            0.72,
            false,
            false,
        );
        self.rect(
            [bx + 16.0, by + bh - 92.0, 158.0, 74.0],
            37.0,
            0xf6a06b,
            0.70,
            false,
            false,
        );
        self.text(
            "from inside",
            [bx + 24.0, by + 32.0, bw - 40.0, 48.0],
            38.0,
            Face::Display,
            0x402310,
            Align::Left,
        );
        self.text(
            "kore",
            [bx + 24.0, by + 85.0, bw - 40.0, 50.0],
            44.0,
            Face::Display,
            0x402310,
            Align::Left,
        );
        self.text(
            "A SMALLER, CLOSER COMPUTER.",
            [bx + 25.0, by + bh - 39.0, bw - 44.0, 22.0],
            9.5,
            Face::Mono,
            0x643312,
            Align::Left,
        );
    }

    fn chrome(&mut self) {
        self.rect(
            [20.0, 20.0, 62.0, 666.0],
            31.0,
            self.paper,
            0.80,
            true,
            true,
        );
        self.rect([36.0, 36.0, 30.0, 30.0], 15.0, 0xc67139, 1.0, false, false);
        self.rect([43.0, 43.0, 16.0, 16.0], 8.0, 0xffe1d0, 1.0, false, false);
        for i in 0..3 {
            let y = 88.0 + i as f32 * 48.0;
            self.rect(
                [33.0, y, 36.0, 36.0],
                12.0,
                if i == 0 { 0xc67139 } else { self.panel },
                1.0,
                false,
                false,
            );
            self.text(
                &(i + 1).to_string(),
                [33.0, y, 36.0, 36.0],
                12.0,
                Face::Mono,
                if i == 0 { 0xfff2eb } else { self.dim },
                Align::Center,
            );
        }
        self.text(
            "+",
            [33.0, 235.0, 36.0, 36.0],
            24.0,
            Face::Body,
            self.dim,
            Align::Center,
        );
        self.rect([34.0, 636.0, 34.0, 34.0], 17.0, 0xccdbb2, 1.0, false, false);
        self.text(
            "K",
            [34.0, 636.0, 34.0, 34.0],
            17.0,
            Face::Display,
            0x3d472b,
            Align::Center,
        );

        self.rect(
            [104.0, 712.0, 1074.0, 64.0],
            32.0,
            self.paper,
            0.87,
            true,
            true,
        );
        self.rect(
            [119.0, 727.0, 144.0, 34.0],
            17.0,
            self.panel,
            1.0,
            false,
            false,
        );
        let selected = if self.options.floating { 191.0 } else { 122.0 };
        self.rect(
            [selected, 730.0, 69.0, 28.0],
            14.0,
            self.paper,
            1.0,
            false,
            false,
        );
        self.text(
            "TILE",
            [122.0, 730.0, 69.0, 28.0],
            10.0,
            Face::Mono,
            self.ink,
            Align::Center,
        );
        self.text(
            "FLOAT",
            [191.0, 730.0, 69.0, 28.0],
            10.0,
            Face::Mono,
            self.ink,
            Align::Center,
        );
        self.pill("DWINDLE", 280.0, 734.0, 91.0, false);
        self.rect(
            [389.0, 726.0, 527.0, 36.0],
            18.0,
            self.panel,
            1.0,
            false,
            false,
        );
        self.text(
            "Ask kore - apps, files, or a question",
            [410.0, 726.0, 440.0, 36.0],
            13.0,
            Face::Body,
            self.dim,
            Align::Left,
        );
        self.text(
            "/",
            [860.0, 726.0, 30.0, 36.0],
            13.0,
            Face::Mono,
            self.dim,
            Align::Center,
        );
        self.pill(
            if self.options.night { "NIGHT" } else { "DAY" },
            938.0,
            734.0,
            69.0,
            true,
        );
        self.text(
            "21:19",
            [1024.0, 725.0, 126.0, 38.0],
            27.0,
            Face::Display,
            self.ink,
            Align::Right,
        );
        self.text(
            "WGSL / AX / AXNN     T: TILE-FLOAT     N: DAY-NIGHT     ESC: CLOSE",
            [104.0, 780.0, 1074.0, 18.0],
            9.0,
            Face::Mono,
            self.dim,
            Align::Left,
        );
    }

    pub fn source(&self, width: usize, height: usize) -> String {
        let palette = if self.options.night {
            [0x2e2b25, 0x8c491a, 0x56633f, 0x474238]
        } else {
            [0xebddc5, 0xffc6a5, 0xccdbb2, 0xf9f4ed]
        };
        include_str!("../../assets/shaders/shell.wgsl")
            .replace("__WIDTH__", &format!("{width}.0"))
            .replace("__HEIGHT__", &format!("{height}.0"))
            .replace("__GROUND__", &rgb(palette[0]))
            .replace("__WARM__", &rgb(palette[1]))
            .replace("__SAGE__", &rgb(palette[2]))
            .replace("__PALE__", &rgb(palette[3]))
            .replace("__PAINTS__", &self.paints)
            .replace("__FINAL__", &format!("c{}", self.count))
    }

    pub fn compose_text_in<A: Allocator + Clone>(
        &self,
        shader: &Graph<'_>,
        viewport: [usize; 2],
        region: [usize; 4],
        allocator: A,
    ) -> GraphBuf<A> {
        let [x_origin, y_origin, width, height] = region;
        let faces = [
            FontFace::parse_with_weight(include_bytes!("../../assets/fonts/Fraunces.ttf"), 700.0)
                .unwrap(),
            FontFace::parse_with_weight(include_bytes!("../../assets/fonts/DMSans.ttf"), 500.0)
                .unwrap(),
            FontFace::parse(include_bytes!("../../assets/fonts/DMMono-Regular.ttf")).unwrap(),
        ];
        let mut atlas = FontAtlas::new(2048, 2048).unwrap();
        let mut glyphs = Vec::<PlacedGlyph>::new();
        let mut scene = Scene::<128>::new();
        let scale = (viewport[0] as f32 / 1200.0).min(viewport[1] as f32 / 800.0);
        let ox = (viewport[0] as f32 - 1200.0 * scale) * 0.5;
        let oy = (viewport[1] as f32 - 800.0 * scale) * 0.5;
        // Labels do not overlap. Group equal inks so axrd compiles one exact
        // coverage layer per colour, rather than a full-frame layer per run.
        let mut labels: Vec<_> = self.labels.iter().collect();
        labels.sort_by_key(|label| label.color);
        for label in labels {
            let [x, y, w, h] = label.rect;
            let left = (ox + x * scale).round() as i32;
            let top = (oy + y * scale).round() as i32;
            let rw = ((ox + (x + w) * scale).round() as i32 - left).max(1) as u16;
            let rh = ((oy + (y + h) * scale).round() as i32 - top).max(1) as u16;
            let face = &faces[match label.face {
                Face::Display => 0,
                Face::Body => 1,
                Face::Mono => 2,
            }];
            let mut px = label.px * scale;
            let mut run = face.layout_line(&mut atlas, &label.text, px).unwrap();
            let bounds = |glyphs: &[PlacedGlyph]| {
                glyphs.iter().filter(|g| g.aw > 0 && g.ah > 0).fold(
                    (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
                    |(x0, y0, x1, y1), g| {
                        (
                            x0.min(g.dx as i32),
                            y0.min(g.dy as i32),
                            x1.max(g.dx as i32 + g.aw as i32),
                            y1.max(g.dy as i32 + g.ah as i32),
                        )
                    },
                )
            };
            let mut ink = bounds(run.glyphs());
            if ink.0 == i32::MAX {
                continue;
            }
            for _ in 0..3 {
                let fit = (rw as f32 / (ink.2 - ink.0).max(1) as f32)
                    .min(rh as f32 / (ink.3 - ink.1).max(1) as f32);
                if fit >= 1.0 {
                    break;
                }
                px *= fit * 0.98;
                run = face.layout_line(&mut atlas, &label.text, px).unwrap();
                ink = bounds(run.glyphs());
            }
            let dx = match label.align {
                Align::Left => 0,
                Align::Center => (rw as i32 - (ink.2 - ink.0)) / 2,
                Align::Right => rw as i32 - (ink.2 - ink.0),
            } - ink.0;
            let dy = (rh as i32 - (ink.3 - ink.1)) / 2 - ink.1;
            let start = u16::try_from(glyphs.len()).unwrap();
            for g in run.glyphs() {
                let mut g = *g;
                g.dx = i16::try_from(g.dx as i32 + dx).unwrap();
                g.dy = i16::try_from(g.dy as i32 + dy).unwrap();
                glyphs.push(g);
            }
            scene
                .add_root(
                    LayoutParams {
                        anchor: Anchor::TOP_LEFT,
                        offset: Offset {
                            x: left - x_origin as i32,
                            y: top - y_origin as i32,
                        },
                        w: rw,
                        h: rh,
                    },
                    Bind::TextRun {
                        glyph_start: start,
                        glyph_len: u16::try_from(run.len()).unwrap(),
                        color: label.color,
                    },
                )
                .unwrap();
        }
        let text = compile_scene_graph_with_output_in(
            &mut scene,
            Rect {
                x: 0,
                y: 0,
                w: width as u16,
                h: height as u16,
            },
            GraphBackground::Inputs,
            Some(AlphaAtlas::from_u16(
                atlas.coverage(),
                atlas.width() as usize,
                atlas.height() as usize,
            )),
            &glyphs,
            GraphOutputFormat::PlanarF32,
            allocator.clone(),
        )
        .unwrap();
        connect_rgb_in(shader, &text.graph(), allocator)
    }
}

/// Specialize the text graph's three background inputs to the shader's RGB
/// outputs. Only the original two coordinate axes remain external inputs.
/// This is graph composition, not an intermediate framebuffer upload.
fn connect_rgb_in<A: Allocator + Clone>(
    shader: &Graph<'_>,
    text: &Graph<'_>,
    allocator: A,
) -> GraphBuf<A> {
    let mut builder = GraphBuilder::new_in(allocator.clone());
    let mut rgb = Vec::new_in(allocator.clone());
    for index in 0..shader.tensor_count() {
        let t = shader.tensor(TensorId(index as u32)).unwrap();
        let dimensions: Vec<_> = t.dimensions().collect();
        let id = builder
            .tensor(
                t.dtype,
                TensorFlags(t.flags.0 & !TensorFlags::OUTPUT.0),
                &dimensions,
                t.constant(),
            )
            .unwrap();
        if t.flags.contains(TensorFlags::OUTPUT) {
            rgb.push(id);
        }
    }
    assert_eq!(rgb.len(), 4, "WGSL shader has RGBA outputs");
    for index in 0..shader.node_count() {
        let n = shader.node(index).unwrap();
        builder
            .node(
                n.op,
                n.flags,
                &n.inputs().collect::<Vec<_>>(),
                &n.outputs().collect::<Vec<_>>(),
                n.attributes(),
            )
            .unwrap();
    }
    let mut map = Vec::new_in(allocator);
    let mut input = 0;
    for index in 0..text.tensor_count() {
        let t = text.tensor(TensorId(index as u32)).unwrap();
        if t.flags.contains(TensorFlags::INPUT) {
            assert!(input < 3, "text graph has exactly RGB inputs");
            let source = shader.tensor(rgb[input]).unwrap();
            assert_eq!(t.dtype, source.dtype);
            assert!(t.dimensions().eq(source.dimensions()));
            map.push(rgb[input]);
            input += 1;
        } else {
            map.push(
                builder
                    .tensor(
                        t.dtype,
                        t.flags,
                        &t.dimensions().collect::<Vec<_>>(),
                        t.constant(),
                    )
                    .unwrap(),
            );
        }
    }
    assert_eq!(input, 3);
    for index in 0..text.node_count() {
        let n = text.node(index).unwrap();
        builder
            .node(
                n.op,
                n.flags,
                &n.inputs().map(|id| map[id.0 as usize]).collect::<Vec<_>>(),
                &n.outputs().map(|id| map[id.0 as usize]).collect::<Vec<_>>(),
                n.attributes(),
            )
            .unwrap();
    }
    builder.finish().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Global;

    fn render(options: Options, viewport: [usize; 2], region: [usize; 4]) -> Vec<Vec<f32>> {
        use axnn_vaccel::{AxDeviceModel, AxHetero, AxTosaAccelerator, Session};
        let shell = Shell::new(options);
        let shader = axshader::compile_wgsl_in(
            &shell.source(viewport[0], viewport[1]),
            axshader::CompileOptions::for_viewport("main", region[2] as u32, region[3] as u32),
            Global,
        )
        .unwrap();
        let joined =
            shell.compose_text_in(&shader.graph().parse().unwrap(), viewport, region, Global);
        let session = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        let mut runtime = AxHetero::<AxTosaAccelerator, AxTosaAccelerator>::new(session, None);
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            ..AxDeviceModel::default()
        };
        let executable = runtime
            .compile_in(&joined.parse().unwrap(), &model, Global)
            .unwrap();
        let axis = |offset: usize, length: usize| -> Vec<u8> {
            (offset..offset + length)
                .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
                .collect()
        };
        let x = axis(region[0], region[2]);
        let y = axis(region[1], region[3]);
        let outputs = runtime.run_in(&executable, &[&x, &y], Global).unwrap();
        runtime.release_executable(executable).unwrap();
        outputs
            .iter()
            .map(|plane| {
                plane
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn native_shell_tiles_match_full_frame_including_text_and_partial_edges() {
        for options in [
            Options::default(),
            Options {
                night: true,
                floating: true,
            },
        ] {
            let viewport = [121, 83];
            let full = render(options, viewport, [0, 0, 121, 83]);
            assert!(
                full.iter()
                    .flatten()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
            assert!(full[0].iter().copied().sum::<f32>() > 100.0);
            for region in [
                [0, 0, 64, 48],
                [64, 0, 57, 48],
                [0, 48, 64, 35],
                [64, 48, 57, 35],
            ] {
                let tile = render(options, viewport, region);
                for c in 0..3 {
                    for y in 0..region[3] {
                        for x in 0..region[2] {
                            let expected = full[c][(y + region[1]) * viewport[0] + x + region[0]];
                            let actual = tile[c][y * region[2] + x];
                            assert!(
                                (expected - actual).abs() < 0.0001,
                                "tile {region:?}, pixel {x},{y}, channel {c}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn both_shell_layouts_compile_and_keep_only_coordinate_inputs() {
        for options in [
            Options::default(),
            Options {
                night: true,
                floating: true,
            },
        ] {
            let shell = Shell::new(options);
            let shader = axshader::compile_wgsl_in(
                &shell.source(120, 80),
                axshader::CompileOptions::for_viewport("main", 120, 80),
                Global,
            )
            .unwrap();
            let joined = shell.compose_text_in(
                &shader.graph().parse().unwrap(),
                [120, 80],
                [0, 0, 120, 80],
                Global,
            );
            let graph = joined.parse().unwrap();
            let flags: Vec<_> = (0..graph.tensor_count())
                .map(|i| graph.tensor(TensorId(i as u32)).unwrap().flags)
                .collect();
            assert_eq!(
                flags
                    .iter()
                    .filter(|f| f.contains(TensorFlags::INPUT))
                    .count(),
                2
            );
            assert_eq!(
                flags
                    .iter()
                    .filter(|f| f.contains(TensorFlags::OUTPUT))
                    .count(),
                3
            );
            assert!(graph.node_count() > shader.graph().parse().unwrap().node_count());
            for i in 0..graph.node_count() {
                assert!(graph.node(i).unwrap().cacheable());
            }
        }
    }
}

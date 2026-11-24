/// Per-corner radii for rounded binds and rounded hit/clip shapes.
///
/// Radii are physical pixels. Every consumer clamps each radius to half the
/// shorter resolved edge of the node, so oversized radii degrade to a pill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Corners {
    pub tl: u16,
    pub tr: u16,
    pub br: u16,
    pub bl: u16,
}

impl Corners {
    /// Square corners.
    pub const ZERO: Self = Self::uniform(0);

    /// The radii as the raster vocabulary orders them:
    /// `[top-left, top-right, bottom-right, bottom-left]`.
    #[inline]
    pub const fn array(self) -> axnn::raster::Radii {
        [self.tl, self.tr, self.br, self.bl]
    }

    /// The same radius on all four corners.
    #[inline]
    pub const fn uniform(r: u16) -> Self {
        Self {
            tl: r,
            tr: r,
            br: r,
            bl: r,
        }
    }

    /// The largest of the four radii.
    ///
    /// Paths that only understand a single radius (the analytical graph
    /// coverage field) use this value.
    #[inline]
    pub const fn max(self) -> u16 {
        let a = if self.tl > self.tr { self.tl } else { self.tr };
        let b = if self.br > self.bl { self.br } else { self.bl };
        if a > b { a } else { b }
    }

    /// Whether every corner is square.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.tl == 0 && self.tr == 0 && self.br == 0 && self.bl == 0
    }

    /// Each radius clamped to `limit` (typically half the shorter edge).
    #[inline]
    pub const fn clamped(self, limit: u16) -> Self {
        Self {
            tl: if self.tl > limit { limit } else { self.tl },
            tr: if self.tr > limit { limit } else { self.tr },
            br: if self.br > limit { limit } else { self.br },
            bl: if self.bl > limit { limit } else { self.bl },
        }
    }
}

/// A declarative rendering command: "this region should look like X."
///
/// Binds don't execute immediately. The scene collects them, then either the
/// portable `.ax` graph compiler or the immediate compatibility renderer
/// lowers them into display work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bind {
    /// Fill the node's rect with a solid XRGB colour.
    ///
    /// Colour format: `0x00RRGGBB` (X channel forced to 0xFF on write).
    Fill { color: u32 },

    /// Fill the node's rect with a rounded rectangle.
    ///
    /// `radii` are physical corner radii in pixels, each clamped to half the
    /// shorter resolved edge. The immediate paths honour every corner
    /// through the backend's analytic `rounded_fill` (pixel-centre
    /// signed-distance coverage, colour premultiplied by the node's
    /// opacity). The portable graph path evaluates an antialiased
    /// signed-distance coverage field in F32 at the uniform radius
    /// [`Corners::max`]; it does not require a rounded-rectangle kernel
    /// from each provider backend.
    RoundedFill { color: u32, radii: Corners },

    /// An inner ring of `width` pixels along the node's (rounded) edge.
    ///
    /// The ring lies inside the node rect: the outer edge is the rect edge,
    /// the inner edge is inset by `width`. The immediate paths lower it
    /// through the backend's `border_ring` (a width that leaves no hole
    /// degenerates to a rounded fill); the graph compiler rejects it with
    /// [`UnsupportedBind`](crate::graph::GraphError::UnsupportedBind). Hit
    /// testing honours the rounded silhouette.
    Border {
        color: u32,
        width: u16,
        radii: Corners,
    },

    /// A soft shadow under a rounded rect.
    ///
    /// `falloff` is the blur extent in pixels (0 = hard edge; values above
    /// [`MAX_FALLOFF`](axnn::MAX_FALLOFF) are clamped and blurs past ten
    /// pixels have a truncated tail). The node's rect is the shadow's
    /// whole extent: the shape that casts it is the rect inset by
    /// [`shadow_halo`](crate::shadow_halo)`(falloff)` on every side, its
    /// interior fully covered, the fade reaching the rect edge — so size a
    /// shadow node as the shaded rect grown by the halo. The immediate
    /// paths lower it through the backend's `shadow_fill` with a `const`
    /// Gaussian table per blur; the graph compiler rejects it.
    Shadow {
        color: u32,
        radii: Corners,
        falloff: u8,
    },

    /// Blit (copy) from a source surface tensor into the node's rect.
    ///
    /// `src_x` / `src_y` specify the top-left origin within the source
    /// surface to copy from. The region size is the node's own `w × h`.
    Blit { src_x: u16, src_y: u16 },

    /// Render a pre-shaped text run from a glyph atlas.
    ///
    /// `glyph_start` and `glyph_len` index into the `glyphs` slice
    /// passed to [`frame_exec`](crate::frame_exec) /
    /// [`frame_exec_tiled`](crate::frame_exec_tiled).
    /// Each [`PlacedGlyph`](crate::PlacedGlyph) in the run specifies
    /// atlas source coordinates plus integer and fractional offsets from this
    /// node's origin. The graph compositor samples high-precision coverage at
    /// that subpixel phase before alpha blending.
    ///
    /// `color` is the text colour (`0x00RRGGBB`). Both paths tint per-glyph
    /// coverage with it: the graph compiler premultiplies it by coverage,
    /// the immediate paths blend it through the [`AtlasView`](crate::AtlasView)
    /// (`U8`, `U16` or packed-alpha) at the node's opacity. A packed
    /// premultiplied atlas contributes only its alpha byte.
    ///
    /// The glyph data is the output of an external text shaping pipeline
    /// (e.g., HarfBuzz) — the compositor never interprets codepoints,
    /// ligatures, or font tables.
    TextRun {
        glyph_start: u16,
        glyph_len: u16,
        color: u32,
    },

    /// Scaled texture sample from a source surface.
    ///
    /// Copies a `(sw × sh)` region starting at `(sx, sy)` in the source
    /// surface and scales it to fill this node's rect using
    /// nearest-neighbour sampling with alpha blending.
    Sample { sx: u16, sy: u16, sw: u16, sh: u16 },

    /// External surface — blit from an app-provided buffer.
    ///
    /// `id` indexes into the `surfaces` slice passed to
    /// [`frame_exec`](crate::frame_exec) /
    /// [`frame_exec_tiled`](crate::frame_exec_tiled).
    /// The surface tensor is blitted 1:1 at the node's origin, clipped
    /// to the node's `w × h`. If the surface is smaller than the node,
    /// only the surface area is painted; if larger, excess is clipped.
    ///
    /// This is the primary mechanism for compositing window content:
    /// each application renders into its own tensor, and the compositor
    /// blits it into the framebuffer at the window's position.
    Surface { id: u16 },

    /// Structural grouping — no visual content of its own.
    ///
    /// Children paint over the parent area in scene order.
    Composite,
}

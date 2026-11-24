//! `axrd` — scene layout and portable display composition for axiom.
//!
//! **The raster path** is one program.  With the `graph` feature,
//! [`SceneProgram`] lowers a scene to the [`ax::raster`] node family — one
//! node per run of same-kind nodes in paint order, built once — and
//! [`TiledFrame::lower_band`] writes the per-frame parameter rows for one
//! tile row, culled from the topmost node covering each tile opaquely
//! and clipped to every ancestor [`Scene::clip_rect`].  The rows carry
//! each bind exactly: `RoundedFill` / `Border` / `Shadow` with per-corner
//! radii (colour premultiplied by opacity, a `const` falloff table per
//! blur — see [`shadow_halo`] for how a shadow node is sized), `TextRun`
//! coverage tinted from a `U8`, `U16` or packed-alpha atlas, and a
//! [`NodeFlags::CLIP_ROUNDED`] `Fill` / `Blit` / `Surface` inside its
//! rounded aperture.  A CPU lane executes the program on `axnn-cpu`'s
//! raster kernels (`ax-exec`); an accelerator lowers each node to its
//! native blit, fill or blend.  [`host::SceneHost`] (feature `host`) runs
//! it on the host for examples and tests; a compositor drives it on the
//! lane it founds, band by band.  Nothing on the warm path allocates.
//! A scene whose node kinds and paint order change from frame to frame
//! (a document) takes the same path with its ops as per-frame data:
//! [`raster_ops`] reads the frame's op table and
//! [`TiledFrame::lower_band_dynamic`] lowers bands against it into the
//! same rows, which a CPU lane paints row by row (`axnn_cpu::paint_rows`).
//!
//! With the `graph` feature, [`graph::compile_scene_graph_in`] also lowers
//! fills and text coverage into ordinary mixed-precision `.ax` elementwise
//! operations — the route today's accelerator providers execute — and
//! [`axui`] lowers an `axui-core` display list through it.
//!
//! # Scene API
//!
//! [`Scene<N>`] is a `const fn`-constructible, stack-fixed retained tree of
//! at most `N` nodes; nothing on the per-frame path allocates.
//!
//! * **Identity** — `add_*` return a generational [`NodeId`]. Every method
//!   taking one validates it and answers [`RdError::Stale`] once the node was
//!   removed, even if its slot has since been reused.
//! * **Building** — [`Scene::add_root`], [`Scene::add_child`], their
//!   `_with_opacity` variants, and [`Scene::add_node`] (parent, opacity and
//!   [`NodeFlags`] in one call). The index-era `add` / `add_with_opacity`
//!   names remain as deprecated aliases.
//! * **Structure** — [`Scene::remove`] (whole subtree),
//!   [`Scene::reparent`] ([`RdError::WouldCycle`] guarded),
//!   [`Scene::raise_to_top`], [`Scene::lower_to_bottom`] and
//!   [`Scene::move_before`]. Z-order is sibling order; the paint order is a
//!   depth-first preorder rebuilt lazily by
//!   [`Scene::resolve_layout_host`].
//! * **Mutation** — `set_layout` / `set_bind` / `set_opacity` /
//!   `set_parallax` / `set_flags` / `set_radii` by id. Layout and flag
//!   changes dirty the subtree.
//! * **Flags** — [`NodeFlags::CLIP_CHILDREN`] feeds [`Scene::clip_rect`]
//!   (the ancestor clip every renderer and [`hit_test`] honours);
//!   [`NodeFlags::HIT_TRANSPARENT`] makes a node pointer-invisible;
//!   [`NodeFlags::HIT_ROUNDED`] / [`NodeFlags::CLIP_ROUNDED`] give a
//!   `Surface` or `Composite` node a rounded silhouette via [`Node::radii`].
//! * **Binds** — [`Bind::RoundedFill`], [`Bind::Border`] and
//!   [`Bind::Shadow`] carry per-corner [`Corners`]. The raster program
//!   rasterises all three analytically; the graph compiler still reports
//!   `Border` / `Shadow` as `GraphError::UnsupportedBind` and evaluates
//!   `RoundedFill` at [`Corners::max`].
//! * **Damage** — [`Scene::damage_rects`] is a [`DamageList`] of up to
//!   [`DAMAGE_RECTS`] exact rects (old rect + new rect per dirty node, each
//!   freed rect on removal) that collapses to its union on overflow;
//!   [`Scene::damage`] is that union. `None` from either means the first
//!   frame has not been painted yet.
//! * **Input** — [`hit_test`] walks front-to-back, skips transparent and
//!   look-through nodes, rejects points outside the ancestor clip, and
//!   applies an integer corner test to rounded nodes. [`Interaction`] keys
//!   hover / press / focus by [`NodeId`] and drops ids that went stale.
//!
//! Renderers keep their slot-indexed view: [`Scene::order`] (parent before
//! child), [`Scene::node`], [`Scene::rect`], [`Scene::clip_rect`], and
//! [`Scene::id_at`] to recover the id of a slot.

#![no_std]
#![cfg_attr(any(feature = "graph", feature = "font"), feature(allocator_api))]

#[cfg(any(test, feature = "font", feature = "graph"))]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod anim;
#[cfg(feature = "axui")]
pub mod axui;
mod bind;
mod damage;
mod error;
#[cfg(feature = "font")]
pub mod font;
pub mod font8x16;
mod glyph;
#[cfg(feature = "graph")]
pub mod graph;
#[cfg(feature = "host")]
pub mod host;
pub mod input;
mod layout;
#[cfg(feature = "graph")]
pub mod program;
mod storage;
#[cfg(feature = "graph")]
pub use storage::Arena as SceneArena;
mod scene;
#[cfg(feature = "bspl")]
pub mod splash;
pub mod term;
mod tile;
mod viewport;

pub use axnn::CoverageFormat;
pub use axnn::raster::shadow_halo;
pub use bind::{Bind, Corners};
pub use damage::DamageList;
pub use error::RdError;
#[cfg(feature = "font")]
pub use font::{
    BitmapFace, FontAtlas, FontError, FontFace, GlyphAtlas, GlyphSource, LineMetrics, LineSpec,
    Overflow, TextRun, layout_line_into,
};
pub use glyph::PlacedGlyph;
pub use input::{InputEvent, Interaction, NodeEvent, hit_test};
pub use layout::{Anchor, LayoutParams, Offset, Rect};
#[cfg(feature = "graph")]
pub use program::{
    BLIT_SOURCE_SLOT, ProgramBindings, ProgramError, SceneProgram, raster_op_of, raster_ops,
};
pub use scene::{Children, DAMAGE_RECTS, NO_PARENT, Node, NodeFlags, NodeId, Parallax, Scene};
pub use tile::{MAX_TILE_ROWS, MAX_TILED_NODES, TileMask, TiledFrame};
pub use viewport::{SubpixelPoint, SubpixelRect, Viewport};

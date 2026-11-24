//! Retained scene tree.
//!
//! [`Scene<N>`] is a fixed-capacity, `const`-constructible tree of up to `N`
//! nodes. Node storage is struct-of-arrays on the stack; nothing on the warm
//! path allocates, locks, or touches a `static`.
//!
//! # Identity
//!
//! Nodes are addressed by [`NodeId`], a `(slot, generation)` pair. Removing
//! a node bumps its slot's generation, so an id held across a removal fails
//! [`Scene::index_of`] with [`RdError::Stale`] instead of silently naming
//! whatever reused the slot.
//!
//! Renderers walk [`Scene::order`] — raw slot indices in depth-first
//! preorder, so every parent precedes its children and later siblings paint
//! over earlier ones — and read [`Scene::node`], [`Scene::rect`] and
//! [`Scene::clip_rect`] by that index.
//!
//! # Structure
//!
//! Siblings form doubly linked lists (per parent, plus one root list), which
//! makes [`remove`](Scene::remove), [`reparent`](Scene::reparent),
//! [`raise_to_top`](Scene::raise_to_top), [`lower_to_bottom`](Scene::lower_to_bottom)
//! and [`move_before`](Scene::move_before) O(1) link edits plus an O(subtree)
//! damage walk. The paint order is rebuilt lazily by
//! [`resolve_layout_host`](Scene::resolve_layout_host) when structure
//! changed.
//!
//! # Damage
//!
//! Damage is a [`DamageList`] of up to [`DAMAGE_RECTS`] rects: the previous
//! rect of every node that changed, was removed, or was reordered, plus the
//! new rect of everything still alive and dirty once layout is resolved. The
//! list collapses to its union on overflow. [`Scene::damage`] exposes the
//! union for the immediate renderers; [`Scene::damage_rects`] the list.

use crate::storage::Storage;
use axnn::{Backend, ComputeBackend};

use crate::bind::{Bind, Corners};
use crate::damage::DamageList;
use crate::error::RdError;
use crate::layout::{LayoutParams, Rect};

/// Sentinel value: node has no parent (is a root).
pub const NO_PARENT: u16 = u16::MAX;

/// Link sentinel: end of a sibling list / no child / free list end.
const NONE: u16 = u16::MAX;

/// Number of distinct damage rects a scene tracks per frame before
/// collapsing to their union.
pub const DAMAGE_RECTS: usize = 8;

/// Generational handle to a scene node.
///
/// Minted by the `add_*` methods; validated by every accessor that takes it.
/// `Copy` and cheap to compare, but never reused for a different node while
/// the scene lives (a slot's generation advances on every free).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: u16,
    generation: u16,
}

impl NodeId {
    /// The raw slot this id names. Only meaningful while the id is live;
    /// prefer [`Scene::index_of`], which also checks the generation.
    #[inline]
    pub const fn index(self) -> u16 {
        self.index
    }

    /// The slot generation this id was minted under.
    #[inline]
    pub const fn generation(self) -> u16 {
        self.generation
    }
}

/// Per-node behaviour bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NodeFlags(u8);

impl NodeFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// Descendants are clipped to this node's resolved rect
    /// (see [`Scene::clip_rect`]).
    pub const CLIP_CHILDREN: Self = Self(1 << 0);
    /// The node never receives pointer hits; hit testing looks through it.
    pub const HIT_TRANSPARENT: Self = Self(1 << 1);
    /// Hit testing rejects the corner regions outside the node's rounded
    /// silhouette. The radii come from the bind when it carries any
    /// ([`Bind::RoundedFill`], [`Bind::Border`], [`Bind::Shadow`]), otherwise
    /// from [`Node::radii`]. Those three binds are hit-tested rounded even
    /// without this flag.
    pub const HIT_ROUNDED: Self = Self(1 << 2);
    /// This node's own `Fill` / `Blit` / `Surface` paint is confined to
    /// its rounded silhouette (the immediate and tiled paths' rounded
    /// aperture, radii from [`Node::radii`] or the bind).  Descendants are
    /// still clipped to the rect: rounded clipping of a subtree is not
    /// implemented, so a child that must stay inside the arc paints as
    /// its own `CLIP_ROUNDED` node.
    pub const CLIP_ROUNDED: Self = Self(1 << 3);

    /// The raw bits.
    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Flags from raw bits; unknown bits are dropped.
    #[inline]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & 0b1111)
    }

    /// Whether every bit of `other` is set in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no bit is set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `self | other`, usable in `const` context.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// `self` with the bits of `other` cleared.
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl core::ops::BitOr for NodeFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for NodeFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for NodeFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// Pointer-driven translation of one prepared scene layer.
///
/// Each coefficient is a signed physical-pixel displacement at normalized
/// pointer input `+1`, encoded in 1/256-pixel units. The retained `.ax` graph
/// compiler uses the coefficients to build a shallow, subpixel parallax
/// layer from canonical arithmetic. The immediate compatibility compositor
/// renders the neutral (`0, 0`) pose.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Parallax {
    x_256: i16,
    y_256: i16,
}

impl Parallax {
    pub const ZERO: Self = Self::new(0, 0);

    /// Construct physical-pixel coefficients in 1/256-pixel units.
    pub const fn new(x_256: i16, y_256: i16) -> Self {
        Self { x_256, y_256 }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.x_256 == 0 && self.y_256 == 0
    }

    #[inline]
    pub fn x_pixels(self) -> f32 {
        f32::from(self.x_256) / 256.0
    }

    #[inline]
    pub fn y_pixels(self) -> f32 {
        f32::from(self.y_256) / 256.0
    }
}

/// A node in the scene graph.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub layout: LayoutParams,
    pub bind: Bind,
    /// Per-node opacity: 255 = fully opaque, 0 = fully transparent.
    /// When < 255, paint uses premultiplied-alpha `blit_blend`.
    pub opacity: u8,
    /// Shallow retained-graph displacement. Zero preserves the ordinary
    /// screen-fixed scene semantic.
    pub parallax: Parallax,
    /// Behaviour bits (clipping, hit testing).
    pub flags: NodeFlags,
    /// Corner geometry for [`NodeFlags::HIT_ROUNDED`] /
    /// [`NodeFlags::CLIP_ROUNDED`] on binds that carry no radii of their own
    /// (a `Surface` window, a `Composite` group). Binds with radii win.
    pub radii: Corners,
    /// Slot index of the parent. [`NO_PARENT`] = root.
    pub parent: u16,
}

impl Node {
    /// Whether this node is a root.
    #[inline]
    pub const fn is_root(&self) -> bool {
        self.parent == NO_PARENT
    }

    /// The radii that shape this node for hit testing and rounded clipping:
    /// the bind's when it has any, else [`Node::radii`].
    #[inline]
    pub const fn shape_radii(&self) -> Corners {
        match self.bind {
            Bind::RoundedFill { radii, .. }
            | Bind::Border { radii, .. }
            | Bind::Shadow { radii, .. } => radii,
            _ => self.radii,
        }
    }

    /// Whether hit testing must apply the rounded-corner test.
    #[inline]
    pub const fn hit_rounded(&self) -> bool {
        self.flags.contains(NodeFlags::HIT_ROUNDED)
            || matches!(
                self.bind,
                Bind::RoundedFill { .. } | Bind::Border { .. } | Bind::Shadow { .. }
            )
    }
}

/// Iterator over the children of one node, in sibling (paint) order.
pub struct Children<'a, const N: usize> {
    scene: &'a Scene<N>,
    next: u16,
}

impl<const N: usize> Iterator for Children<'_, N> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.next == NONE {
            return None;
        }
        let idx = self.next;
        self.next = self.scene.next_sibling[idx as usize];
        Some(NodeId {
            index: idx,
            generation: self.scene.generation[idx as usize],
        })
    }
}

/// Fixed-capacity retained scene tree (see the crate docs, "Scene API").
///
/// `N` is the maximum number of nodes. Node metadata lives on the stack.
/// Typical usage: `Scene<64>`, `Scene<128>`, `Scene<256>`.
pub struct Scene<const N: usize> {
    nodes: Storage<Node, N>,
    rects: Storage<Rect, N>,
    prev_rects: Storage<Rect, N>,
    /// Resolved ancestor clip per slot; `None` = unclipped.
    clips: Storage<Option<Rect>, N>,
    /// Depth-first preorder of live slots (valid when `!order_dirty`).
    order: Storage<u16, N>,
    dirty: Storage<bool, N>,
    alive: Storage<bool, N>,
    generation: Storage<u16, N>,
    first_child: Storage<u16, N>,
    last_child: Storage<u16, N>,
    next_sibling: Storage<u16, N>,
    prev_sibling: Storage<u16, N>,
    first_root: u16,
    last_root: u16,
    /// Head of the free-slot list, threaded through `next_sibling`.
    free_head: u16,
    /// Slots ever handed out; everything at or above is pristine.
    high: u16,
    alive_count: u16,
    order_len: u16,
    order_dirty: bool,
    damage: DamageList<DAMAGE_RECTS>,
    layout_resolved: bool,
    /// True after the first frame has been painted (prev_rects are valid).
    has_prev: bool,
}

impl<const N: usize> Default for Scene<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Scene<N> {
    const EMPTY_NODE: Node = Node {
        layout: LayoutParams {
            anchor: crate::layout::Anchor::TOP_LEFT,
            offset: crate::layout::Offset::ZERO,
            w: 0,
            h: 0,
        },
        bind: Bind::Composite,
        opacity: 255,
        parallax: Parallax::ZERO,
        flags: NodeFlags::NONE,
        radii: Corners::ZERO,
        parent: NO_PARENT,
    };

    /// Create an empty scene.
    pub const fn new() -> Self {
        const {
            assert!(
                N < u16::MAX as usize,
                "Scene capacity must leave u16::MAX free as a sentinel"
            );
        }
        Scene {
            nodes: Storage::inline([Self::EMPTY_NODE; N]),
            rects: Storage::inline([Rect::ZERO; N]),
            prev_rects: Storage::inline([Rect::ZERO; N]),
            clips: Storage::inline([None; N]),
            order: Storage::inline([0u16; N]),
            dirty: Storage::inline([false; N]),
            alive: Storage::inline([false; N]),
            generation: Storage::inline([0u16; N]),
            first_child: Storage::inline([NONE; N]),
            last_child: Storage::inline([NONE; N]),
            next_sibling: Storage::inline([NONE; N]),
            prev_sibling: Storage::inline([NONE; N]),
            first_root: NONE,
            last_root: NONE,
            free_head: NONE,
            high: 0,
            alive_count: 0,
            order_len: 0,
            order_dirty: false,
            damage: DamageList::new(),
            layout_resolved: false,
            has_prev: false,
        }
    }

    /// An empty runtime-sized scene backed by the caller's named arena.
    /// Use `Scene::<0>::new_in` to keep inline metadata out of its owner.
    #[cfg(feature = "graph")]
    pub fn new_in(arena: crate::SceneArena) -> Self {
        let mut scene = Self::new();
        scene.nodes = Storage::in_arena([Self::EMPTY_NODE; N], arena);
        scene.rects = Storage::in_arena([Rect::ZERO; N], arena);
        scene.prev_rects = Storage::in_arena([Rect::ZERO; N], arena);
        scene.clips = Storage::in_arena([None; N], arena);
        scene.order = Storage::in_arena([0u16; N], arena);
        scene.dirty = Storage::in_arena([false; N], arena);
        scene.alive = Storage::in_arena([false; N], arena);
        scene.generation = Storage::in_arena([0u16; N], arena);
        scene.first_child = Storage::in_arena([NONE; N], arena);
        scene.last_child = Storage::in_arena([NONE; N], arena);
        scene.next_sibling = Storage::in_arena([NONE; N], arena);
        scene.prev_sibling = Storage::in_arena([NONE; N], arena);
        scene
    }

    /// Reserve room for additional pristine slots before structural edits.
    /// A refusal preserves every live node, link, and generation.
    #[cfg(feature = "graph")]
    pub fn reserve_nodes(&mut self, additional: usize) -> Result<(), RdError<()>> {
        let len = (self.high as usize)
            .checked_add(additional)
            .filter(|len| *len < u16::MAX as usize)
            .ok_or(RdError::Full)?;
        if len <= self.nodes.len() {
            return Ok(());
        }
        let len = len
            .max(self.nodes.len().saturating_mul(2))
            .min(u16::MAX as usize - 1);
        self.nodes.reserve_to(len).map_err(|_| RdError::Full)?;
        self.rects.reserve_to(len).map_err(|_| RdError::Full)?;
        self.prev_rects.reserve_to(len).map_err(|_| RdError::Full)?;
        self.clips.reserve_to(len).map_err(|_| RdError::Full)?;
        self.order.reserve_to(len).map_err(|_| RdError::Full)?;
        self.dirty.reserve_to(len).map_err(|_| RdError::Full)?;
        self.alive.reserve_to(len).map_err(|_| RdError::Full)?;
        self.generation.reserve_to(len).map_err(|_| RdError::Full)?;
        self.first_child
            .reserve_to(len)
            .map_err(|_| RdError::Full)?;
        self.last_child.reserve_to(len).map_err(|_| RdError::Full)?;
        self.next_sibling
            .reserve_to(len)
            .map_err(|_| RdError::Full)?;
        self.prev_sibling
            .reserve_to(len)
            .map_err(|_| RdError::Full)?;
        self.nodes.resize_reserved(len, Self::EMPTY_NODE);
        self.rects.resize_reserved(len, Rect::ZERO);
        self.prev_rects.resize_reserved(len, Rect::ZERO);
        self.clips.resize_reserved(len, None);
        self.order.resize_reserved(len, 0u16);
        self.dirty.resize_reserved(len, false);
        self.alive.resize_reserved(len, false);
        self.generation.resize_reserved(len, 0u16);
        self.first_child.resize_reserved(len, NONE);
        self.last_child.resize_reserved(len, NONE);
        self.next_sibling.resize_reserved(len, NONE);
        self.prev_sibling.resize_reserved(len, NONE);
        Ok(())
    }

    // ── Identity ───────────────────────────────────────────────────

    /// Number of live nodes in the scene.
    #[inline]
    pub fn len(&self) -> u16 {
        self.alive_count
    }

    /// Number of live nodes (alias of [`len`](Self::len)).
    #[inline]
    pub fn alive_count(&self) -> u16 {
        self.alive_count
    }

    /// Whether the scene has no live nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// Resolve an id to its slot index, or `None` when the id is stale.
    #[inline]
    pub fn index_of(&self, id: NodeId) -> Option<u16> {
        let i = id.index as usize;
        if i < self.nodes.len() && self.alive[i] && self.generation[i] == id.generation {
            Some(id.index)
        } else {
            None
        }
    }

    #[inline]
    fn slot(&self, id: NodeId) -> Result<usize, RdError<()>> {
        self.index_of(id).map(|i| i as usize).ok_or(RdError::Stale)
    }

    /// Whether `id` still names a live node.
    #[inline]
    pub fn is_alive(&self, id: NodeId) -> bool {
        self.index_of(id).is_some()
    }

    /// The live id occupying slot `idx` (from [`order`](Self::order)),
    /// or `None` for a free slot.
    #[inline]
    pub fn id_at(&self, idx: u16) -> Option<NodeId> {
        let i = idx as usize;
        if i < self.nodes.len() && self.alive[i] {
            Some(NodeId {
                index: idx,
                generation: self.generation[i],
            })
        } else {
            None
        }
    }

    /// Every live node id, in slot order (not paint order).
    pub fn iter_alive(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.high).filter_map(move |idx| self.id_at(idx))
    }

    /// The children of `id` in sibling order. Empty for a stale id.
    pub fn children(&self, id: NodeId) -> Children<'_, N> {
        let next = match self.index_of(id) {
            Some(idx) => self.first_child[idx as usize],
            None => NONE,
        };
        Children { scene: self, next }
    }

    /// The root nodes in sibling order.
    pub fn roots(&self) -> Children<'_, N> {
        Children {
            scene: self,
            next: self.first_root,
        }
    }

    /// The parent of `id`; `None` for a root or a stale id.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        let idx = self.index_of(id)?;
        let p = self.nodes[idx as usize].parent;
        if p == NO_PARENT { None } else { self.id_at(p) }
    }

    // ── Construction ───────────────────────────────────────────────

    /// Add a root node (no parent) with full opacity.
    pub fn add_root(&mut self, layout: LayoutParams, bind: Bind) -> Result<NodeId, RdError<()>> {
        self.add_node(None, layout, bind, 255, NodeFlags::NONE)
    }

    /// Add a root node with explicit opacity (0–255).
    pub fn add_root_with_opacity(
        &mut self,
        layout: LayoutParams,
        bind: Bind,
        opacity: u8,
    ) -> Result<NodeId, RdError<()>> {
        self.add_node(None, layout, bind, opacity, NodeFlags::NONE)
    }

    /// Add a child as the last (topmost) child of `parent`, fully opaque.
    pub fn add_child(
        &mut self,
        parent: NodeId,
        layout: LayoutParams,
        bind: Bind,
    ) -> Result<NodeId, RdError<()>> {
        self.add_node(Some(parent), layout, bind, 255, NodeFlags::NONE)
    }

    /// Add a child of `parent` with explicit opacity (0–255).
    pub fn add_child_with_opacity(
        &mut self,
        parent: NodeId,
        layout: LayoutParams,
        bind: Bind,
        opacity: u8,
    ) -> Result<NodeId, RdError<()>> {
        self.add_node(Some(parent), layout, bind, opacity, NodeFlags::NONE)
    }

    /// Add a child node under `parent` with full opacity.
    #[deprecated(note = "renamed to `add_child`")]
    pub fn add(
        &mut self,
        parent: NodeId,
        layout: LayoutParams,
        bind: Bind,
    ) -> Result<NodeId, RdError<()>> {
        self.add_child(parent, layout, bind)
    }

    /// Add a child node under `parent` with explicit opacity (0–255).
    #[deprecated(note = "renamed to `add_child_with_opacity`")]
    pub fn add_with_opacity(
        &mut self,
        parent: NodeId,
        layout: LayoutParams,
        bind: Bind,
        opacity: u8,
    ) -> Result<NodeId, RdError<()>> {
        self.add_child_with_opacity(parent, layout, bind, opacity)
    }

    /// The general insert: append a node as the last child of `parent`
    /// (or as the last root) with explicit opacity and flags.
    ///
    /// Errors: [`RdError::Stale`] for a dead parent, [`RdError::Full`]
    /// when every slot is taken.
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        layout: LayoutParams,
        bind: Bind,
        opacity: u8,
        flags: NodeFlags,
    ) -> Result<NodeId, RdError<()>> {
        let parent = match parent {
            Some(id) => self.slot(id)? as u16,
            None => NO_PARENT,
        };
        let idx = self.alloc_slot()?;
        let i = idx as usize;
        self.nodes[i] = Node {
            layout,
            bind,
            opacity,
            parallax: Parallax::ZERO,
            flags,
            radii: Corners::ZERO,
            parent,
        };
        self.link_last(idx);
        self.touch_structure();
        self.mark_dirty(idx);
        Ok(NodeId {
            index: idx,
            generation: self.generation[i],
        })
    }

    /// Remove `id` and its whole subtree, freeing every slot.
    ///
    /// Each freed node's on-screen rect joins the damage list. Ids of the
    /// removed nodes become stale.
    pub fn remove(&mut self, id: NodeId) -> Result<(), RdError<()>> {
        let root = self.slot(id)? as u16;
        self.unlink(root);
        // Postorder traversal through the retained links needs no stack or
        // allocation, regardless of the scene's capacity.
        let mut current = root;
        loop {
            while self.first_child[current as usize] != NONE {
                current = self.first_child[current as usize];
            }
            loop {
                let next = self.next_sibling[current as usize];
                let parent = self.nodes[current as usize].parent;
                self.free_slot(current);
                if current == root {
                    self.touch_structure();
                    return Ok(());
                }
                if next != NONE {
                    current = next;
                    break;
                }
                current = parent;
            }
        }
    }

    /// Move `id` (with its subtree) under `parent` (`None` = make it a
    /// root), as the last child.
    ///
    /// Errors: [`RdError::Stale`] for a dead id or parent,
    /// [`RdError::WouldCycle`] when `parent` is `id` or one of its
    /// descendants.
    pub fn reparent(&mut self, id: NodeId, parent: Option<NodeId>) -> Result<(), RdError<()>> {
        let idx = self.slot(id)? as u16;
        let new_parent = match parent {
            Some(p) => self.slot(p)? as u16,
            None => NO_PARENT,
        };
        let mut a = new_parent;
        while a != NO_PARENT {
            if a == idx {
                return Err(RdError::WouldCycle);
            }
            a = self.nodes[a as usize].parent;
        }
        self.mark_subtree_dirty(idx);
        self.unlink(idx);
        self.nodes[idx as usize].parent = new_parent;
        self.link_last(idx);
        self.touch_structure();
        Ok(())
    }

    /// Make `id` the last (frontmost) among its siblings.
    pub fn raise_to_top(&mut self, id: NodeId) -> Result<(), RdError<()>> {
        let idx = self.slot(id)? as u16;
        if self.next_sibling[idx as usize] == NONE {
            return Ok(());
        }
        self.mark_subtree_dirty(idx);
        self.unlink(idx);
        self.link_last(idx);
        self.touch_structure();
        Ok(())
    }

    /// Make `id` the first (rearmost) among its siblings.
    pub fn lower_to_bottom(&mut self, id: NodeId) -> Result<(), RdError<()>> {
        let idx = self.slot(id)? as u16;
        if self.prev_sibling[idx as usize] == NONE {
            return Ok(());
        }
        self.mark_subtree_dirty(idx);
        self.unlink(idx);
        self.link_first(idx);
        self.touch_structure();
        Ok(())
    }

    /// Place `id` immediately behind `sibling` in paint order.
    ///
    /// Both must share a parent; otherwise [`RdError::BadNode`] names the
    /// sibling's slot. Moving a node before itself is a no-op.
    pub fn move_before(&mut self, id: NodeId, sibling: NodeId) -> Result<(), RdError<()>> {
        let idx = self.slot(id)? as u16;
        let sib = self.slot(sibling)? as u16;
        if idx == sib {
            return Ok(());
        }
        if self.nodes[idx as usize].parent != self.nodes[sib as usize].parent {
            return Err(RdError::BadNode(sib));
        }
        if self.next_sibling[idx as usize] == sib {
            return Ok(());
        }
        self.mark_subtree_dirty(idx);
        self.unlink(idx);
        self.link_before(idx, sib);
        self.touch_structure();
        Ok(())
    }

    // ── Slot and link maintenance ─────────────────────────────────

    fn alloc_slot(&mut self) -> Result<u16, RdError<()>> {
        #[cfg(feature = "graph")]
        if self.free_head == NONE && self.high as usize == self.nodes.len() {
            self.reserve_nodes(1)?;
        }
        let idx = if self.free_head != NONE {
            let idx = self.free_head;
            self.free_head = self.next_sibling[idx as usize];
            idx
        } else if (self.high as usize) < self.nodes.len() {
            let idx = self.high;
            self.high += 1;
            idx
        } else {
            return Err(RdError::Full);
        };
        let i = idx as usize;
        self.alive[i] = true;
        self.dirty[i] = false;
        self.rects[i] = Rect::ZERO;
        self.prev_rects[i] = Rect::ZERO;
        self.clips[i] = None;
        self.first_child[i] = NONE;
        self.last_child[i] = NONE;
        self.next_sibling[i] = NONE;
        self.prev_sibling[i] = NONE;
        self.alive_count += 1;
        Ok(idx)
    }

    /// Return one slot to the free list. The caller has already unlinked
    /// the subtree root; descendants are freed without unlinking because
    /// their whole list dies with them.
    fn free_slot(&mut self, idx: u16) {
        let i = idx as usize;
        if self.has_prev {
            self.damage.push(self.prev_rects[i]);
        }
        self.alive[i] = false;
        self.dirty[i] = false;
        self.generation[i] = self.generation[i].wrapping_add(1);
        self.nodes[i] = Self::EMPTY_NODE;
        self.rects[i] = Rect::ZERO;
        self.prev_rects[i] = Rect::ZERO;
        self.clips[i] = None;
        self.first_child[i] = NONE;
        self.last_child[i] = NONE;
        self.prev_sibling[i] = NONE;
        self.next_sibling[i] = self.free_head;
        self.free_head = idx;
        self.alive_count -= 1;
    }

    #[inline]
    fn head(&self, parent: u16) -> u16 {
        if parent == NO_PARENT {
            self.first_root
        } else {
            self.first_child[parent as usize]
        }
    }

    #[inline]
    fn tail(&self, parent: u16) -> u16 {
        if parent == NO_PARENT {
            self.last_root
        } else {
            self.last_child[parent as usize]
        }
    }

    #[inline]
    fn set_head(&mut self, parent: u16, v: u16) {
        if parent == NO_PARENT {
            self.first_root = v;
        } else {
            self.first_child[parent as usize] = v;
        }
    }

    #[inline]
    fn set_tail(&mut self, parent: u16, v: u16) {
        if parent == NO_PARENT {
            self.last_root = v;
        } else {
            self.last_child[parent as usize] = v;
        }
    }

    fn link_last(&mut self, idx: u16) {
        let p = self.nodes[idx as usize].parent;
        let prev = self.tail(p);
        self.prev_sibling[idx as usize] = prev;
        self.next_sibling[idx as usize] = NONE;
        if prev == NONE {
            self.set_head(p, idx);
        } else {
            self.next_sibling[prev as usize] = idx;
        }
        self.set_tail(p, idx);
    }

    fn link_first(&mut self, idx: u16) {
        let p = self.nodes[idx as usize].parent;
        let next = self.head(p);
        self.next_sibling[idx as usize] = next;
        self.prev_sibling[idx as usize] = NONE;
        if next == NONE {
            self.set_tail(p, idx);
        } else {
            self.prev_sibling[next as usize] = idx;
        }
        self.set_head(p, idx);
    }

    fn link_before(&mut self, idx: u16, sib: u16) {
        let p = self.nodes[idx as usize].parent;
        let prev = self.prev_sibling[sib as usize];
        self.prev_sibling[idx as usize] = prev;
        self.next_sibling[idx as usize] = sib;
        self.prev_sibling[sib as usize] = idx;
        if prev == NONE {
            self.set_head(p, idx);
        } else {
            self.next_sibling[prev as usize] = idx;
        }
    }

    fn unlink(&mut self, idx: u16) {
        let p = self.nodes[idx as usize].parent;
        let prev = self.prev_sibling[idx as usize];
        let next = self.next_sibling[idx as usize];
        if prev == NONE {
            self.set_head(p, next);
        } else {
            self.next_sibling[prev as usize] = next;
        }
        if next == NONE {
            self.set_tail(p, prev);
        } else {
            self.prev_sibling[next as usize] = prev;
        }
        self.prev_sibling[idx as usize] = NONE;
        self.next_sibling[idx as usize] = NONE;
    }

    #[inline]
    fn touch_structure(&mut self) {
        self.order_dirty = true;
        self.layout_resolved = false;
    }

    // ── Read access (by slot, for renderers) ──────────────────────

    /// Read-only access to a node by slot index.
    #[inline]
    pub fn node(&self, idx: u16) -> &Node {
        &self.nodes[idx as usize]
    }

    /// Read the resolved rect for a slot.
    ///
    /// Only valid after [`resolve_layout`](Self::resolve_layout) or
    /// [`resolve_layout_host`](Self::resolve_layout_host).
    #[inline]
    pub fn rect(&self, idx: u16) -> Rect {
        self.rects[idx as usize]
    }

    /// The ancestor clip for a slot: the intersection of every
    /// [`NodeFlags::CLIP_CHILDREN`] ancestor's rect, or `None` when no
    /// ancestor clips. Valid after layout resolution.
    #[inline]
    pub fn clip_rect(&self, idx: u16) -> Option<Rect> {
        self.clips[idx as usize]
    }

    /// The depth-first preorder of live slots (valid after layout
    /// resolution): parents precede children, later siblings paint over
    /// earlier ones.
    pub fn order(&self) -> &[u16] {
        &self.order[..self.order_len as usize]
    }

    // ── Read access (by id) ───────────────────────────────────────

    /// The node behind `id`, or `None` when stale.
    #[inline]
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.index_of(id).map(|idx| &self.nodes[idx as usize])
    }

    /// The resolved rect behind `id`, or `None` when stale.
    #[inline]
    pub fn rect_of(&self, id: NodeId) -> Option<Rect> {
        self.index_of(id).map(|idx| self.rects[idx as usize])
    }

    // ── Layout ────────────────────────────────────────────────────

    /// Resolve layout and propagate rects in parent-before-child order.
    ///
    /// `root_rect` is the bounding box for root nodes (typically the
    /// framebuffer dimensions). All node positions are resolved
    /// relative to their parent, in dependency order.
    /// The backend parameter remains for source compatibility with the
    /// immediate raster path. New graph compilers should call
    /// [`Self::resolve_layout_host`] directly.
    pub fn resolve_layout<B: Backend + ComputeBackend>(
        &mut self,
        root_rect: Rect,
        _be: &B,
    ) -> Result<(), RdError<B::Error>> {
        self.resolve_layout_host(root_rect);
        Ok(())
    }

    /// Resolve layout without allocating tensors or selecting a compute
    /// backend.
    ///
    /// Rebuilds the paint order (iterative depth-first preorder over the
    /// sibling lists — no recursion, no stack) when structure changed, then
    /// propagates rects and ancestor clips in that order.
    pub fn resolve_layout_host(&mut self, root_rect: Rect) {
        if self.order_dirty {
            self.rebuild_order();
        }
        for i in 0..self.order_len as usize {
            let idx = self.order[i] as usize;
            let node = &self.nodes[idx];
            let (parent_rect, clip) = if node.parent == NO_PARENT {
                (root_rect, None)
            } else {
                let p = node.parent as usize;
                let parent_rect = self.rects[p];
                let clip = if self.nodes[p].flags.contains(NodeFlags::CLIP_CHILDREN) {
                    Some(match self.clips[p] {
                        Some(c) => c.intersect(&parent_rect),
                        None => parent_rect,
                    })
                } else {
                    self.clips[p]
                };
                (parent_rect, clip)
            };
            self.rects[idx] = crate::layout::resolve_child(&parent_rect, &node.layout);
            self.clips[idx] = clip;
        }
        self.layout_resolved = true;
    }

    fn rebuild_order(&mut self) {
        let mut n = 0usize;
        let mut cur = self.first_root;
        while cur != NONE {
            self.order[n] = cur;
            n += 1;
            let first = self.first_child[cur as usize];
            if first != NONE {
                cur = first;
                continue;
            }
            loop {
                let next = self.next_sibling[cur as usize];
                if next != NONE {
                    cur = next;
                    break;
                }
                cur = self.nodes[cur as usize].parent;
                if cur == NO_PARENT {
                    break;
                }
            }
        }
        debug_assert_eq!(n, self.alive_count as usize);
        self.order_len = n as u16;
        self.order_dirty = false;
    }

    /// Whether layout has been resolved since the last modification.
    #[inline]
    pub fn is_resolved(&self) -> bool {
        self.layout_resolved
    }

    /// Whether previous-frame rects are available for incremental repaint.
    #[inline]
    pub(crate) fn has_prev(&self) -> bool {
        self.has_prev
    }

    // ── Mutation by id ────────────────────────────────────────────

    /// Change a node's bind and mark it dirty.
    pub fn set_bind(&mut self, id: NodeId, bind: Bind) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        self.nodes[i].bind = bind;
        self.mark_dirty(i as u16);
        Ok(())
    }

    /// Change a node's opacity and mark it dirty.
    pub fn set_opacity(&mut self, id: NodeId, opacity: u8) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        self.nodes[i].opacity = opacity;
        self.mark_dirty(i as u16);
        Ok(())
    }

    /// Change the retained-graph parallax coefficients for a node.
    pub fn set_parallax(&mut self, id: NodeId, parallax: Parallax) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        self.nodes[i].parallax = parallax;
        self.mark_dirty(i as u16);
        Ok(())
    }

    /// Change a node's flags. Clip flags take effect at the next layout
    /// resolution; the subtree is marked dirty.
    pub fn set_flags(&mut self, id: NodeId, flags: NodeFlags) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        if self.nodes[i].flags == flags {
            return Ok(());
        }
        self.nodes[i].flags = flags;
        self.layout_resolved = false;
        self.mark_subtree_dirty(i as u16);
        Ok(())
    }

    /// Change the corner geometry used by the rounded hit/clip flags.
    pub fn set_radii(&mut self, id: NodeId, radii: Corners) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        self.nodes[i].radii = radii;
        self.mark_dirty(i as u16);
        Ok(())
    }

    /// Change a node's layout and mark its subtree dirty.
    ///
    /// Descendants move with the node, so their old and new rects join the
    /// damage as well. The next resolution re-propagates all positions.
    pub fn set_layout(&mut self, id: NodeId, layout: LayoutParams) -> Result<(), RdError<()>> {
        let i = self.slot(id)?;
        self.nodes[i].layout = layout;
        self.layout_resolved = false;
        self.mark_subtree_dirty(i as u16);
        Ok(())
    }

    // ── Damage tracking ────────────────────────────────────────────

    /// Record damage that did not originate in this scene — a client
    /// surface that committed new pixels under a `Surface` node — so the
    /// next tiled frame repaints those tiles.  Ignored until a first frame
    /// has been committed (everything repaints then anyway).
    pub fn add_damage(&mut self, rect: Rect) {
        if self.has_prev && rect.w != 0 && rect.h != 0 {
            self.damage.push(rect);
        }
    }

    /// The accumulated damage union for this frame.
    ///
    /// Returns `None` if the entire scene needs repainting (nothing has
    /// been painted yet). Returns `Some(Rect::ZERO)` if nothing is dirty.
    pub fn damage(&self) -> Option<Rect> {
        if !self.has_prev {
            return None; // full repaint needed
        }
        Some(self.damage.union())
    }

    /// The accumulated damage rects for this frame (at most
    /// [`DAMAGE_RECTS`]; a single union once the list overflowed).
    ///
    /// Returns `None` if the entire scene needs repainting.
    pub fn damage_rects(&self) -> Option<&[Rect]> {
        if !self.has_prev {
            return None;
        }
        Some(self.damage.rects())
    }

    /// Whether a resolved scene needs rendering, without preparing tile masks
    /// or merging individual dirty nodes into the bounded damage list. An
    /// unchanged frame can publish an independent cursor without allocating.
    /// Includes newly visible nodes whose previous rectangle was empty.
    pub fn needs_repaint(&self) -> bool {
        !self.has_prev
            || !self.damage.rects().is_empty()
            || self.dirty[..self.high as usize].iter().any(|&dirty| dirty)
    }

    /// Whether a specific slot is dirty (changed since last commit).
    #[inline]
    pub(crate) fn is_node_dirty(&self, idx: u16) -> bool {
        self.dirty[idx as usize]
    }

    /// The previous frame's rect for a slot (valid when `has_prev`).
    #[inline]
    pub(crate) fn prev_rect(&self, idx: u16) -> Rect {
        self.prev_rects[idx as usize]
    }

    /// Mark a single slot as dirty, accumulating its previous rect into damage.
    fn mark_dirty(&mut self, idx: u16) {
        let i = idx as usize;
        if self.dirty[i] {
            return; // already tracked
        }
        self.dirty[i] = true;
        if self.has_prev {
            self.damage.push(self.prev_rects[i]);
        }
    }

    /// Mark `idx` and every descendant dirty. Iterative walk over the
    /// sibling links, bounded to the subtree; no stack.
    fn mark_subtree_dirty(&mut self, idx: u16) {
        let mut cur = idx;
        loop {
            self.mark_dirty(cur);
            let first = self.first_child[cur as usize];
            if first != NONE {
                cur = first;
                continue;
            }
            loop {
                if cur == idx {
                    return;
                }
                let next = self.next_sibling[cur as usize];
                if next != NONE {
                    cur = next;
                    break;
                }
                cur = self.nodes[cur as usize].parent;
            }
        }
    }

    /// Finalize damage after layout resolution: add the new rects of dirty
    /// nodes to the damage region.  Called by the host executor; renderers
    /// that prepare a [`TiledFrame`](crate::TiledFrame) themselves
    /// call this before reading [`damage_rects`](Self::damage_rects), and
    /// [`commit_frame`](Self::commit_frame) after painting.
    pub fn finalize_damage(&mut self) {
        if !self.has_prev {
            return; // full repaint — no damage clipping
        }
        for i in 0..self.high as usize {
            if self.dirty[i] && self.alive[i] {
                self.damage.push(self.rects[i]);
            }
        }
    }

    /// Snapshot current rects and clear dirty state — called after paint.
    /// Marks the frame painted: clears per-node dirty flags and the damage
    /// list, and records that previous-frame rects now exist.
    pub fn commit_frame(&mut self) {
        let n = self.high as usize;
        self.prev_rects[..n].copy_from_slice(&self.rects[..n]);
        self.dirty.fill(false);
        self.damage.clear();
        self.has_prev = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Anchor, Offset};
    use alloc::vec::Vec;

    fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
        LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset { x, y },
            w,
            h,
        }
    }

    fn r(x: i32, y: i32, w: u16, h: u16) -> Rect {
        Rect { x, y, w, h }
    }

    const ROOT: Rect = Rect {
        x: 0,
        y: 0,
        w: 1000,
        h: 1000,
    };

    fn fill() -> Bind {
        Bind::Fill { color: 0x00FF0000 }
    }

    #[test]
    fn repaint_query_is_read_only_and_covers_unpainted_and_newly_visible_nodes() {
        let mut scene = Scene::<8>::new();
        let node = scene
            .add_node(None, lp(10, 10, 0, 0), fill(), 0, NodeFlags::NONE)
            .unwrap();
        scene.resolve_layout_host(ROOT);
        assert!(scene.needs_repaint());
        scene.commit_frame();
        assert!(!scene.needs_repaint());
        scene.set_layout(node, lp(10, 10, 20, 20)).unwrap();
        scene.set_opacity(node, 255).unwrap();
        scene.resolve_layout_host(ROOT);
        let before = scene.damage();
        assert!(
            scene.needs_repaint(),
            "old empty bounds cannot hide newly visible pixels"
        );
        assert_eq!(scene.damage(), before, "query must not merge damage");
        scene.commit_frame();
        assert!(!scene.needs_repaint());
        scene.add_damage(r(40, 40, 2, 2));
        assert!(
            scene.needs_repaint(),
            "external pixels without node changes"
        );
        scene.commit_frame();
        scene.remove(node).unwrap();
        assert!(
            scene.needs_repaint(),
            "removed node leaves exposed background"
        );
    }

    /// Resolve, finalize and commit one frame so `prev_rects` are live.
    fn paint<const N: usize>(s: &mut Scene<N>) {
        s.resolve_layout_host(ROOT);
        s.finalize_damage();
        s.commit_frame();
    }

    fn ids<const N: usize>(s: &Scene<N>) -> Vec<u16> {
        s.order().to_vec()
    }

    fn assert_parent_before_child<const N: usize>(s: &Scene<N>) {
        let order = s.order();
        assert_eq!(order.len(), s.len() as usize);
        for (pos, &idx) in order.iter().enumerate() {
            let p = s.node(idx).parent;
            if p != NO_PARENT {
                let ppos = order.iter().position(|&o| o == p).expect("parent listed");
                assert!(ppos < pos, "parent {p} must precede child {idx}");
            }
        }
    }

    #[test]
    fn empty_scene() {
        let s = Scene::<64>::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(s.order().is_empty());
    }

    #[test]
    fn add_root() {
        let mut s = Scene::<64>::new();
        let id = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        assert_eq!(id.index(), 0);
        assert_eq!(s.len(), 1);
        assert!(s.get(id).unwrap().is_root());
        assert_eq!(s.index_of(id), Some(0));
    }

    #[test]
    fn add_child() {
        let mut s = Scene::<64>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let child = s.add_child(root, lp(10, 10, 30, 30), fill()).unwrap();
        assert_eq!(child.index(), 1);
        assert_eq!(s.node(1).parent, 0);
        assert_eq!(s.parent(child), Some(root));
        assert_eq!(s.children(root).collect::<Vec<_>>(), [child]);
        s.resolve_layout_host(ROOT);
        assert_eq!(s.rect_of(child), Some(r(10, 10, 30, 30)));
    }

    #[test]
    fn parallax_uses_fixed_physical_subpixels() {
        let mut s = Scene::<2>::new();
        let root = s.add_root(lp(0, 0, 10, 10), Bind::Composite).unwrap();
        let child = s
            .add_child(root, lp(0, 0, 4, 4), Bind::Fill { color: 0 })
            .unwrap();
        let parallax = Parallax::new(384, -128);
        s.set_parallax(child, parallax).unwrap();
        assert_eq!(s.get(child).unwrap().parallax, parallax);
        assert_eq!(parallax.x_pixels(), 1.5);
        assert_eq!(parallax.y_pixels(), -0.5);
    }

    #[test]
    fn capacity_overflow() {
        let mut s = Scene::<2>::new();
        let root = s.add_root(lp(0, 0, 10, 10), Bind::Composite).unwrap();
        s.add_child(root, lp(0, 0, 5, 5), fill()).unwrap();
        let err = s.add_root(lp(0, 0, 1, 1), Bind::Composite);
        assert_eq!(err, Err(RdError::Full));
        // Freeing a slot makes room again.
        s.remove(root).unwrap();
        assert!(s.is_empty());
        s.add_root(lp(0, 0, 1, 1), Bind::Composite).unwrap();
        s.add_root(lp(0, 0, 1, 1), Bind::Composite).unwrap();
    }

    #[test]
    fn stale_id_rejected_after_remove_and_reuse() {
        let mut s = Scene::<4>::new();
        let a = s.add_root(lp(0, 0, 10, 10), fill()).unwrap();
        let b = s.add_root(lp(0, 0, 10, 10), fill()).unwrap();
        s.remove(a).unwrap();
        assert_eq!(s.index_of(a), None);
        assert!(!s.is_alive(a));
        assert_eq!(s.set_opacity(a, 1), Err(RdError::Stale));
        assert_eq!(s.add_child(a, lp(0, 0, 1, 1), fill()), Err(RdError::Stale));
        assert_eq!(s.remove(a), Err(RdError::Stale));

        // The slot is reused with a new generation; the old id stays dead.
        let c = s.add_root(lp(0, 0, 10, 10), fill()).unwrap();
        assert_eq!(c.index(), a.index());
        assert_ne!(c, a);
        assert_eq!(s.index_of(a), None);
        assert_eq!(s.index_of(c), Some(c.index()));
        assert_eq!(s.index_of(b), Some(b.index()));
        assert_eq!(s.id_at(c.index()), Some(c));
    }

    #[test]
    fn remove_subtree_frees_all() {
        let mut s = Scene::<8>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let a = s.add_child(root, lp(0, 0, 10, 10), fill()).unwrap();
        let a1 = s.add_child(a, lp(0, 0, 5, 5), fill()).unwrap();
        let a2 = s.add_child(a, lp(0, 0, 5, 5), fill()).unwrap();
        let a21 = s.add_child(a2, lp(0, 0, 2, 2), fill()).unwrap();
        let b = s.add_child(root, lp(0, 0, 10, 10), fill()).unwrap();
        assert_eq!(s.len(), 6);

        s.remove(a).unwrap();
        assert_eq!(s.len(), 2);
        for dead in [a, a1, a2, a21] {
            assert!(!s.is_alive(dead));
        }
        assert!(s.is_alive(root));
        assert!(s.is_alive(b));
        assert_eq!(s.children(root).collect::<Vec<_>>(), [b]);
        assert_eq!(s.iter_alive().count(), 2);

        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [root.index(), b.index()]);
        assert_parent_before_child(&s);

        // All four freed slots can be reused.
        for _ in 0..4 {
            s.add_child(b, lp(0, 0, 1, 1), fill()).unwrap();
        }
        assert_eq!(s.len(), 6);
        s.resolve_layout_host(ROOT);
        assert_parent_before_child(&s);
    }

    #[test]
    fn remove_root_and_middle_keeps_lists_linked() {
        let mut s = Scene::<8>::new();
        let a = s.add_root(lp(0, 0, 1, 1), fill()).unwrap();
        let b = s.add_root(lp(0, 0, 1, 1), fill()).unwrap();
        let c = s.add_root(lp(0, 0, 1, 1), fill()).unwrap();
        s.remove(b).unwrap();
        assert_eq!(s.roots().collect::<Vec<_>>(), [a, c]);
        s.remove(a).unwrap();
        assert_eq!(s.roots().collect::<Vec<_>>(), [c]);
        s.remove(c).unwrap();
        assert!(s.roots().next().is_none());
        assert!(s.is_empty());
        s.resolve_layout_host(ROOT);
        assert!(s.order().is_empty());
    }

    #[test]
    fn z_order_edits_change_order() {
        let mut s = Scene::<8>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let a = s.add_child(root, lp(0, 0, 10, 10), fill()).unwrap();
        let a1 = s.add_child(a, lp(0, 0, 5, 5), fill()).unwrap();
        let b = s.add_child(root, lp(0, 0, 10, 10), fill()).unwrap();
        let c = s.add_child(root, lp(0, 0, 10, 10), fill()).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 1, 2, 3, 4]);

        s.raise_to_top(a).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 3, 4, 1, 2]);
        assert_parent_before_child(&s);

        s.lower_to_bottom(c).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 4, 3, 1, 2]);

        s.move_before(a, b).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 4, 1, 2, 3]);
        assert_parent_before_child(&s);

        // Same-parent requirement and self no-op.
        assert_eq!(s.move_before(a1, b), Err(RdError::BadNode(b.index())));
        s.move_before(a, a).unwrap();
        // Already in place: no structural churn.
        s.raise_to_top(b).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 4, 1, 2, 3]);
        assert_eq!(s.children(root).collect::<Vec<_>>(), [c, a, b]);
    }

    #[test]
    fn reparent_moves_subtree_and_refuses_cycles() {
        let mut s = Scene::<8>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let a = s.add_child(root, lp(10, 10, 50, 50), fill()).unwrap();
        let a1 = s.add_child(a, lp(1, 1, 5, 5), fill()).unwrap();
        let b = s.add_child(root, lp(20, 20, 50, 50), fill()).unwrap();

        assert_eq!(s.reparent(a, Some(a)), Err(RdError::WouldCycle));
        assert_eq!(s.reparent(a, Some(a1)), Err(RdError::WouldCycle));
        assert_eq!(s.reparent(root, Some(a1)), Err(RdError::WouldCycle));

        s.reparent(a, Some(b)).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(ids(&s), [0, 3, 1, 2]);
        assert_parent_before_child(&s);
        assert_eq!(s.parent(a), Some(b));
        assert_eq!(s.rect_of(a), Some(r(30, 30, 50, 50)));
        assert_eq!(s.rect_of(a1), Some(r(31, 31, 5, 5)));

        s.reparent(a1, None).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(s.parent(a1), None);
        assert_eq!(s.roots().collect::<Vec<_>>(), [root, a1]);
        assert_eq!(s.rect_of(a1), Some(r(1, 1, 5, 5)));
        assert_parent_before_child(&s);
    }

    #[test]
    fn order_stays_parent_before_child_under_churn() {
        let mut s = Scene::<32>::new();
        let mut live: Vec<NodeId> = Vec::new();
        live.push(s.add_root(lp(0, 0, 10, 10), fill()).unwrap());
        let mut seed = 0x9E37_79B9u32;
        for step in 0..400 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let pick = (seed as usize) % live.len();
            match step % 5 {
                0 | 1 if s.len() < 32 => {
                    let id = s.add_child(live[pick], lp(1, 1, 3, 3), fill()).unwrap();
                    live.push(id);
                }
                2 if live.len() > 1 => {
                    let victim = live[pick];
                    s.remove(victim).unwrap();
                    live.retain(|&id| s.is_alive(id));
                    if live.is_empty() {
                        live.push(s.add_root(lp(0, 0, 10, 10), fill()).unwrap());
                    }
                }
                3 => s.raise_to_top(live[pick]).unwrap(),
                _ => {
                    let target = live[(pick + 1) % live.len()];
                    let _ = s.reparent(live[pick], Some(target));
                }
            }
            s.resolve_layout_host(ROOT);
            assert_parent_before_child(&s);
            assert_eq!(live.len(), s.len() as usize);
        }
    }

    #[test]
    fn damage_after_remove_is_exact_rect_set() {
        let mut s = Scene::<8>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let a = s.add_child(root, lp(10, 10, 20, 20), fill()).unwrap();
        s.add_child(a, lp(2, 2, 4, 4), fill()).unwrap();
        let b = s.add_child(root, lp(60, 60, 20, 20), fill()).unwrap();
        assert_eq!(s.damage_rects(), None);
        paint(&mut s);
        assert_eq!(s.damage_rects(), Some(&[][..]));

        s.remove(a).unwrap();
        // The grandchild sits inside `a`, so the list stays exact and tight.
        assert_eq!(s.damage_rects(), Some(&[r(10, 10, 20, 20)][..]));
        s.remove(b).unwrap();
        assert_eq!(
            s.damage_rects(),
            Some(&[r(10, 10, 20, 20), r(60, 60, 20, 20)][..])
        );
        assert_eq!(s.damage(), Some(r(10, 10, 70, 70)));
        s.resolve_layout_host(ROOT);
        s.finalize_damage();
        assert_eq!(
            s.damage_rects(),
            Some(&[r(10, 10, 20, 20), r(60, 60, 20, 20)][..])
        );
    }

    #[test]
    fn damage_after_move_is_old_plus_new() {
        let mut s = Scene::<8>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let a = s.add_child(root, lp(10, 10, 20, 20), fill()).unwrap();
        let a1 = s.add_child(a, lp(0, 0, 5, 5), fill()).unwrap();
        paint(&mut s);

        s.set_layout(a, lp(50, 50, 20, 20)).unwrap();
        s.resolve_layout_host(ROOT);
        s.finalize_damage();
        assert_eq!(
            s.damage_rects(),
            Some(&[r(10, 10, 20, 20), r(50, 50, 20, 20)][..])
        );
        assert_eq!(s.rect_of(a1), Some(r(50, 50, 5, 5)));
        s.commit_frame();

        // A z-order change repaints the node where it is.
        let b = s.add_child(root, lp(0, 0, 3, 3), fill()).unwrap();
        paint(&mut s);
        s.lower_to_bottom(b).unwrap();
        s.resolve_layout_host(ROOT);
        s.finalize_damage();
        assert_eq!(s.damage_rects(), Some(&[r(0, 0, 3, 3)][..]));
    }

    #[test]
    fn damage_overflow_collapses() {
        let mut s = Scene::<16>::new();
        let root = s.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let mut kids = [None; 12];
        for (i, slot) in kids.iter_mut().enumerate() {
            let x = (i as i32 % 4) * 25;
            let y = (i as i32 / 4) * 30;
            *slot = Some(s.add_child(root, lp(x, y, 5, 5), fill()).unwrap());
        }
        paint(&mut s);
        for id in kids.into_iter().flatten() {
            s.remove(id).unwrap();
        }
        let rects = s.damage_rects().unwrap();
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0], r(0, 0, 80, 65));
        assert_eq!(s.damage(), Some(r(0, 0, 80, 65)));
    }

    #[test]
    fn clip_children_propagates_intersection() {
        let mut s = Scene::<8>::new();
        let root = s
            .add_node(
                None,
                lp(0, 0, 100, 100),
                Bind::Composite,
                255,
                NodeFlags::CLIP_CHILDREN,
            )
            .unwrap();
        let panel = s
            .add_node(
                Some(root),
                lp(50, 50, 100, 100),
                fill(),
                255,
                NodeFlags::CLIP_CHILDREN,
            )
            .unwrap();
        let leaf = s.add_child(panel, lp(-10, -10, 200, 200), fill()).unwrap();
        let free = s.add_child(root, lp(0, 0, 1, 1), fill()).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(s.clip_rect(root.index()), None);
        assert_eq!(s.clip_rect(panel.index()), Some(r(0, 0, 100, 100)));
        assert_eq!(s.clip_rect(leaf.index()), Some(r(50, 50, 50, 50)));
        assert_eq!(s.clip_rect(free.index()), Some(r(0, 0, 100, 100)));

        // Dropping the flag lifts the clip on the next resolution.
        s.set_flags(root, NodeFlags::NONE).unwrap();
        s.resolve_layout_host(ROOT);
        assert_eq!(s.clip_rect(panel.index()), None);
        assert_eq!(s.clip_rect(leaf.index()), Some(r(50, 50, 100, 100)));
    }

    #[test]
    fn deprecated_index_free_aliases_still_work() {
        #![allow(deprecated)]
        let mut s = Scene::<4>::new();
        let root = s.add_root(lp(0, 0, 10, 10), Bind::Composite).unwrap();
        let a = s.add(root, lp(0, 0, 1, 1), fill()).unwrap();
        let b = s.add_with_opacity(root, lp(0, 0, 1, 1), fill(), 7).unwrap();
        assert_eq!(s.children(root).collect::<Vec<_>>(), [a, b]);
        assert_eq!(s.get(b).unwrap().opacity, 7);
    }

    #[test]
    fn node_flags_bit_ops() {
        let f = NodeFlags::CLIP_CHILDREN | NodeFlags::HIT_ROUNDED;
        assert!(f.contains(NodeFlags::CLIP_CHILDREN));
        assert!(!f.contains(NodeFlags::HIT_TRANSPARENT));
        assert_eq!(
            f.difference(NodeFlags::HIT_ROUNDED),
            NodeFlags::CLIP_CHILDREN
        );
        assert_eq!(NodeFlags::from_bits_truncate(0xFF).bits(), 0b1111);
        assert!(NodeFlags::NONE.is_empty());
    }
}

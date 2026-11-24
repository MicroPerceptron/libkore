/// OS-agnostic input event types and interaction state.
///
/// Design principles:
/// - **No OS assumptions**: raw numeric codes, no keysym tables.
///   The platform constructs `InputEvent` from its own input model.
/// - **No callbacks/closures**: the caller pushes events in and reads
///   derived `NodeEvent`s back. Pure data in, pure data out.
/// - **No allocations**: all state is stack-allocated, fixed-capacity.
/// - **No trait objects**: everything is concrete and inlineable.
///
/// Typical integration:
/// ```ignore
/// // Platform fills in raw events from its input source:
/// let raw = InputEvent::PointerMove { x: 320, y: 240 };
///
/// // Feed to interaction state, get back node-level events:
/// let mut out = [NodeEvent::NONE; 8];
/// let count = interaction.feed(&scene, raw, &mut out);
/// for evt in &out[..count] {
///     match evt {
///         NodeEvent::PointerDown { node, x, y, .. } => { /* app logic */ }
///         _ => {}
///     }
/// }
/// ```

// ── Raw input events (platform → compositor) ──────────────────

/// A resolved event entering axrd interaction routing.
///
/// Coordinates are in framebuffer pixel space (same coordinate system
/// as `Rect`). [`axinput::Seat`] owns device/viewport policy and
/// [`InputEvent::from_seat`] bridges the resulting pointer events here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEvent {
    /// Pointer moved to `(x, y)` without button state change.
    PointerMove { x: i32, y: i32 },

    /// Pointer button pressed at `(x, y)`.
    ///
    /// `button`: 0 = primary (left/touch), 1 = secondary (right),
    /// 2 = tertiary (middle). Platform maps its own button model.
    PointerDown { x: i32, y: i32, button: u8 },

    /// Pointer button released at `(x, y)`.
    PointerUp { x: i32, y: i32, button: u8 },

    /// Key pressed. `code` is a platform-defined scancode.
    /// The compositor doesn't interpret key codes — it just routes
    /// them to the focused node.
    KeyDown { code: u16 },

    /// Key released.
    KeyUp { code: u16 },

    /// Focus request: move focus forward (e.g. Tab).
    FocusNext,

    /// Focus request: move focus backward (e.g. Shift+Tab).
    FocusPrev,
}

impl InputEvent {
    /// Bridges one portable seat event into axrd's scene interaction model.
    ///
    /// Scroll and contact events remain available to higher-level widget or
    /// gesture policy and return `None`; axrd's present scene router has no
    /// corresponding node event yet. Pointer buttons above `u8::MAX` likewise
    /// remain representable in `axinput` but cannot enter this legacy ABI.
    pub fn from_seat(event: axinput::SeatEvent) -> Option<Self> {
        use axinput::{ButtonState, SeatEventKind};

        let x = event.position.x;
        let y = event.position.y;
        match event.kind {
            SeatEventKind::Motion => Some(Self::PointerMove { x, y }),
            SeatEventKind::Button { button, state } => {
                let button = u8::try_from(button.0).ok()?;
                Some(match state {
                    ButtonState::Pressed => Self::PointerDown { x, y, button },
                    ButtonState::Released => Self::PointerUp { x, y, button },
                })
            }
            SeatEventKind::Scroll { .. } | SeatEventKind::Contact { .. } => None,
        }
    }
}

use crate::bind::Corners;
use crate::layout::Rect;
use crate::scene::{NodeId, Scene};

// ── Node-level events (compositor → application) ──────────────

/// A derived event targeted at a specific scene node.
///
/// Produced by [`Interaction::feed`] after hit-testing and focus
/// resolution. The application matches on these to implement behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeEvent {
    /// No event (sentinel for fixed-size output buffers).
    None,

    /// Pointer entered a node's rect.
    PointerEnter { node: NodeId },

    /// Pointer left a node's rect.
    PointerLeave { node: NodeId },

    /// Pointer moved within a node's rect.
    ///
    /// `lx` / `ly` are local coordinates relative to the node's
    /// top-left corner.
    PointerMove { node: NodeId, lx: i32, ly: i32 },

    /// Pointer button pressed on a node.
    PointerDown {
        node: NodeId,
        lx: i32,
        ly: i32,
        button: u8,
    },

    /// Pointer button released on a node.
    PointerUp {
        node: NodeId,
        lx: i32,
        ly: i32,
        button: u8,
    },

    /// Node gained focus.
    FocusIn { node: NodeId },

    /// Node lost focus.
    FocusOut { node: NodeId },

    /// Key pressed while node has focus.
    KeyDown { node: NodeId, code: u16 },

    /// Key released while node has focus.
    KeyUp { node: NodeId, code: u16 },
}

impl NodeEvent {
    /// Sentinel value for empty output buffer slots.
    pub const NONE: Self = NodeEvent::None;
}

// ── Hit testing ───────────────────────────────────────────────

/// Hit-test a point against the scene's resolved rects.
///
/// Walks nodes in **reverse paint order** (front-to-back) and returns
/// the first node that contains `(px, py)`. A node is skipped when it is
/// structural (`Composite`), fully transparent, flagged
/// [`NodeFlags::HIT_TRANSPARENT`](crate::NodeFlags::HIT_TRANSPARENT), when
/// the point lies outside its ancestor [`clip_rect`](Scene::clip_rect), or
/// when it is rounded (`RoundedFill` / `Border` / `Shadow` or
/// [`HIT_ROUNDED`](crate::NodeFlags::HIT_ROUNDED)) and the point falls in a
/// corner outside the silhouette.
///
/// Returns `None` if no node contains the point.
pub fn hit_test<const N: usize>(scene: &Scene<N>, px: i32, py: i32) -> Option<NodeId> {
    let order = scene.order();

    // Reverse iteration: last painted = frontmost.
    for &idx in order.iter().rev() {
        let node = scene.node(idx);

        // Skip structural, invisible and look-through nodes.
        if node.opacity == 0 {
            continue;
        }
        if matches!(node.bind, crate::bind::Bind::Composite) {
            continue;
        }
        if node
            .flags
            .contains(crate::scene::NodeFlags::HIT_TRANSPARENT)
        {
            continue;
        }

        let r = scene.rect(idx);
        if !r.contains_point(px, py) {
            continue;
        }
        if let Some(clip) = scene.clip_rect(idx) {
            if !clip.contains_point(px, py) {
                continue;
            }
        }
        if node.hit_rounded() && !rounded_contains(r, node.shape_radii(), px, py) {
            continue;
        }
        return scene.id_at(idx);
    }

    None
}

/// Integer rounded-rectangle containment for the pixel `(px, py)`.
///
/// Matches the immediate rasteriser's coverage rule: a pixel is inside a
/// corner arc when its centre `(l + 0.5)` lies within the circle of radius
/// `r` centred `r` pixels in from both edges, evaluated as
/// `(2l + 1 - 2r)² + ... <= (2r)²` so no float or sqrt is needed. Radii
/// are clamped to half the shorter edge.
pub fn rounded_contains(rect: Rect, radii: Corners, px: i32, py: i32) -> bool {
    if !rect.contains_point(px, py) {
        return false;
    }
    let radii = radii.clamped(rect.w.min(rect.h) / 2);
    if radii.is_zero() {
        return true;
    }
    let lx = px - rect.x;
    let ly = py - rect.y;
    let w = i32::from(rect.w);
    let h = i32::from(rect.h);
    // (radius, distance from the corner along x, along y)
    let corners = [
        (i32::from(radii.tl), lx, ly),
        (i32::from(radii.tr), w - 1 - lx, ly),
        (i32::from(radii.br), w - 1 - lx, h - 1 - ly),
        (i32::from(radii.bl), lx, h - 1 - ly),
    ];
    for (r, cx, cy) in corners {
        if r == 0 || cx >= r || cy >= r {
            continue;
        }
        let d = 2 * r;
        let dx = 2 * cx + 1 - d;
        let dy = 2 * cy + 1 - d;
        if dx * dx + dy * dy > d * d {
            return false;
        }
    }
    true
}

// ── Interaction state machine ─────────────────────────────────

/// Tracks pointer hover, press, and keyboard focus state.
///
/// Stack-allocated, `no_std`-safe. Feed it raw [`InputEvent`]s and
/// read back derived [`NodeEvent`]s.
///
/// Does **not** own the scene — borrows it during [`feed`](Self::feed)
/// for hit-testing. The caller is responsible for keeping the scene's
/// layout resolved before feeding events.
pub struct Interaction {
    /// Currently hovered node (pointer is inside its rect).
    hover: Option<NodeId>,
    /// Node that received the last `PointerDown` (for press tracking).
    pressed: Option<NodeId>,
    /// Currently focused node (receives key events).
    focus: Option<NodeId>,
    /// Last known pointer position.
    pointer_x: i32,
    pointer_y: i32,
}

impl Interaction {
    /// Create a new interaction state with no hover, press, or focus.
    pub const fn new() -> Self {
        Interaction {
            hover: None,
            pressed: None,
            focus: None,
            pointer_x: 0,
            pointer_y: 0,
        }
    }

    /// The currently hovered node, if any.
    #[inline]
    pub fn hovered(&self) -> Option<NodeId> {
        self.hover
    }

    /// The node that received the last pointer-down, if any.
    #[inline]
    pub fn pressed(&self) -> Option<NodeId> {
        self.pressed
    }

    /// The currently focused node, if any.
    #[inline]
    pub fn focused(&self) -> Option<NodeId> {
        self.focus
    }

    /// Set focus to a specific node (or `None` to clear).
    pub fn set_focus(&mut self, node: Option<NodeId>) {
        self.focus = node;
    }

    /// Feed a raw input event, producing zero or more node events.
    ///
    /// Returns the number of events written to `out`. The caller
    /// provides a fixed-size buffer; events beyond its capacity are
    /// silently dropped.
    ///
    /// The scene must have resolved layout (rects valid) before
    /// calling this. Ids held from an earlier frame (hover, press, focus)
    /// that have since been removed are dropped without an event.
    pub fn feed<const N: usize>(
        &mut self,
        scene: &Scene<N>,
        event: InputEvent,
        out: &mut [NodeEvent],
    ) -> usize {
        let mut count = 0;

        let mut emit = |evt: NodeEvent| {
            if count < out.len() {
                out[count] = evt;
                count += 1;
            }
        };

        match event {
            InputEvent::PointerMove { x, y } => {
                self.pointer_x = x;
                self.pointer_y = y;
                let target = hit_test(scene, x, y);

                // Handle enter/leave transitions.
                if target != self.hover {
                    if let Some(old) = self.hover {
                        emit(NodeEvent::PointerLeave { node: old });
                    }
                    if let Some(new) = target {
                        emit(NodeEvent::PointerEnter { node: new });
                    }
                    self.hover = target;
                }

                // Emit move on current hover target.
                if let Some(node) = target {
                    if let Some(r) = scene.rect_of(node) {
                        emit(NodeEvent::PointerMove {
                            node,
                            lx: x - r.x,
                            ly: y - r.y,
                        });
                    }
                }
            }

            InputEvent::PointerDown { x, y, button } => {
                self.pointer_x = x;
                self.pointer_y = y;
                let target = hit_test(scene, x, y);

                // Update hover (in case PointerDown arrives without
                // a prior PointerMove).
                if target != self.hover {
                    if let Some(old) = self.hover {
                        emit(NodeEvent::PointerLeave { node: old });
                    }
                    if let Some(new) = target {
                        emit(NodeEvent::PointerEnter { node: new });
                    }
                    self.hover = target;
                }

                // Press + focus transfer.
                if let Some(node) = target {
                    self.pressed = Some(node);

                    // Transfer focus on press.
                    if self.focus != Some(node) {
                        if let Some(old) = self.focus {
                            emit(NodeEvent::FocusOut { node: old });
                        }
                        self.focus = Some(node);
                        emit(NodeEvent::FocusIn { node });
                    }

                    if let Some(r) = scene.rect_of(node) {
                        emit(NodeEvent::PointerDown {
                            node,
                            lx: x - r.x,
                            ly: y - r.y,
                            button,
                        });
                    }
                } else {
                    // Clicked empty space — clear focus.
                    if let Some(old) = self.focus {
                        emit(NodeEvent::FocusOut { node: old });
                        self.focus = None;
                    }
                    self.pressed = None;
                }
            }

            InputEvent::PointerUp { x, y, button } => {
                self.pointer_x = x;
                self.pointer_y = y;

                // Deliver PointerUp to the node that received PointerDown
                // (not necessarily the node under the pointer now).
                if let Some(node) = self.pressed {
                    if let Some(r) = scene.rect_of(node) {
                        emit(NodeEvent::PointerUp {
                            node,
                            lx: x - r.x,
                            ly: y - r.y,
                            button,
                        });
                    }
                    self.pressed = None;
                }

                // Update hover for the release position.
                let target = hit_test(scene, x, y);
                if target != self.hover {
                    if let Some(old) = self.hover {
                        emit(NodeEvent::PointerLeave { node: old });
                    }
                    if let Some(new) = target {
                        emit(NodeEvent::PointerEnter { node: new });
                    }
                    self.hover = target;
                }
            }

            InputEvent::KeyDown { code } => {
                if let Some(node) = self.focus {
                    emit(NodeEvent::KeyDown { node, code });
                }
            }

            InputEvent::KeyUp { code } => {
                if let Some(node) = self.focus {
                    emit(NodeEvent::KeyUp { node, code });
                }
            }

            InputEvent::FocusNext => {
                self.cycle_focus(scene, true, &mut emit);
            }

            InputEvent::FocusPrev => {
                self.cycle_focus(scene, false, &mut emit);
            }
        }

        count
    }

    /// Cycle focus forward or backward through focusable nodes.
    ///
    /// Focusable = non-Composite, non-zero opacity, not hit-transparent,
    /// in paint order.
    fn cycle_focus<const N: usize>(
        &mut self,
        scene: &Scene<N>,
        forward: bool,
        emit: &mut dyn FnMut(NodeEvent),
    ) {
        let order = scene.order();
        if order.is_empty() {
            return;
        }

        // Scan the retained order directly; runtime-sized scenes need no
        // second focus table or temporary allocation.
        let focusable = order.iter().copied().filter(|&idx| {
            let node = scene.node(idx);
            node.opacity != 0
                && !matches!(node.bind, crate::bind::Bind::Composite)
                && !node
                    .flags
                    .contains(crate::scene::NodeFlags::HIT_TRANSPARENT)
        });
        let fcount = focusable.clone().count();
        if fcount == 0 {
            return;
        }

        // Find current focus position (a stale focus counts as none).
        let cur_pos = self
            .focus
            .and_then(|f| scene.index_of(f))
            .and_then(|f| focusable.clone().position(|n| n == f));

        let next_pos = match cur_pos {
            Some(pos) => {
                if forward {
                    (pos + 1) % fcount
                } else {
                    (pos + fcount - 1) % fcount
                }
            }
            None => {
                if forward {
                    0
                } else {
                    fcount - 1
                }
            }
        };

        let Some(next_node) = focusable
            .clone()
            .nth(next_pos)
            .and_then(|idx| scene.id_at(idx))
        else {
            return;
        };

        // Emit focus transitions.
        if let Some(old) = self.focus {
            if old != next_node {
                emit(NodeEvent::FocusOut { node: old });
            }
        }
        if self.focus != Some(next_node) {
            self.focus = Some(next_node);
            emit(NodeEvent::FocusIn { node: next_node });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::Bind;
    use crate::layout::{Anchor, LayoutParams, Offset, Rect};
    use crate::scene::{NodeFlags, Scene};

    /// The live id in slot `idx` (fixtures are built in slot order).
    fn id<const N: usize>(scene: &Scene<N>, idx: u16) -> NodeId {
        scene.id_at(idx).unwrap()
    }

    fn lp(x: i32, y: i32, w: u16, h: u16) -> LayoutParams {
        LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset { x, y },
            w,
            h,
        }
    }

    fn make_test_scene() -> Scene<64> {
        let mut scene = Scene::<64>::new();
        let root = scene
            .add_root(
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset::ZERO,
                    w: 100,
                    h: 100,
                },
                Bind::Composite,
            )
            .unwrap();

        // Node 1: red button at (10, 10) 30×20.
        scene
            .add_child(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x: 10, y: 10 },
                    w: 30,
                    h: 20,
                },
                Bind::Fill { color: 0x00FF0000 },
            )
            .unwrap();

        // Node 2: green button at (50, 10) 30×20.
        scene
            .add_child(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset { x: 50, y: 10 },
                    w: 30,
                    h: 20,
                },
                Bind::Fill { color: 0x0000FF00 },
            )
            .unwrap();

        scene
    }

    fn resolve<const N: usize>(scene: &mut Scene<N>) {
        use axnn_cpu::CpuBackend;
        let be = CpuBackend::<{ 64 * 1024 }>::new();
        let root_rect = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        scene.resolve_layout(root_rect, &be).unwrap();
    }

    #[test]
    fn hit_test_finds_frontmost() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        // Hit node 1 (red).
        assert_eq!(hit_test(&scene, 15, 15), Some(id(&scene, 1)));
        // Hit node 2 (green).
        assert_eq!(hit_test(&scene, 55, 15), Some(id(&scene, 2)));
        // Miss everything.
        assert_eq!(hit_test(&scene, 0, 0), None);
        // Just outside node 1.
        assert_eq!(hit_test(&scene, 9, 10), None);
    }

    #[test]
    fn hit_test_skips_zero_opacity() {
        let mut scene = Scene::<64>::new();
        let root = scene
            .add_root(
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset::ZERO,
                    w: 100,
                    h: 100,
                },
                Bind::Composite,
            )
            .unwrap();
        scene
            .add_child_with_opacity(
                root,
                LayoutParams {
                    anchor: Anchor::TOP_LEFT,
                    offset: Offset::ZERO,
                    w: 50,
                    h: 50,
                },
                Bind::Fill { color: 0x00FF0000 },
                0,
            )
            .unwrap();
        resolve(&mut scene);
        // Invisible node should not be hit.
        assert_eq!(hit_test(&scene, 10, 10), None);
    }

    #[test]
    fn pointer_enter_leave() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // Move onto node 1.
        let n = inter.feed(&scene, InputEvent::PointerMove { x: 15, y: 15 }, &mut out);
        assert!(n >= 1);
        assert_eq!(
            out[0],
            NodeEvent::PointerEnter {
                node: id(&scene, 1)
            }
        );
        assert_eq!(inter.hovered(), Some(id(&scene, 1)));

        // Move onto node 2 — should leave 1, enter 2.
        let n = inter.feed(&scene, InputEvent::PointerMove { x: 55, y: 15 }, &mut out);
        assert!(n >= 2);
        assert_eq!(
            out[0],
            NodeEvent::PointerLeave {
                node: id(&scene, 1)
            }
        );
        assert_eq!(
            out[1],
            NodeEvent::PointerEnter {
                node: id(&scene, 2)
            }
        );
        assert_eq!(inter.hovered(), Some(id(&scene, 2)));

        // Move to empty space — should leave 2.
        let n = inter.feed(&scene, InputEvent::PointerMove { x: 0, y: 0 }, &mut out);
        assert_eq!(n, 1);
        assert_eq!(
            out[0],
            NodeEvent::PointerLeave {
                node: id(&scene, 2)
            }
        );
        assert_eq!(inter.hovered(), None);
    }

    #[test]
    fn pointer_down_transfers_focus() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // Click node 1 — should enter + focus + down.
        let n = inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        assert!(n >= 2); // at least Enter + FocusIn + Down
        assert_eq!(inter.focused(), Some(id(&scene, 1)));
        assert_eq!(inter.pressed(), Some(id(&scene, 1)));

        // Click node 2 — should focus-out 1, focus-in 2.
        let n = inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 55,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        let events = &out[..n];
        assert!(events.contains(&NodeEvent::FocusOut {
            node: id(&scene, 1)
        }));
        assert!(events.contains(&NodeEvent::FocusIn {
            node: id(&scene, 2)
        }));
        assert_eq!(inter.focused(), Some(id(&scene, 2)));
    }

    #[test]
    fn pointer_up_delivered_to_pressed_node() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // Press on node 1.
        inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );

        // Release outside node 1 — PointerUp still delivered to node 1.
        let n = inter.feed(
            &scene,
            InputEvent::PointerUp {
                x: 0,
                y: 0,
                button: 0,
            },
            &mut out,
        );
        let events = &out[..n];
        assert!(events.contains(&NodeEvent::PointerUp {
            node: id(&scene, 1),
            lx: -10,
            ly: -10,
            button: 0,
        }));
        assert_eq!(inter.pressed(), None);
    }

    #[test]
    fn key_events_routed_to_focus() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // No focus — key events should produce nothing.
        let n = inter.feed(&scene, InputEvent::KeyDown { code: 42 }, &mut out);
        assert_eq!(n, 0);

        // Focus node 2, then send key.
        inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 55,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        let n = inter.feed(&scene, InputEvent::KeyDown { code: 42 }, &mut out);
        assert_eq!(n, 1);
        assert_eq!(
            out[0],
            NodeEvent::KeyDown {
                node: id(&scene, 2),
                code: 42
            }
        );
    }

    #[test]
    fn focus_cycling() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // FocusNext with no focus — should focus first focusable (node 1).
        let n = inter.feed(&scene, InputEvent::FocusNext, &mut out);
        assert_eq!(n, 1);
        assert_eq!(
            out[0],
            NodeEvent::FocusIn {
                node: id(&scene, 1)
            }
        );
        assert_eq!(inter.focused(), Some(id(&scene, 1)));

        // FocusNext — should move to node 2.
        let n = inter.feed(&scene, InputEvent::FocusNext, &mut out);
        assert!(n >= 2);
        assert!(out[..n].contains(&NodeEvent::FocusOut {
            node: id(&scene, 1)
        }));
        assert!(out[..n].contains(&NodeEvent::FocusIn {
            node: id(&scene, 2)
        }));
        assert_eq!(inter.focused(), Some(id(&scene, 2)));

        // FocusNext — should wrap to node 1.
        let n = inter.feed(&scene, InputEvent::FocusNext, &mut out);
        assert!(n >= 2);
        assert_eq!(inter.focused(), Some(id(&scene, 1)));

        // FocusPrev — should go back to node 2.
        let n = inter.feed(&scene, InputEvent::FocusPrev, &mut out);
        assert!(n >= 2);
        assert_eq!(inter.focused(), Some(id(&scene, 2)));
    }

    #[test]
    fn local_coordinates() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // Node 1 is at (10, 10). Pointer at (25, 18) → local (15, 8).
        let n = inter.feed(&scene, InputEvent::PointerMove { x: 25, y: 18 }, &mut out);
        let move_evt = out[..n]
            .iter()
            .find(|e| matches!(e, NodeEvent::PointerMove { .. }));
        assert_eq!(
            move_evt,
            Some(&NodeEvent::PointerMove {
                node: id(&scene, 1),
                lx: 15,
                ly: 8
            })
        );
    }

    #[test]
    fn click_empty_clears_focus() {
        let mut scene = make_test_scene();
        resolve(&mut scene);

        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];

        // Focus node 1.
        inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        assert_eq!(inter.focused(), Some(id(&scene, 1)));

        // Release.
        inter.feed(
            &scene,
            InputEvent::PointerUp {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );

        // Click empty space.
        let n = inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 0,
                y: 0,
                button: 0,
            },
            &mut out,
        );
        assert_eq!(inter.focused(), None);
        assert!(out[..n].contains(&NodeEvent::FocusOut {
            node: id(&scene, 1)
        }));
    }

    #[test]
    fn portable_seat_event_bridges_into_scene_input() {
        let event = axinput::SeatEvent {
            seat: axinput::SeatId(1),
            device: axinput::DeviceId(2),
            timestamp: Some(axinput::InputTimestamp(3)),
            position: axinput::PointerPosition::new(12, 34),
            kind: axinput::SeatEventKind::Button {
                button: axinput::PointerButton::PRIMARY,
                state: axinput::ButtonState::Pressed,
            },
        };

        assert_eq!(
            InputEvent::from_seat(event),
            Some(InputEvent::PointerDown {
                x: 12,
                y: 34,
                button: 0
            })
        );
    }

    #[test]
    fn hit_test_skips_hit_transparent() {
        let mut scene = make_test_scene();
        let overlay = scene
            .add_node(
                Some(id(&scene, 0)),
                lp(0, 0, 100, 100),
                Bind::Fill { color: 0x00000000 },
                255,
                NodeFlags::HIT_TRANSPARENT,
            )
            .unwrap();
        resolve(&mut scene);
        // The overlay paints last but the pointer looks through it.
        assert_eq!(hit_test(&scene, 15, 15), Some(id(&scene, 1)));
        assert_eq!(hit_test(&scene, 0, 0), None);
        scene.set_flags(overlay, NodeFlags::NONE).unwrap();
        resolve(&mut scene);
        assert_eq!(hit_test(&scene, 15, 15), Some(overlay));
    }

    #[test]
    fn hit_test_rejects_points_outside_ancestor_clip() {
        let mut scene = Scene::<8>::new();
        let root = scene.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let panel = scene
            .add_node(
                Some(root),
                lp(10, 10, 20, 20),
                Bind::Composite,
                255,
                NodeFlags::CLIP_CHILDREN,
            )
            .unwrap();
        // The child overhangs its clipping parent on every side.
        let child = scene
            .add_child(
                panel,
                lp(-10, -10, 40, 40),
                Bind::Fill { color: 0x00FF0000 },
            )
            .unwrap();
        resolve(&mut scene);
        assert_eq!(
            scene.rect_of(child),
            Some(Rect {
                x: 0,
                y: 0,
                w: 40,
                h: 40
            })
        );
        assert_eq!(hit_test(&scene, 15, 15), Some(child));
        assert_eq!(hit_test(&scene, 5, 5), None);
        assert_eq!(hit_test(&scene, 35, 15), None);
        assert_eq!(hit_test(&scene, 29, 29), Some(child));
        assert_eq!(hit_test(&scene, 30, 30), None);
    }

    #[test]
    fn rounded_corner_misses() {
        let mut scene = Scene::<8>::new();
        let root = scene.add_root(lp(0, 0, 100, 100), Bind::Composite).unwrap();
        let pill = scene
            .add_child(
                root,
                lp(10, 10, 40, 20),
                Bind::RoundedFill {
                    color: 0x00FF0000,
                    radii: crate::bind::Corners::uniform(8),
                },
            )
            .unwrap();
        resolve(&mut scene);
        // Corner pixels lie outside the arc; the straight edges do not.
        assert_eq!(hit_test(&scene, 10, 10), None);
        assert_eq!(hit_test(&scene, 49, 10), None);
        assert_eq!(hit_test(&scene, 49, 29), None);
        assert_eq!(hit_test(&scene, 10, 29), None);
        assert_eq!(hit_test(&scene, 30, 10), Some(pill));
        assert_eq!(hit_test(&scene, 10, 20), Some(pill));
        assert_eq!(hit_test(&scene, 17, 17), Some(pill));
        // Along the diagonal with r=8: pixel (1,1) from the corner has its
        // centre outside the arc, pixel (2,2) inside.
        assert_eq!(hit_test(&scene, 11, 11), None);
        assert_eq!(hit_test(&scene, 12, 12), Some(pill));

        // A square `Surface` window with HIT_ROUNDED takes its shape from
        // `Node::radii`; a plain Fill with no flag keeps square corners.
        let window = scene
            .add_node(
                Some(root),
                lp(60, 60, 30, 30),
                Bind::Surface { id: 0 },
                255,
                NodeFlags::HIT_ROUNDED,
            )
            .unwrap();
        scene
            .set_radii(
                window,
                crate::bind::Corners {
                    tl: 10,
                    tr: 0,
                    br: 0,
                    bl: 0,
                },
            )
            .unwrap();
        resolve(&mut scene);
        assert_eq!(hit_test(&scene, 60, 60), None);
        assert_eq!(hit_test(&scene, 89, 60), Some(window));
        assert_eq!(hit_test(&scene, 60, 89), Some(window));
    }

    #[test]
    fn rounded_contains_matches_row_inset_rule() {
        let rect = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let radii = crate::bind::Corners::uniform(4);
        // Row 0: an inset of 2 is the first pixel inside a radius-4 arc.
        assert!(!rounded_contains(rect, radii, 1, 0));
        assert!(rounded_contains(rect, radii, 2, 0));
        assert!(rounded_contains(rect, radii, 13, 0));
        assert!(!rounded_contains(rect, radii, 14, 0));
        // Rows past the radius are full width.
        assert!(rounded_contains(rect, radii, 0, 4));
        assert!(rounded_contains(rect, radii, 15, 11));
        assert!(!rounded_contains(rect, radii, 16, 4));
        // Oversized radii clamp to the pill.
        assert!(rounded_contains(
            rect,
            crate::bind::Corners::uniform(200),
            8,
            8
        ));
    }

    #[test]
    fn removed_pressed_node_is_dropped() {
        let mut scene = make_test_scene();
        resolve(&mut scene);
        let mut inter = Interaction::new();
        let mut out = [NodeEvent::NONE; 8];
        inter.feed(
            &scene,
            InputEvent::PointerDown {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        let btn = id(&scene, 1);
        assert_eq!(inter.pressed(), Some(btn));
        scene.remove(btn).unwrap();
        resolve(&mut scene);
        let n = inter.feed(
            &scene,
            InputEvent::PointerUp {
                x: 15,
                y: 15,
                button: 0,
            },
            &mut out,
        );
        assert!(
            !out[..n]
                .iter()
                .any(|e| matches!(e, NodeEvent::PointerUp { .. }))
        );
        assert_eq!(inter.pressed(), None);
        // The stale focus is skipped by focus cycling.
        assert_eq!(inter.focused(), Some(btn));
        let n = inter.feed(&scene, InputEvent::FocusNext, &mut out);
        assert!(n >= 1);
        assert_eq!(inter.focused(), Some(id(&scene, 2)));
    }
}

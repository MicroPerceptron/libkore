//! One routed-expert wave on the grouped GPU kernels
//! (`kore_xe_kernels::moe_wave`; res/XE_MOE_OVERHAUL.md, Tables): where each
//! expert's AXEX v3 block is bound, and the region the wave's kernels share.
//!
//! The region, in the executor's scratch, is `u32` words: one 16-word header
//! per chunk set, the group, route, tile, token and list records every launch
//! shares, then the F32 sections HIDDEN `[T, H]`, ACT `[R, I]`, CONTRIB
//! `[R, H]` and OUT `[T, H]`. Chunk set `j`'s grouped up and down launches
//! bind the region from header `j` (64 bytes in per set), and each header
//! gives every offset relative to its own base; the combine binds it from
//! header 0. A set is up to [`CHUNK_SLOTS`] of the expert-cache imports the
//! wave's blocks live in, bound whole, so a wave whose experts span more
//! imports takes more launch pairs but still one combine, which adds each
//! token's routes in expert order as the CPU does.
//!
//! The kernels trust every word: these checks are their only bounds.

use axnn_cpu::{ExpertWeights, MAX_WAVE_EXPERTS, MoeBatchRow};

/// The expert shape the sealed wave packages are built for.
pub const MOE_HIDDEN: usize = 2688;
pub const MOE_INTERMEDIATE: usize = 1856;
/// Expert-cache imports one grouped launch binds.
pub const CHUNK_SLOTS: usize = 5;
/// Launch pairs one wave may need: every expert in an import of its own.
pub const MAX_SETS: usize = MAX_WAVE_EXPERTS.div_ceil(CHUNK_SLOTS);
/// Words of one launch's header; its binding starts this far past the last.
pub const HEADER_WORDS: usize = 16;
/// Routes one tile covers: the kernels' eight A rows.
const TILE_ROUTES: usize = 8;
const PAGE: usize = 4096;
/// The largest storage binding the Xe surface state describes (4 GiB less
/// its low padding bits, `MAX_RAW_BUFFER_BYTES`): a chunk slot binds its
/// whole import, so an import past this cannot be bound.
pub const MAX_CHUNK_BYTES: u64 = (1 << 32) - 4096;

/// Where an AXEX v3 block's sections sit, by the v3 rule (axmodel's
/// `ExpertSections`): up codes at 0, up block scales on the next page, the
/// two F32 tensor scales right after them, down codes and down block scales
/// each on the next page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSections {
    pub up_scales: usize,
    pub tensor_scales: usize,
    pub down_codes: usize,
    pub down_scales: usize,
    /// The end of the down block scales: what a block must hold.
    pub used: usize,
}

impl BlockSections {
    pub const fn new(inner: usize, hidden: usize) -> Self {
        const fn page(bytes: usize) -> usize {
            bytes.div_ceil(PAGE) * PAGE
        }
        let up_scales = page(inner * hidden / 2);
        let tensor_scales = up_scales + inner * hidden / 16;
        let down_codes = page(tensor_scales + 8);
        let down_scales = page(down_codes + hidden * inner / 2);
        Self {
            up_scales,
            tensor_scales,
            down_codes,
            down_scales,
            used: down_scales + hidden * inner / 16,
        }
    }
}

/// The sealed packages' block layout.
pub const SECTIONS: BlockSections = BlockSections::new(MOE_INTERMEDIATE, MOE_HIDDEN);

/// Why a wave stays on the CPU. Nothing was staged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveRefusal {
    /// The routes, experts or tokens are not a wave of the sealed shape, or
    /// an expert's views are not one AXEX v3 block.
    Geometry,
    /// A block lies outside every resident import, or runs past one.
    NotResident,
}

/// A wave's chunk sets, tiles and region layout, checked and not yet
/// written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WavePlan {
    sets: usize,
    /// Per set, its imports as `(buffer, bytes)`, `None` for an unused slot.
    /// The provider refuses a launch whose bindings overlap, so an unused
    /// slot must not repeat a used one: the executor binds it to a page of
    /// its own.
    chunks: [[Option<(u64, u64)>; CHUNK_SLOTS]; MAX_SETS],
    /// Per set, its first tile and tile count.
    tiles: [(usize, usize); MAX_SETS],
    /// Per expert (pick), its set, slot and first page.
    placement: [(u8, u8, u32); MAX_WAVE_EXPERTS],
    tokens: usize,
    routes: usize,
    groups: usize,
    /// Absolute word offsets of the records and sections, and the end.
    records: [usize; 5],
    sections: [usize; 4],
    words: usize,
}

const GROUPS: usize = 0;
const ROUTES: usize = 1;
const TILE: usize = 2;
const TOKENS: usize = 3;
const LIST: usize = 4;
const HIDDEN: usize = 0;
const ACT: usize = 1;
const CONTRIB: usize = 2;
const OUT: usize = 3;

/// The same checked layout drives capacity sizing and actual admission.
/// All shader offsets/counts are u32 words and each raw storage binding must
/// fit Xe's surface limit. Reject before any allocation or publication.
fn wave_layout(
    tokens: usize,
    routes: usize,
    groups: usize,
    tiles: usize,
    sets: usize,
) -> Option<([usize; 5], [usize; 4], usize)> {
    wave_layout_from(sets.checked_mul(HEADER_WORDS)?, tokens, routes, groups, tiles)
}

/// [`wave_layout`]'s records and sections from word `start` on.
fn wave_layout_from(
    start: usize,
    tokens: usize,
    routes: usize,
    groups: usize,
    tiles: usize,
) -> Option<([usize; 5], [usize; 4], usize)> {
    let mut records = [0; 5];
    let mut sections = [0; 4];
    let mut cursor = start;
    for (record, count, stride) in [
        (GROUPS, groups, 4),
        (ROUTES, routes, 2),
        (TILE, tiles, 2),
        (TOKENS, tokens, 2),
        (LIST, routes, 1),
    ] {
        records[record] = cursor;
        cursor = cursor.checked_add(count.checked_mul(stride)?)?;
    }
    for (section, count, stride) in [
        (HIDDEN, tokens, MOE_HIDDEN),
        (ACT, routes, MOE_INTERMEDIATE),
        (CONTRIB, routes, MOE_HIDDEN),
        (OUT, tokens, MOE_HIDDEN),
    ] {
        cursor = cursor.checked_next_multiple_of(HEADER_WORDS)?;
        sections[section] = cursor;
        cursor = cursor.checked_add(count.checked_mul(stride)?)?;
    }
    let bytes = cursor.checked_mul(size_of::<u32>())?;
    (bytes as u64 <= MAX_CHUNK_BYTES).then_some((records, sections, cursor))
}

impl WavePlan {
    /// Largest expert prefix whose worst-case resident-import layout fits.
    /// This runs before the cache loads weights, so exact placements are not
    /// known yet. The number of resident imports bounds the number of chunk
    /// sets; the later admission validates the exact layout and packages.
    /// `fits` checks the caller's current command and scratch budgets without
    /// allocating or reserving either. Route/expert order remains unchanged.
    pub fn capacity(
        routes: &[MoeBatchRow],
        tokens: usize,
        resident_imports: usize,
        mut fits: impl FnMut(usize, u64) -> bool,
    ) -> usize {
        if tokens == 0 || resident_imports == 0 {
            return 0;
        }
        let (mut start, mut groups, mut tiles, mut accepted) = (0, 0, 0, 0);
        while start < routes.len() && groups < MAX_WAVE_EXPERTS {
            let count = group_len(routes, start);
            start += count;
            groups += 1;
            tiles += count.div_ceil(TILE_ROUTES);
            let sets = groups.min(resident_imports).div_ceil(CHUNK_SLOTS);
            let Some((_, _, words)) = wave_layout(tokens, start, groups, tiles, sets) else {
                break;
            };
            if !fits(sets, (words * size_of::<u32>()) as u64) {
                break;
            }
            accepted = groups;
        }
        accepted
    }

    /// Plan `routes` (grouped by expert, ascending; the wave's `n`th expert
    /// is `experts[n]`) over `tokens` tokens. `resolve(address)` names the
    /// resident import holding a block's first byte as `(buffer, offset,
    /// import bytes)`.
    pub fn new(
        routes: &[MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        tokens: usize,
        mut resolve: impl FnMut(usize) -> Option<(u64, u64, u64)>,
    ) -> Result<Self, WaveRefusal> {
        use WaveRefusal::{Geometry, NotResident};
        let (codes, scales) = (
            MOE_INTERMEDIATE * MOE_HIDDEN / 2,
            MOE_INTERMEDIATE * MOE_HIDDEN / 16,
        );
        if routes.is_empty()
            || experts.is_empty()
            || experts.len() > MAX_WAVE_EXPERTS
            || tokens == 0
            || u32::try_from(routes.len()).is_err()
            || u32::try_from(tokens).is_err()
        {
            return Err(Geometry);
        }
        let mut plan = Self {
            sets: 0,
            chunks: [[None; CHUNK_SLOTS]; MAX_SETS],
            tiles: [(0, 0); MAX_SETS],
            placement: [(0, 0, 0); MAX_WAVE_EXPERTS],
            tokens,
            routes: routes.len(),
            groups: experts.len(),
            records: [0; 5],
            sections: [0; 4],
            words: 0,
        };
        // Imports in order of first use: import `n` is set `n / 5`, slot
        // `n % 5`.
        let mut imports = [(0u64, 0u64); MAX_WAVE_EXPERTS];
        let mut used = 0;
        for (pick, expert) in experts.iter().enumerate() {
            let base = expert.up.as_ptr() as usize;
            let at = |view: &[u8]| (view.as_ptr() as usize).wrapping_sub(base);
            if expert.up.len() != codes
                || expert.down.len() != codes
                || expert.up_block_scales.len() != scales
                || expert.down_block_scales.len() != scales
                || at(expert.up_block_scales) != SECTIONS.up_scales
                || at(expert.down) != SECTIONS.down_codes
                || at(expert.down_block_scales) != SECTIONS.down_scales
                || !expert.up_scale.is_finite()
                || !expert.down_scale.is_finite()
            {
                return Err(Geometry);
            }
            let (buffer, offset, bytes) = resolve(base).ok_or(NotResident)?;
            if buffer == 0
                || offset % PAGE as u64 != 0
                || bytes > MAX_CHUNK_BYTES
                || offset
                    .checked_add(SECTIONS.used as u64)
                    .is_none_or(|end| end > bytes)
            {
                return Err(NotResident);
            }
            let import = match imports[..used].iter().position(|&(b, _)| b == buffer) {
                Some(import) => import,
                None => {
                    imports[used] = (buffer, bytes);
                    used += 1;
                    used - 1
                }
            };
            let page = u32::try_from(offset / PAGE as u64).map_err(|_| NotResident)?;
            plan.placement[pick] = (
                (import / CHUNK_SLOTS) as u8,
                (import % CHUNK_SLOTS) as u8,
                page,
            );
        }
        plan.sets = used.div_ceil(CHUNK_SLOTS);
        for set in 0..plan.sets {
            let members = &imports[set * CHUNK_SLOTS..used.min((set + 1) * CHUNK_SLOTS)];
            for slot in 0..CHUNK_SLOTS {
                plan.chunks[set][slot] = members.get(slot).copied();
            }
        }
        // Routes grouped by expert, ascending, tokens ascending within one,
        // and exactly one group per expert.
        let mut group = 0;
        for (index, route) in routes.iter().enumerate() {
            if route.token() >= tokens || !route.mixture().is_finite() {
                return Err(Geometry);
            }
            if index > 0 {
                let previous = routes[index - 1];
                if route.expert() != previous.expert() {
                    if route.expert() < previous.expert() {
                        return Err(Geometry);
                    }
                    group += 1;
                } else if route.token() <= previous.token() {
                    return Err(Geometry);
                }
            }
        }
        if group + 1 != experts.len() {
            return Err(Geometry);
        }
        // Tiles, set by set.
        let mut tiles = 0;
        for set in 0..plan.sets {
            let first = tiles;
            let mut start = 0;
            for pick in 0..experts.len() {
                let count = group_len(routes, start);
                if usize::from(plan.placement[pick].0) == set {
                    tiles += count.div_ceil(TILE_ROUTES);
                }
                start += count;
            }
            plan.tiles[set] = (first, tiles - first);
        }
        (plan.records, plan.sections, plan.words) =
            wave_layout(tokens, routes.len(), experts.len(), tiles, plan.sets).ok_or(Geometry)?;
        Ok(plan)
    }

    /// Launch pairs (grouped up and down) the wave takes.
    pub fn sets(&self) -> usize {
        self.sets
    }

    /// Set `set`'s imports, `(buffer, bytes)` per slot, `None` unused.
    pub fn chunks(&self, set: usize) -> [Option<(u64, u64)>; CHUNK_SLOTS] {
        self.chunks[set]
    }

    /// Set `set`'s tile count: its grouped launches' Y extent.
    pub fn tiles(&self, set: usize) -> usize {
        self.tiles[set].1
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The region's bytes.
    pub fn bytes(&self) -> u64 {
        (self.words * 4) as u64
    }

    /// Byte offset of launch `set`'s binding in the region.
    pub fn header_offset(set: usize) -> u64 {
        (set * HEADER_WORDS * 4) as u64
    }

    /// Byte offset of OUT in the region, and its bytes.
    pub fn out(&self) -> (u64, u64) {
        (
            (self.sections[OUT] * 4) as u64,
            (self.tokens * MOE_HIDDEN * 4) as u64,
        )
    }

    /// Write the region: headers, records, `hidden` into HIDDEN and
    /// `output` (what earlier waves left) into OUT, which the combine adds
    /// onto. ACT and CONTRIB are written by the kernels before any read.
    pub fn fill(
        &self,
        words: &mut [u32],
        routes: &[MoeBatchRow],
        hidden: &[f32],
        output: &[f32],
    ) -> Result<(), WaveRefusal> {
        let rows = self.tokens * MOE_HIDDEN;
        if words.len() != self.words
            || routes.len() != self.routes
            || hidden.len() != rows
            || output.len() != rows
        {
            return Err(WaveRefusal::Geometry);
        }
        let [groups, route_at, tile_at, token_at, list_at] = self.records;
        for set in 0..self.sets {
            let base = set * HEADER_WORDS;
            let relative = |absolute: usize| (absolute - base) as u32;
            let (first, count) = self.tiles[set];
            let header = [
                first as u32,
                count as u32,
                self.groups as u32,
                self.routes as u32,
                self.tokens as u32,
                1, // the combine adds onto OUT
                relative(groups),
                relative(route_at),
                relative(tile_at),
                relative(token_at),
                relative(list_at),
                relative(self.sections[HIDDEN]),
                relative(self.sections[ACT]),
                relative(self.sections[CONTRIB]),
                relative(self.sections[OUT]),
                0,
            ];
            words[base..base + HEADER_WORDS].copy_from_slice(&header);
        }
        // Groups, and each set's tiles in pick order.
        let mut start = 0;
        let mut next_tile = self.tiles.map(|(first, _)| first);
        for pick in 0..self.groups {
            let count = group_len(routes, start);
            let (set, slot, page) = self.placement[pick];
            let at = groups + 4 * pick;
            words[at..at + 4].copy_from_slice(&[u32::from(slot), page, start as u32, count as u32]);
            for first in (start..start + count).step_by(TILE_ROUTES) {
                let at = tile_at + 2 * next_tile[usize::from(set)];
                words[at..at + 2].copy_from_slice(&[pick as u32, first as u32]);
                next_tile[usize::from(set)] += 1;
            }
            start += count;
        }
        for (index, route) in routes.iter().enumerate() {
            let at = route_at + 2 * index;
            words[at..at + 2].copy_from_slice(&[route.token() as u32, route.mixture().to_bits()]);
        }
        // Each token's routes, in route order, which is expert order (the
        // CPU's): count them, give each token its first list slot, place.
        words[token_at..token_at + 2 * self.tokens].fill(0);
        for route in routes {
            words[token_at + 2 * route.token() + 1] += 1;
        }
        let mut first = 0;
        for token in 0..self.tokens {
            words[token_at + 2 * token] = first;
            first += core::mem::take(&mut words[token_at + 2 * token + 1]);
        }
        for (index, route) in routes.iter().enumerate() {
            let record = token_at + 2 * route.token();
            let slot = (words[record] + words[record + 1]) as usize;
            words[list_at + slot] = index as u32;
            words[record + 1] += 1;
        }
        let f32_words = |values: &[f32], at: usize, words: &mut [u32]| {
            for (word, value) in words[at..at + values.len()].iter_mut().zip(values) {
                *word = value.to_bits();
            }
        };
        f32_words(hidden, self.sections[HIDDEN], words);
        f32_words(output, self.sections[OUT], words);
        Ok(())
    }
}

/// Experts of a layer the device routes itself (`xe-moe-plan.comp`).
pub const ROUTED_EXPERTS: usize = 128;
/// Each token's picks, as the plan kernel is built.
pub const ROUTED_TOP_K: usize = 6;
/// Tokens one plan routes (its workgroup's token bitmap).
pub const ROUTED_MAX_TOKENS: usize = 512;
/// The plan's control block, right after header 0.
const ROUTED_CONTROL: usize = HEADER_WORDS;
const ROUTED_CONTROL_WORDS: usize = 16;

/// A wave the device plans for itself: every expert of a resident layer in
/// one launch set, the router's scores read where they were produced. The
/// region is laid out for the worst case of `tokens` tokens; the host
/// writes only header 0's offsets and FLAGS, the plan's control block, the
/// correction bias and the expert directory ([`Self::fill`]), and the plan
/// kernel writes the counts, the records and HIDDEN that the grouped up,
/// down and combine kernels then read exactly as a host-planned wave's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedPlan {
    tokens: usize,
    /// The layer's imports, `(buffer, bytes)` per slot, `None` unused.
    chunks: [Option<(u64, u64)>; CHUNK_SLOTS],
    /// Per expert, its slot and first page.
    directory: [(u8, u32); ROUTED_EXPERTS],
    max_tiles: usize,
    bias: usize,
    directory_at: usize,
    records: [usize; 5],
    sections: [usize; 4],
    words: usize,
}

impl RoutedPlan {
    /// The plan for `tokens` tokens over `experts` (every expert of the
    /// layer, in expert order, resident). `resolve(address)` names the
    /// resident import holding a block's first byte as `(buffer, offset,
    /// import bytes)`; the layer's blocks must lie in at most
    /// [`CHUNK_SLOTS`] imports.
    pub fn new(
        experts: &[ExpertWeights<'_>],
        tokens: usize,
        mut resolve: impl FnMut(usize) -> Option<(u64, u64, u64)>,
    ) -> Result<Self, WaveRefusal> {
        use WaveRefusal::{Geometry, NotResident};
        let (codes, scales) = (
            MOE_INTERMEDIATE * MOE_HIDDEN / 2,
            MOE_INTERMEDIATE * MOE_HIDDEN / 16,
        );
        if experts.len() != ROUTED_EXPERTS || tokens == 0 || tokens > ROUTED_MAX_TOKENS {
            return Err(Geometry);
        }
        let mut chunks = [None; CHUNK_SLOTS];
        let mut directory = [(0u8, 0u32); ROUTED_EXPERTS];
        let mut used = 0;
        for (entry, expert) in directory.iter_mut().zip(experts) {
            let base = expert.up.as_ptr() as usize;
            let at = |view: &[u8]| (view.as_ptr() as usize).wrapping_sub(base);
            if expert.up.len() != codes
                || expert.down.len() != codes
                || expert.up_block_scales.len() != scales
                || expert.down_block_scales.len() != scales
                || at(expert.up_block_scales) != SECTIONS.up_scales
                || at(expert.down) != SECTIONS.down_codes
                || at(expert.down_block_scales) != SECTIONS.down_scales
                || !expert.up_scale.is_finite()
                || !expert.down_scale.is_finite()
            {
                return Err(Geometry);
            }
            let (buffer, offset, bytes) = resolve(base).ok_or(NotResident)?;
            if buffer == 0
                || offset % PAGE as u64 != 0
                || bytes > MAX_CHUNK_BYTES
                || offset
                    .checked_add(SECTIONS.used as u64)
                    .is_none_or(|end| end > bytes)
            {
                return Err(NotResident);
            }
            let slot = match chunks[..used].iter().position(|c| *c == Some((buffer, bytes))) {
                Some(slot) => slot,
                None if used < CHUNK_SLOTS => {
                    chunks[used] = Some((buffer, bytes));
                    used += 1;
                    used - 1
                }
                None => return Err(NotResident),
            };
            *entry = (
                slot as u8,
                u32::try_from(offset / PAGE as u64).map_err(|_| NotResident)?,
            );
        }
        let routes = tokens * ROUTED_TOP_K;
        let groups = ROUTED_EXPERTS.min(routes);
        // A group of c routes has ceil(c / 8) tiles: at most (R + 7G) / 8.
        let max_tiles = (routes + 7 * groups) / TILE_ROUTES;
        let bias = ROUTED_CONTROL + ROUTED_CONTROL_WORDS;
        let directory_at = bias + ROUTED_EXPERTS;
        let (records, sections, words) = wave_layout_from(
            directory_at + 2 * ROUTED_EXPERTS,
            tokens,
            routes,
            groups,
            max_tiles,
        )
        .ok_or(Geometry)?;
        Ok(Self {
            tokens,
            chunks,
            directory,
            max_tiles,
            bias,
            directory_at,
            records,
            sections,
            words,
        })
    }

    /// The layer's imports, `(buffer, bytes)` per slot, `None` unused.
    pub fn chunks(&self) -> [Option<(u64, u64)>; CHUNK_SLOTS] {
        self.chunks
    }

    /// The grouped launches' Y extent: tile workgroups past the plan's
    /// count return at once.
    pub fn max_tiles(&self) -> usize {
        self.max_tiles
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The region's bytes.
    pub fn bytes(&self) -> u64 {
        (self.words * 4) as u64
    }

    /// Byte offset of OUT in the region, and its bytes.
    pub fn out(&self) -> (u64, u64) {
        (
            (self.sections[OUT] * 4) as u64,
            (self.tokens * MOE_HIDDEN * 4) as u64,
        )
    }

    /// Write what the host owns of the region: header 0's offsets, FLAGS
    /// (the combine overwrites OUT: this wave is the layer's only one), the
    /// plan's control block, `bias` and the directory. Every other word is
    /// the plan's or a kernel's to write before any read.
    pub fn fill(
        &self,
        words: &mut [u32],
        bias: &[f32],
        scale: f32,
        normalize: bool,
    ) -> Result<(), WaveRefusal> {
        if words.len() != self.words || bias.len() != ROUTED_EXPERTS || !scale.is_finite() {
            return Err(WaveRefusal::Geometry);
        }
        let [groups, routes, tiles, tokens, list] = self.records;
        let header = &mut words[..HEADER_WORDS];
        header[..6].fill(0);
        header[6..].copy_from_slice(&[
            groups as u32,
            routes as u32,
            tiles as u32,
            tokens as u32,
            list as u32,
            self.sections[HIDDEN] as u32,
            self.sections[ACT] as u32,
            self.sections[CONTRIB] as u32,
            self.sections[OUT] as u32,
            0,
        ]);
        let control = &mut words[ROUTED_CONTROL..ROUTED_CONTROL + ROUTED_CONTROL_WORDS];
        control.fill(0);
        control[..6].copy_from_slice(&[
            self.tokens as u32,
            scale.to_bits(),
            u32::from(normalize),
            self.bias as u32,
            self.directory_at as u32,
            self.max_tiles as u32,
        ]);
        for (word, value) in words[self.bias..self.bias + ROUTED_EXPERTS].iter_mut().zip(bias) {
            *word = value.to_bits();
        }
        for (expert, &(slot, page)) in self.directory.iter().enumerate() {
            let at = self.directory_at + 2 * expert;
            words[at..at + 2].copy_from_slice(&[u32::from(slot), page]);
        }
        Ok(())
    }
}

/// Routes in the group starting at `start`.
fn group_len(routes: &[MoeBatchRow], start: usize) -> usize {
    let expert = routes[start].expert();
    routes[start..].partition_point(|route| route.expert() == expert)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// The v3 stride: the sections page-rounded.
    const STRIDE: usize = 5_619_712;

    /// `count` AXEX v3 blocks, back to back (zeroed, so untouched pages cost
    /// nothing), and each block's expert views.
    fn blocks(count: usize) -> Vec<u8> {
        vec![0u8; count * STRIDE]
    }

    fn expert(blocks: &[u8], index: usize) -> ExpertWeights<'_> {
        let block = &blocks[index * STRIDE..(index + 1) * STRIDE];
        let (codes, scales) = (
            MOE_INTERMEDIATE * MOE_HIDDEN / 2,
            MOE_INTERMEDIATE * MOE_HIDDEN / 16,
        );
        ExpertWeights {
            up: &block[..codes],
            up_block_scales: &block[SECTIONS.up_scales..][..scales],
            down: &block[SECTIONS.down_codes..][..codes],
            down_block_scales: &block[SECTIONS.down_scales..][..scales],
            up_scale: 1.0,
            down_scale: 2.0,
        }
    }

    /// Imports of `per` blocks each: buffer `1 + n` holds blocks
    /// `n * per ..`.
    fn imports(blocks: &[u8], per: usize) -> impl FnMut(usize) -> Option<(u64, u64, u64)> + '_ {
        let base = blocks.as_ptr() as usize;
        move |address| {
            let offset = address.checked_sub(base)?;
            if offset >= blocks.len() {
                return None;
            }
            let import = offset / (per * STRIDE);
            let bytes = (per * STRIDE).min(blocks.len() - import * per * STRIDE);
            Some((
                1 + import as u64,
                (offset - import * per * STRIDE) as u64,
                bytes as u64,
            ))
        }
    }

    fn route(expert: u32, token: u32, mixture: f32) -> MoeBatchRow {
        MoeBatchRow::new(expert, token, mixture)
    }

    /// A resident layer's directory names each expert's slot and page in
    /// the layer's at most five imports, the region holds the worst case,
    /// and the host writes header 0's offsets, the control block, the bias
    /// and the directory, nothing else. Six imports, or a short layer, are
    /// refused.
    #[test]
    fn a_routed_plan_lays_out_the_worst_case_and_writes_only_its_inputs() {
        let blocks = blocks(ROUTED_EXPERTS);
        let experts: Vec<_> = (0..ROUTED_EXPERTS).map(|e| expert(&blocks, e)).collect();
        let tokens = 8;
        let plan = RoutedPlan::new(&experts, tokens, imports(&blocks, 30)).unwrap();
        // 128 experts at 30 per import: five imports.
        assert_eq!(plan.chunks().iter().flatten().count(), 5);
        let routes = tokens * ROUTED_TOP_K;
        assert_eq!(plan.max_tiles(), (routes + 7 * routes) / 8);
        for (e, &(slot, page)) in plan.directory.iter().enumerate() {
            assert_eq!(usize::from(slot), e / 30);
            assert_eq!(page as usize, (e % 30) * STRIDE / PAGE);
        }
        let mut words = vec![0xa5a5_a5a5u32; plan.bytes() as usize / 4];
        let bias: Vec<f32> = (0..ROUTED_EXPERTS).map(|e| e as f32 / 64.0 - 1.0).collect();
        plan.fill(&mut words, &bias, 2.5, true).unwrap();
        assert_eq!(&words[..6], &[0; 6], "the plan writes the counts; FLAGS overwrite");
        let [groups, routes_at, tiles, tokens_at, list] = plan.records;
        assert!(groups >= plan.directory_at + 2 * ROUTED_EXPERTS);
        assert_eq!(
            &words[6..16],
            &[
                groups as u32,
                routes_at as u32,
                tiles as u32,
                tokens_at as u32,
                list as u32,
                plan.sections[HIDDEN] as u32,
                plan.sections[ACT] as u32,
                plan.sections[CONTRIB] as u32,
                plan.sections[OUT] as u32,
                0
            ]
        );
        assert_eq!(
            &words[16..22],
            &[8, 2.5f32.to_bits(), 1, plan.bias as u32, plan.directory_at as u32, plan.max_tiles as u32]
        );
        assert_eq!(&words[22..32], &[0; 10]);
        assert_eq!(words[plan.bias + 5], bias[5].to_bits());
        assert_eq!(words[plan.directory_at + 2 * 31..][..2], [1, (STRIDE / PAGE) as u32]);
        assert!(words[groups..].iter().all(|&w| w == 0xa5a5_a5a5), "records and sections untouched");
        let (out, out_bytes) = plan.out();
        assert_eq!(out_bytes, (tokens * MOE_HIDDEN * 4) as u64);
        assert!(out + out_bytes <= plan.bytes());

        assert_eq!(
            RoutedPlan::new(&experts, tokens, imports(&blocks, 25)),
            Err(WaveRefusal::NotResident),
            "six imports"
        );
        assert_eq!(
            RoutedPlan::new(&experts[..ROUTED_EXPERTS - 1], tokens, imports(&blocks, 30)),
            Err(WaveRefusal::Geometry)
        );
        assert_eq!(
            RoutedPlan::new(&experts, ROUTED_MAX_TOKENS + 1, imports(&blocks, 30)),
            Err(WaveRefusal::Geometry)
        );
    }

    #[test]
    fn sections_are_the_axex_v3_rule() {
        assert_eq!(
            SECTIONS,
            BlockSections {
                up_scales: 2_494_464,
                tensor_scales: 2_806_272,
                down_codes: 2_809_856,
                down_scales: 5_304_320,
                used: 5_304_320 + 311_808,
            }
        );
        assert_eq!(SECTIONS.used.div_ceil(PAGE) * PAGE, STRIDE);
    }

    #[test]
    fn capacity_stops_at_command_scratch_and_metadata_bounds() {
        let routes: Vec<_> = (0..MAX_WAVE_EXPERTS + 3)
            .map(|e| route(e as u32, 0, 1.0))
            .collect();
        assert_eq!(
            WavePlan::capacity(&routes, 1, 8, |_, _| true),
            MAX_WAVE_EXPERTS
        );
        // Worst case: one import per expert. Two chunk sets hold ten.
        assert_eq!(
            WavePlan::capacity(&routes, 1, routes.len(), |sets, _| sets <= 2),
            2 * CHUNK_SLOTS
        );
        let seven_bytes = wave_layout(1, 7, 7, 7, 1).unwrap().2 * 4;
        assert_eq!(
            WavePlan::capacity(&routes, 1, 1, |_, bytes| bytes <= seven_bytes as u64),
            7
        );
        assert_eq!(WavePlan::capacity(&routes, 1, 1, |_, _| false), 0);
        assert_eq!(WavePlan::capacity(&routes, 1, 0, |_, _| true), 0);
        assert_eq!(WavePlan::capacity(&routes, usize::MAX, 1, |_, _| true), 0);
    }

    /// Headers, records and sections of a one-set wave: each group names
    /// its slot and page, tiles cover every route eight at a time, and each
    /// token lists its routes in expert order.
    #[test]
    fn a_wave_is_laid_out_for_its_launches() {
        let memory = blocks(3);
        let experts = [expert(&memory, 0), expert(&memory, 1), expert(&memory, 2)];
        // Experts 4, 9, 17: blocks 0 and 1 in import 1, block 2 in import 2.
        let mut routes = vec![route(4, 0, 0.5), route(4, 1, 0.25), route(4, 2, 1.0)];
        routes.push(route(9, 1, 2.0));
        routes.extend((0..10).map(|t| route(17, t, 0.125)));
        let tokens = 11;
        let plan = WavePlan::new(&routes, &experts, tokens, imports(&memory, 2)).unwrap();
        assert_eq!(
            WavePlan::capacity(&routes, tokens, 2, |sets, bytes| {
                sets == plan.sets() && bytes <= plan.bytes()
            }),
            experts.len()
        );
        assert_eq!(plan.sets(), 1);
        let [first, second, unused, ..] = plan.chunks(0);
        assert_eq!(first, Some((1, 2 * STRIDE as u64)));
        assert_eq!(second, Some((2, STRIDE as u64)));
        assert_eq!(unused, None, "an unused slot names no import");
        assert_eq!(plan.tiles(0), 1 + 1 + 2);
        let hidden: Vec<f32> = (0..tokens * MOE_HIDDEN).map(|i| i as f32).collect();
        let output: Vec<f32> = (0..tokens * MOE_HIDDEN).map(|i| -(i as f32)).collect();
        let mut words = vec![0xdead_beef; plan.bytes() as usize / 4];
        plan.fill(&mut words, &routes, &hidden, &output).unwrap();
        let header = &words[..HEADER_WORDS];
        assert_eq!(header[..6], [0, 4, 3, 14, 11, 1]);
        let at = |word: usize| header[word] as usize;
        assert_eq!(
            words[at(6)..at(6) + 12],
            [0, 0, 0, 3, 0, 1372, 3, 1, 1, 0, 4, 10],
            "groups: slot, page, first route, count"
        );
        assert_eq!(
            words[at(8)..at(8) + 8],
            [0, 0, 1, 3, 2, 4, 2, 12],
            "tiles: group, first route"
        );
        assert_eq!(
            words[at(7) + 6..at(7) + 8],
            [1, 2.0f32.to_bits()],
            "route 3 is token 1's expert 9"
        );
        // Token 1: expert 4's route 1, expert 9's route 3, expert 17's
        // route 5; token 9 only expert 17's route 13, and token 10 none.
        let token = |t: usize| {
            let (first, count) = (
                words[at(9) + 2 * t] as usize,
                words[at(9) + 2 * t + 1] as usize,
            );
            words[at(10) + first..at(10) + first + count].to_vec()
        };
        assert_eq!(token(1), [1, 3, 5]);
        assert_eq!(token(9), [13]);
        assert!(token(10).is_empty());
        assert_eq!(token(0), [0, 4]);
        for (section, values) in [(11, &hidden), (14, &output)] {
            assert_eq!(at(section) % HEADER_WORDS, 0);
            assert!(
                words[at(section)..at(section) + values.len()]
                    .iter()
                    .zip(values.iter())
                    .all(|(&word, value)| word == value.to_bits())
            );
        }
        assert_eq!(
            plan.out(),
            ((at(14) * 4) as u64, (tokens * MOE_HIDDEN * 4) as u64)
        );
        assert_eq!(at(12), at(11) + (tokens * MOE_HIDDEN).next_multiple_of(16));
        assert_eq!(
            at(13),
            at(12) + (14 * MOE_INTERMEDIATE).next_multiple_of(16)
        );
    }

    /// Seven experts in seven imports take two sets: each has its own
    /// header, tiles and slots, and offsets relative to its own base.
    #[test]
    fn more_imports_than_slots_take_another_set() {
        let memory = blocks(7);
        let experts: Vec<_> = (0..7).map(|b| expert(&memory, b)).collect();
        let routes: Vec<_> = (0..7).map(|e| route(e, e % 2, 1.0)).collect();
        let plan = WavePlan::new(&routes, &experts, 2, imports(&memory, 1)).unwrap();
        assert_eq!(plan.sets(), 2);
        assert_eq!(
            plan.chunks(1)[..3],
            [Some((6, STRIDE as u64)), Some((7, STRIDE as u64)), None]
        );
        assert_eq!((plan.tiles(0), plan.tiles(1)), (5, 2));
        let hidden = vec![0.0; 2 * MOE_HIDDEN];
        let mut words = vec![0; plan.bytes() as usize / 4];
        plan.fill(&mut words, &routes, &hidden, &hidden).unwrap();
        let (zero, one) = (&words[..16], &words[16..32]);
        assert_eq!((one[0], one[1]), (5, 2));
        for word in 6..15 {
            assert_eq!(zero[word], one[word] + HEADER_WORDS as u32);
        }
        assert_eq!(WavePlan::header_offset(1), 64);
        // Expert 6 is set 1, slot 1.
        let groups = zero[6] as usize;
        assert_eq!(words[groups + 4 * 6..groups + 4 * 6 + 2], [1, 0]);
    }

    #[test]
    fn a_wave_that_is_not_v3_blocks_in_resident_imports_is_refused() {
        let memory = blocks(2);
        let good = [expert(&memory, 0)];
        let routes = [route(3, 0, 1.0)];
        let plan = |experts: &[ExpertWeights<'_>], routes: &[MoeBatchRow], tokens| {
            WavePlan::new(routes, experts, tokens, imports(&memory, 2))
        };
        assert!(plan(&good, &routes, 1).is_ok());
        let mut shifted = good[0];
        shifted.down = &memory[SECTIONS.down_codes + 64..][..shifted.down.len()];
        assert_eq!(plan(&[shifted], &routes, 1), Err(WaveRefusal::Geometry));
        let mut unscaled = good[0];
        unscaled.up_scale = f32::NAN;
        assert_eq!(plan(&[unscaled], &routes, 1), Err(WaveRefusal::Geometry));
        // A token past the wave, tokens out of order, two groups for one
        // expert.
        assert_eq!(plan(&good, &routes, 0), Err(WaveRefusal::Geometry));
        assert_eq!(
            plan(&good, &[route(3, 1, 1.0)], 1),
            Err(WaveRefusal::Geometry)
        );
        assert_eq!(
            plan(&good, &[route(3, 1, 1.0), route(3, 0, 1.0)], 2),
            Err(WaveRefusal::Geometry)
        );
        assert_eq!(
            plan(&good, &[route(3, 0, 1.0), route(4, 1, 1.0)], 2),
            Err(WaveRefusal::Geometry)
        );
        // A block no import holds, one at an unaligned offset, one running
        // past its import.
        let outside = |_| None;
        assert_eq!(
            WavePlan::new(&routes, &good, 1, outside),
            Err(WaveRefusal::NotResident)
        );
        let unaligned = |_| Some((1, 8, 2 * STRIDE as u64));
        assert_eq!(
            WavePlan::new(&routes, &good, 1, unaligned),
            Err(WaveRefusal::NotResident)
        );
        let short = |_| Some((1, 0, SECTIONS.used as u64 - 1));
        assert_eq!(
            WavePlan::new(&routes, &good, 1, short),
            Err(WaveRefusal::NotResident)
        );
        // An import no one surface can bind whole.
        let huge = |_| Some((1, 0, MAX_CHUNK_BYTES + 4096));
        assert_eq!(
            WavePlan::new(&routes, &good, 1, huge),
            Err(WaveRefusal::NotResident)
        );
    }
}

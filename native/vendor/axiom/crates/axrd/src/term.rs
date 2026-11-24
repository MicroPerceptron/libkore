//! Terminal character grid — a fixed-size text buffer with cursor,
//! line editing, and scrollback.
//!
//! Pure data structure, `no_std`-safe. The rendering side (font atlas,
//! glyph placement) is separate — this module only tracks characters
//! and cursor state.
//!
//! Typical flow:
//! ```ignore
//! let mut term = TermGrid::<80, 24>::new();
//! term.put_char('H');
//! term.put_char('i');
//! term.newline();
//! // term.cell(0, 0) == 'H', term.cell(1, 0) == 'i'
//! // cursor is now at (0, 1)
//! ```

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// Maximum cells in a terminal grid (cols × rows).
/// 80×50 = 4000, which covers common terminal sizes.
const MAX_CELLS: usize = 80 * 50;

/// Snapshot of terminal cells copied out of a [`TermGrid`].
pub struct TermCells {
    bytes: [u8; MAX_CELLS],
    len: usize,
}

impl core::ops::Deref for TermCells {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes[..self.len]
    }
}

/// A fixed-size terminal character grid.
///
/// Characters are stored as `u8` (ASCII only — 0x20..=0x7E printable,
/// 0x00 = empty cell). The grid scrolls up when the cursor moves past
/// the last row.
pub struct TermGrid {
    /// Character cells, row-major: `cells[row * cols + col]`.
    cells: [AtomicU8; MAX_CELLS],
    cols: usize,
    rows: usize,
    /// Cursor column (0-based).
    pub cx: usize,
    /// Cursor row (0-based).
    pub cy: usize,
    /// Append-only cursor for lock-free multi-producer writes.
    cursor: AtomicUsize,
    /// Whether the grid content changed since last `clear_dirty()`.
    dirty: AtomicBool,
}

impl TermGrid {
    /// Create a new empty grid with the given dimensions.
    ///
    /// Panics if `cols * rows > MAX_CELLS` (4000).
    pub fn new(cols: usize, rows: usize) -> Self {
        assert!(cols * rows <= MAX_CELLS, "terminal too large");
        TermGrid {
            cells: [const { AtomicU8::new(0) }; MAX_CELLS],
            cols,
            rows,
            cx: 0,
            cy: 0,
            cursor: AtomicUsize::new(0),
            dirty: AtomicBool::new(true),
        }
    }

    /// Read a cell. Returns 0 for empty cells.
    #[inline]
    pub fn cell(&self, col: usize, row: usize) -> u8 {
        if col < self.cols && row < self.rows {
            self.cells[row * self.cols + col].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Write a printable ASCII character at the cursor and advance.
    pub fn put_char(&mut self, ch: u8) {
        if !(0x20..=0x7E).contains(&ch) {
            return; // ignore non-printable
        }
        if self.cx >= self.cols {
            self.wrap();
        }
        self.cells[self.cy * self.cols + self.cx].store(ch, Ordering::Relaxed);
        self.cx += 1;
        self.sync_append_cursor();
        self.dirty.store(true, Ordering::Release);
    }

    /// Lock-free append-only character write.
    ///
    /// Each call claims the next linear cell position with `fetch_add`,
    /// making this suitable for multi-producer append patterns. This
    /// path does **not** perform scrolling or line editing; once the
    /// grid is full, new writes are dropped and `false` is returned.
    ///
    /// Mixing this with the cursor-editing APIs (`newline`,
    /// `backspace`, `insert_char`, etc.) requires external
    /// synchronization.
    pub fn put_char_atomic(&self, ch: u8) -> bool {
        if !(0x20..=0x7E).contains(&ch) {
            return false;
        }

        let slot = self.cursor.fetch_add(1, Ordering::Relaxed);
        let total = self.cell_count();
        if slot >= total {
            return false;
        }

        self.cells[slot].store(ch, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Release);
        true
    }

    /// Lock-free append-only string write.
    ///
    /// Returns the number of printable characters that were written.
    pub fn put_str_atomic(&self, s: &[u8]) -> usize {
        let mut written = 0;
        for &ch in s {
            if self.put_char_atomic(ch) {
                written += 1;
            }
        }
        written
    }

    /// Current append cursor as a linear cell index, clamped to capacity.
    #[inline]
    pub fn atomic_cursor(&self) -> usize {
        self.cursor.load(Ordering::Relaxed).min(self.cell_count())
    }

    /// Write a string of ASCII bytes.
    pub fn put_str(&mut self, s: &[u8]) {
        for &ch in s {
            if ch == b'\n' {
                self.newline();
            } else {
                self.put_char(ch);
            }
        }
    }

    /// Move cursor to the start of the next line.
    pub fn newline(&mut self) {
        self.cx = 0;
        self.cy += 1;
        if self.cy >= self.rows {
            self.scroll_up();
            self.cy = self.rows - 1;
        }
        self.sync_append_cursor();
        self.dirty.store(true, Ordering::Release);
    }

    /// Delete the character behind the cursor (backspace).
    pub fn backspace(&mut self) {
        if self.cx > 0 {
            self.cx -= 1;
            self.cells[self.cy * self.cols + self.cx].store(0, Ordering::Relaxed);
            self.sync_append_cursor();
            self.dirty.store(true, Ordering::Release);
        }
    }

    /// Move cursor one column left (clamped to column 0).
    pub fn cursor_left(&mut self) {
        if self.cx > 0 {
            self.cx -= 1;
            self.sync_append_cursor();
        }
    }

    /// Move cursor one column right (clamped to `cols - 1`).
    pub fn cursor_right(&mut self) {
        if self.cx + 1 < self.cols {
            self.cx += 1;
            self.sync_append_cursor();
        }
    }

    /// Delete the character under the cursor, shifting the rest of the
    /// row left by one. Used for mid-line backspace.
    pub fn delete_at_cursor(&mut self) {
        let row_start = self.cy * self.cols;
        let pos = row_start + self.cx;
        let row_end = row_start + self.cols;
        // Shift cells left within this row.
        for i in pos..row_end - 1 {
            let next = self.cells[i + 1].load(Ordering::Relaxed);
            self.cells[i].store(next, Ordering::Relaxed);
        }
        // Clear the last cell in the row.
        self.cells[row_end - 1].store(0, Ordering::Relaxed);
        self.sync_append_cursor();
        self.dirty.store(true, Ordering::Release);
    }

    /// Insert a character at the cursor, shifting the rest of the row
    /// right by one. The last cell in the row is lost if full.
    pub fn insert_char(&mut self, ch: u8) {
        if !(0x20..=0x7E).contains(&ch) {
            return;
        }
        if self.cx >= self.cols {
            self.wrap();
        }
        let row_start = self.cy * self.cols;
        let pos = row_start + self.cx;
        let row_end = row_start + self.cols;
        // Shift cells right within this row.
        for i in (pos + 1..row_end).rev() {
            let prev = self.cells[i - 1].load(Ordering::Relaxed);
            self.cells[i].store(prev, Ordering::Relaxed);
        }
        self.cells[pos].store(ch, Ordering::Relaxed);
        self.cx += 1;
        self.sync_append_cursor();
        self.dirty.store(true, Ordering::Release);
    }

    /// Clear the entire grid and reset cursor.
    pub fn clear(&mut self) {
        for cell in &self.cells[..self.cols * self.rows] {
            cell.store(0, Ordering::Relaxed);
        }
        self.cx = 0;
        self.cy = 0;
        self.cursor.store(0, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Release);
    }

    /// Whether the grid has changed since last `clear_dirty()`.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Clear the dirty flag.
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty.store(false, Ordering::Release);
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The active cells slice (row-major, `cols * rows` elements).
    #[inline]
    pub fn cells(&self) -> TermCells {
        let len = self.cols * self.rows;
        let mut bytes = [0u8; MAX_CELLS];
        for (dst, src) in bytes[..len].iter_mut().zip(&self.cells[..len]) {
            *dst = src.load(Ordering::Relaxed);
        }
        TermCells { bytes, len }
    }

    // ── Internal ──────────────────────────────────────────────────

    /// Wrap cursor to the next line.
    fn wrap(&mut self) {
        self.cx = 0;
        self.cy += 1;
        if self.cy >= self.rows {
            self.scroll_up();
            self.cy = self.rows - 1;
        }
        self.sync_append_cursor();
    }

    /// Scroll the grid up by one line, clearing the bottom row.
    fn scroll_up(&mut self) {
        let cols = self.cols;
        let len = cols * self.rows;
        for i in 0..len - cols {
            let next = self.cells[i + cols].load(Ordering::Relaxed);
            self.cells[i].store(next, Ordering::Relaxed);
        }
        let last_row_start = (self.rows - 1) * cols;
        for cell in &self.cells[last_row_start..last_row_start + cols] {
            cell.store(0, Ordering::Relaxed);
        }
    }

    #[inline]
    fn cell_count(&self) -> usize {
        self.cols * self.rows
    }

    #[inline]
    fn sync_append_cursor(&self) {
        self.cursor
            .store(self.cy * self.cols + self.cx, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_typing() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"Hello");
        assert_eq!(t.cell(0, 0), b'H');
        assert_eq!(t.cell(4, 0), b'o');
        assert_eq!(t.cx, 5);
        assert_eq!(t.cy, 0);
    }

    #[test]
    fn newline_advances_row() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"Hi\nWorld");
        assert_eq!(t.cell(0, 0), b'H');
        assert_eq!(t.cell(1, 0), b'i');
        assert_eq!(t.cell(0, 1), b'W');
        assert_eq!(t.cy, 1);
        assert_eq!(t.cx, 5);
    }

    #[test]
    fn backspace_erases() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"abc");
        t.backspace();
        assert_eq!(t.cx, 2);
        assert_eq!(t.cell(2, 0), 0);
        assert_eq!(t.cell(1, 0), b'b');
    }

    #[test]
    fn backspace_at_start_does_nothing() {
        let mut t = TermGrid::new(80, 24);
        t.backspace();
        assert_eq!(t.cx, 0);
        assert_eq!(t.cy, 0);
    }

    #[test]
    fn line_wrap() {
        let mut t = TermGrid::new(4, 3);
        t.put_str(b"ABCDE"); // wraps after 4 chars
        assert_eq!(t.cell(0, 0), b'A');
        assert_eq!(t.cell(3, 0), b'D');
        assert_eq!(t.cell(0, 1), b'E');
        assert_eq!(t.cy, 1);
        assert_eq!(t.cx, 1);
    }

    #[test]
    fn scroll_up() {
        let mut t = TermGrid::new(4, 2);
        t.put_str(b"AA\nBB\nCC");
        // "AA" was on row 0, "BB" on row 1, "CC" triggers scroll.
        // After scroll: row 0 = "BB", row 1 = "CC".
        assert_eq!(t.cell(0, 0), b'B');
        assert_eq!(t.cell(1, 0), b'B');
        assert_eq!(t.cell(0, 1), b'C');
        assert_eq!(t.cell(1, 1), b'C');
    }

    #[test]
    fn clear_resets() {
        let mut t = TermGrid::new(4, 2);
        t.put_str(b"ABCD");
        t.clear();
        assert_eq!(t.cx, 0);
        assert_eq!(t.cy, 0);
        assert_eq!(t.cell(0, 0), 0);
    }

    #[test]
    fn non_printable_ignored() {
        let mut t = TermGrid::new(80, 24);
        t.put_char(0x01); // control char
        t.put_char(0x7F); // DEL
        assert_eq!(t.cx, 0); // cursor didn't move
    }

    #[test]
    fn cursor_left_right() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"abc");
        assert_eq!(t.cx, 3);
        t.cursor_left();
        assert_eq!(t.cx, 2);
        t.cursor_left();
        t.cursor_left();
        assert_eq!(t.cx, 0);
        t.cursor_left(); // clamped
        assert_eq!(t.cx, 0);
        t.cursor_right();
        assert_eq!(t.cx, 1);
    }

    #[test]
    fn insert_char_mid_line() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"ac");
        // cursor at col 2, move left once to col 1 (between 'a' and 'c')
        t.cursor_left();
        t.insert_char(b'b');
        assert_eq!(t.cell(0, 0), b'a');
        assert_eq!(t.cell(1, 0), b'b');
        assert_eq!(t.cell(2, 0), b'c');
        assert_eq!(t.cx, 2);
    }

    #[test]
    fn delete_at_cursor() {
        let mut t = TermGrid::new(80, 24);
        t.put_str(b"abcd");
        // Move cursor to col 1 (over 'b'), delete it.
        t.cx = 1;
        t.delete_at_cursor();
        assert_eq!(t.cell(0, 0), b'a');
        assert_eq!(t.cell(1, 0), b'c');
        assert_eq!(t.cell(2, 0), b'd');
        assert_eq!(t.cell(3, 0), 0); // shifted in empty
    }

    #[test]
    fn dirty_tracking() {
        let mut t = TermGrid::new(80, 24);
        assert!(t.is_dirty()); // dirty on creation
        t.clear_dirty();
        assert!(!t.is_dirty());
        t.put_char(b'A');
        assert!(t.is_dirty());
    }

    #[test]
    fn atomic_append_claims_linear_slots() {
        let t = TermGrid::new(4, 2);
        assert!(t.put_char_atomic(b'A'));
        assert!(t.put_char_atomic(b'B'));
        assert!(t.put_char_atomic(b'C'));
        assert_eq!(t.cell(0, 0), b'A');
        assert_eq!(t.cell(1, 0), b'B');
        assert_eq!(t.cell(2, 0), b'C');
        assert_eq!(t.atomic_cursor(), 3);
    }

    #[test]
    fn atomic_append_stops_when_grid_is_full() {
        let t = TermGrid::new(2, 1);
        assert!(t.put_char_atomic(b'A'));
        assert!(t.put_char_atomic(b'B'));
        assert!(!t.put_char_atomic(b'C'));
        assert_eq!(t.cell(0, 0), b'A');
        assert_eq!(t.cell(1, 0), b'B');
    }
}

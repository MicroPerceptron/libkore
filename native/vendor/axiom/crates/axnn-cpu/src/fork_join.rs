//! Caller-lent parallelism.
//!
//! Kernels never create threads. A host that has workers lends them through
//! [`ForkJoin`], and a kernel that splits its work computes every output
//! element exactly as its serial form does, only on another thread: results
//! are bit-identical at any width, including [`Serial`].

use core::marker::PhantomData;

/// Runs independent tasks, possibly concurrently.
///
/// # Safety
///
/// `run(tasks, task)` must call `task(i)` exactly once for every `i` in
/// `0..tasks`, never for any other index, and return only after every call
/// has returned. Kernels rely on this to give task `i` exclusive access to
/// its own part of an output.
pub unsafe trait ForkJoin: Sync {
    /// How many tasks can make progress at once, the calling thread
    /// included. Kernels size their splits by it.
    fn width(&self) -> usize;

    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync));
}

/// Every task on the calling thread, in index order.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

// SAFETY: runs each index once, in order, before returning.
unsafe impl ForkJoin for Serial {
    fn width(&self) -> usize {
        1
    }

    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        for index in 0..tasks {
            task(index);
        }
    }
}

/// Multiply-accumulates a task should carry at least, so dispatch stays a
/// small fraction of its work.
const MIN_TASK_WORK: usize = 1 << 15;

/// Tasks per unit of width: more tasks than workers lets fast cores claim
/// the share of slow ones (performance and efficiency cores differ).
const TASKS_PER_WORKER: usize = 8;

/// A slice split into chunks of `chunk` elements that tasks claim by index.
struct Chunks<'a, T> {
    base: *mut T,
    len: usize,
    chunk: usize,
    _borrow: PhantomData<&'a mut [T]>,
}

// SAFETY: chunks are disjoint, and `get` hands each index out at most once
// per `ForkJoin::run`.
unsafe impl<T: Send> Sync for Chunks<'_, T> {}

impl<'a, T> Chunks<'a, T> {
    fn new(slice: &'a mut [T], chunk: usize) -> Self {
        Self {
            base: slice.as_mut_ptr(),
            len: slice.len(),
            chunk,
            _borrow: PhantomData,
        }
    }

    /// # Safety
    /// No other borrow of chunk `index` is live.
    unsafe fn get(&self, index: usize) -> &'a mut [T] {
        let start = index * self.chunk;
        let len = self.chunk.min(self.len - start);
        // SAFETY: `start < len` for every index a run hands out, chunks do
        // not overlap, and the caller guarantees exclusivity.
        unsafe { core::slice::from_raw_parts_mut(self.base.add(start), len) }
    }
}

/// How many tasks `rows` rows of `row_work` multiply-accumulates each
/// deserve, and the rows each takes; `None` keeps them on the caller.
fn split(parallel: &dyn ForkJoin, rows: usize, row_work: usize) -> Option<(usize, usize)> {
    let by_work = (rows * row_work) / MIN_TASK_WORK;
    let tasks = by_work
        .min(parallel.width() * TASKS_PER_WORKER)
        .min(rows)
        .max(1);
    if tasks == 1 {
        return None;
    }
    let rows_per_task = rows.div_ceil(tasks);
    Some((rows.div_ceil(rows_per_task), rows_per_task))
}

/// Split `rows` rows of `row_work` multiply-accumulates each into
/// contiguous ranges and run `body(first_row, rows_of_output)` on each,
/// where `output` holds `row_elements` elements per row.
///
/// Small outputs run in one call on the calling thread.
pub(crate) fn for_rows<T: Send>(
    parallel: &dyn ForkJoin,
    output: &mut [T],
    row_elements: usize,
    row_work: usize,
    body: impl Fn(usize, &mut [T]) + Sync,
) {
    let rows = output.len() / row_elements.max(1);
    let Some((tasks, rows_per_task)) = split(parallel, rows, row_work) else {
        body(0, output);
        return;
    };
    let chunks = Chunks::new(output, rows_per_task * row_elements);
    parallel.run(tasks, &|index| {
        // SAFETY: `ForkJoin` runs each index once, so each chunk has one
        // borrower.
        let chunk = unsafe { chunks.get(index) };
        body(index * rows_per_task, chunk);
    });
}

/// Split `rows` rows of `row_work` multiply-accumulates each into
/// contiguous ranges and run `body(first_row, rows)` on each. For outputs
/// no single slice holds, such as one column range across several rows of
/// a matrix: the caller keeps each range's writes to itself.
pub(crate) fn for_row_ranges(
    parallel: &dyn ForkJoin,
    rows: usize,
    row_work: usize,
    body: impl Fn(usize, usize) + Sync,
) {
    let Some((tasks, rows_per_task)) = split(parallel, rows, row_work) else {
        body(0, rows);
        return;
    };
    parallel.run(tasks, &|index| {
        let first = index * rows_per_task;
        body(first, rows_per_task.min(rows - first));
    });
}

/// [`for_rows`] over two outputs split at the same rows: `a` holds
/// `a_row` elements per row and `b` holds `b_row`, for as many rows.
pub(crate) fn for_row_pairs<A: Send, B: Send>(
    parallel: &dyn ForkJoin,
    (a, a_row): (&mut [A], usize),
    (b, b_row): (&mut [B], usize),
    row_work: usize,
    body: impl Fn(usize, &mut [A], &mut [B]) + Sync,
) {
    let rows = a.len() / a_row.max(1);
    debug_assert_eq!(rows, b.len() / b_row.max(1));
    let Some((tasks, rows_per_task)) = split(parallel, rows, row_work) else {
        body(0, a, b);
        return;
    };
    let (a, b) = (
        Chunks::new(a, rows_per_task * a_row),
        Chunks::new(b, rows_per_task * b_row),
    );
    parallel.run(tasks, &|index| {
        // SAFETY: `ForkJoin` runs each index once, so each pair of chunks
        // has one borrower.
        let (a, b) = unsafe { (a.get(index), b.get(index)) };
        body(index * rows_per_task, a, b);
    });
}

/// Runs tasks in reverse at a given width, as a stand-in for any other
/// order, for tests that check a split changes no result.
#[cfg(test)]
pub(crate) struct Reversed(pub usize);

// SAFETY: runs each index once, before returning.
#[cfg(test)]
unsafe impl ForkJoin for Reversed {
    fn width(&self) -> usize {
        self.0
    }
    fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        for index in (0..tasks).rev() {
            task(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn rows_are_covered_once_in_contiguous_chunks() {
        for (rows, width) in [(1000, 8), (7, 8), (1, 4), (1 << 12, 3)] {
            let mut output = vec![0_u32; rows * 2];
            let calls = AtomicUsize::new(0);
            for_rows(
                &Reversed(width),
                &mut output,
                2,
                MIN_TASK_WORK / 16,
                |first, chunk| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    for (offset, pair) in chunk.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        pair[0] += (first + offset) as u32;
                        pair[1] += 1;
                    }
                },
            );
            let expected: Vec<u32> = (0..rows).flat_map(|r| [r as u32, 1]).collect();
            assert_eq!(output, expected, "{rows} rows at width {width}");
            let calls = calls.into_inner();
            assert!(calls <= (width * TASKS_PER_WORKER).max(1));
            if rows * (MIN_TASK_WORK / 16) < MIN_TASK_WORK {
                assert_eq!(calls, 1, "small work stays on the caller");
            }
        }
    }
}

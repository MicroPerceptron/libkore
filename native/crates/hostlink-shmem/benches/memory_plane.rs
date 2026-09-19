use std::alloc::{Layout as AllocationLayout, alloc_zeroed, dealloc, handle_alloc_error};
use std::hint::black_box;
use std::time::{Duration, Instant};

use kore_hostlink_shmem::{ALIGN, Epoch, Layout, split};

struct AlignedWindow {
    pointer: *mut u8,
    allocation: AllocationLayout,
}

impl AlignedWindow {
    fn new(bytes: usize) -> Self {
        let allocation = AllocationLayout::from_size_align(bytes, ALIGN).unwrap();
        // SAFETY: valid nonzero layout; allocation is owned until Drop.
        let pointer = unsafe { alloc_zeroed(allocation) };
        if pointer.is_null() {
            handle_alloc_error(allocation);
        }
        Self {
            pointer,
            allocation,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: this object uniquely owns the complete allocation.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.allocation.size()) }
    }
}

impl Drop for AlignedWindow {
    fn drop(&mut self) {
        // SAFETY: pointer came from alloc_zeroed with this exact layout.
        unsafe { dealloc(self.pointer, self.allocation) }
    }
}

fn iterations(payload: usize) -> usize {
    // Move at least 64 MiB, but retain enough iterations to measure small
    // records. The shared-window case performs one write and one read copy.
    (64 * 1024 * 1024 / payload.max(1)).max(2_000)
}

fn rate(operations: usize, bytes: usize, elapsed: Duration) -> (f64, f64) {
    let ns = elapsed.as_secs_f64() * 1e9 / operations as f64;
    let mib = bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64();
    (ns, mib)
}

fn main() {
    println!("HostLink shared-memory core cost (not a device or serial benchmark)");
    println!("payload       direct-copy ns/op  slot roundtrip ns/op  roundtrip MiB/s");

    for payload in [64usize, 256, 4096, 65_536, 1_048_576] {
        let count = iterations(payload);
        let source = vec![0xA5u8; payload];
        let mut destination = vec![0u8; payload];

        let start = Instant::now();
        for _ in 0..count {
            destination.copy_from_slice(black_box(&source));
            black_box(&destination);
        }
        let direct = start.elapsed();

        let layout = Layout::new(2, payload as u32).unwrap();
        let mut window = AlignedWindow::new(layout.window_bytes());
        let epoch = Epoch::new(1).unwrap();
        let (mut a, mut b) = split(window.as_mut_slice(), layout, epoch).unwrap();
        let start = Instant::now();
        for _ in 0..count {
            a.tx().try_push(black_box(&source)).unwrap();
            let read = b.rx().try_pop(black_box(&mut destination)).unwrap();
            black_box(read);
        }
        let shared = start.elapsed();

        let (direct_ns, _) = rate(count, count * payload, direct);
        let (shared_ns, shared_mib) = rate(count, count * payload, shared);
        println!(
            "{:>7} B {:>20.1} {:>21.1} {:>18.1}",
            payload, direct_ns, shared_ns, shared_mib
        );
    }
}

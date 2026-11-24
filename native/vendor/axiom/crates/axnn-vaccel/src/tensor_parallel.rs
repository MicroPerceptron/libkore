//! Measured tensor-parallel work division.
//!
//! A service curve includes the setup cost that makes small accelerator shards unattractive.
//! The splitter assigns aligned output rows one quantum at a time to the device whose predicted
//! completion time would be smallest after accepting that quantum. This minimizes the observed
//! makespan without baking a CPU/GPU/NPU ratio into a model implementation.

use std::vec::Vec;

/// Linear service-time estimate for one device: `setup + units * per_unit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceCurve {
    pub setup_ns: u64,
    pub per_unit_ns: u64,
}

impl ServiceCurve {
    pub const fn new(setup_ns: u64, per_unit_ns: u64) -> Self {
        Self {
            setup_ns,
            per_unit_ns: if per_unit_ns == 0 { 1 } else { per_unit_ns },
        }
    }

    /// Fit a curve from two measured shard sizes. Noise that would imply a negative setup or
    /// slope is clamped to a conservative, monotonic estimate.
    pub fn fit(first: (usize, u64), second: (usize, u64)) -> Option<Self> {
        let ((small_units, small_ns), (large_units, large_ns)) = if first.0 <= second.0 {
            (first, second)
        } else {
            (second, first)
        };
        if small_units == 0 || small_units == large_units {
            return None;
        }
        let unit_delta = (large_units - small_units) as u64;
        let slope = large_ns
            .saturating_sub(small_ns)
            .checked_div(unit_delta)
            .unwrap_or(0)
            .max(1);
        let setup = small_ns.saturating_sub(slope.saturating_mul(small_units as u64));
        Some(Self::new(setup, slope))
    }

    pub fn predict_ns(self, units: usize) -> u64 {
        if units == 0 {
            0
        } else {
            self.setup_ns
                .saturating_add(self.per_unit_ns.saturating_mul(units as u64))
        }
    }

    /// Fold a completed shard into this curve. Setup remains independently measured; the row
    /// rate uses a stable 7/8 EWMA so one thermal or scheduling outlier cannot swing a split.
    pub fn observe(&mut self, units: usize, elapsed_ns: u64) {
        if units == 0 {
            return;
        }
        let sample = elapsed_ns
            .saturating_sub(self.setup_ns)
            .div_ceil(units as u64)
            .max(1);
        self.per_unit_ns = self
            .per_unit_ns
            .saturating_mul(7)
            .saturating_add(sample)
            .div_ceil(8);
    }
}

/// One contiguous output-row interval assigned to a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceShard {
    pub device: usize,
    pub first: usize,
    pub rows: usize,
    pub predicted_ns: u64,
}

/// Allocate `rows` among measured devices. Missing entries are unavailable. All non-final
/// assignments are multiples of `alignment`; returned intervals cover the output exactly once.
pub fn measured_split(
    rows: usize,
    alignment: usize,
    devices: &[Option<ServiceCurve>],
) -> Vec<DeviceShard> {
    if rows == 0 || devices.iter().all(Option::is_none) {
        return Vec::new();
    }
    let quantum = alignment.max(1);
    let mut assigned = vec![0usize; devices.len()];
    let mut remaining = rows;
    while remaining != 0 {
        let chunk = quantum.min(remaining);
        let device = devices
            .iter()
            .enumerate()
            .filter_map(|(device, curve)| curve.map(|curve| (device, curve)))
            .min_by_key(|(device, curve)| (curve.predict_ns(assigned[*device] + chunk), *device))
            .map(|(device, _)| device)
            .expect("at least one measured device");
        assigned[device] += chunk;
        remaining -= chunk;
    }
    let mut first = 0;
    assigned
        .into_iter()
        .enumerate()
        .filter_map(|(device, rows)| {
            if rows == 0 {
                return None;
            }
            let shard = DeviceShard {
                device,
                first,
                rows,
                predicted_ns: devices[device].unwrap().predict_ns(rows),
            };
            first += rows;
            Some(shard)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_preserves_setup_and_row_rate() {
        assert_eq!(
            ServiceCurve::fit((64, 228), (128, 356)),
            Some(ServiceCurve::new(100, 2))
        );
    }

    #[test]
    fn equalizes_predicted_completion_instead_of_rows() {
        let split = measured_split(
            160,
            8,
            &[
                Some(ServiceCurve::new(0, 4)),
                Some(ServiceCurve::new(80, 1)),
            ],
        );
        assert_eq!(split.iter().map(|s| s.rows).sum::<usize>(), 160);
        assert_eq!(split[0].first, 0);
        assert_eq!(split[1].first, split[0].rows);
        assert!(split[1].rows > split[0].rows);
        let spread = split.iter().map(|s| s.predicted_ns).max().unwrap()
            - split.iter().map(|s| s.predicted_ns).min().unwrap();
        assert!(spread <= 8 * 4);
    }

    #[test]
    fn unavailable_devices_receive_nothing_and_tail_is_not_lost() {
        let split = measured_split(35, 16, &[None, Some(ServiceCurve::new(7, 3)), None]);
        assert_eq!(
            split,
            vec![DeviceShard {
                device: 1,
                first: 0,
                rows: 35,
                predicted_ns: 112,
            }]
        );
    }

    #[test]
    fn observations_move_the_next_split_gradually() {
        let mut curve = ServiceCurve::new(100, 10);
        curve.observe(10, 300);
        assert_eq!(curve.per_unit_ns, 12);
    }
}

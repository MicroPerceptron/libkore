//! SCL count derivation — the **fallback** when firmware declared none.
//!
//! Firmware's own counts (`SSCN`/`FMCN` in ACPI) always win. They account
//! for the board's trace capacitance and rise time, which no formula can
//! know; this module exists for the machine that declares nothing, and is
//! documented as second-best on purpose.
//!
//! The formulas are Synopsys's, as Linux implements them in
//! `drivers/i2c/busses/i2c-designware-common.c`, with the I2C-bus-spec
//! symbol times. They are transcribed, not re-derived — a "simplified"
//! version of a timing formula is a bus that mostly works.

/// I2C-spec standard-mode `tHD;STA` = `tHIGH` = 4.0 µs, in ns.
const SS_T_HIGH_NS: u32 = 4_000;
/// I2C-spec standard-mode `tLOW` = 4.7 µs, in ns.
const SS_T_LOW_NS: u32 = 4_700;
/// I2C-spec fast-mode `tHD;STA` = `tHIGH` = 0.6 µs, in ns.
const FS_T_HIGH_NS: u32 = 600;
/// I2C-spec fast-mode `tLOW` = 1.3 µs, in ns.
const FS_T_LOW_NS: u32 = 1_300;
/// Default SDA/SCL fall time when firmware declares none, in ns — the
/// value Linux assumes for the same reason.
const DEFAULT_FALL_NS: u32 = 300;

/// Round `a / b` to nearest, away from zero on a tie.
const fn div_round_closest(a: u64, b: u64) -> u64 {
    (a + b / 2) / b
}

/// SCL high count for a symbol time.
///
/// The `- 3` is the DesignWare core's own pipeline offset (the databook's
/// "DW default"); the ideal-generator variant subtracts 8 and is not used
/// here because no in-tree integration is an ideal generator.
const fn scl_hcnt(ic_clk_khz: u32, t_symbol_ns: u32, t_fall_ns: u32) -> u32 {
    let ticks = div_round_closest(
        ic_clk_khz as u64 * (t_symbol_ns + t_fall_ns) as u64,
        1_000_000,
    );
    // Saturating: a clock slow enough to underflow the offset would wrap
    // to an enormous count and silently produce a bus running at a
    // fraction of the requested speed.
    (ticks.saturating_sub(3)) as u32
}

/// SCL low count for a symbol time.
const fn scl_lcnt(ic_clk_khz: u32, t_low_ns: u32, t_fall_ns: u32) -> u32 {
    let ticks = div_round_closest(ic_clk_khz as u64 * (t_low_ns + t_fall_ns) as u64, 1_000_000);
    (ticks.saturating_sub(1)) as u32
}

/// Clamp a derived count into the register's 16-bit field.
///
/// A count of zero would make the core generate no clock at all, so the
/// floor is 1 rather than 0 — a stopped bus is worse than a slow one, and
/// both are better than a wrapped register.
const fn clamp_count(n: u32) -> u16 {
    if n == 0 {
        1
    } else if n > u16::MAX as u32 {
        u16::MAX
    } else {
        n as u16
    }
}

/// Derive `(hcnt, lcnt)` for a speed from the controller's input clock.
pub const fn derive(speed: kore_i2c::Speed, input_clock_hz: u32) -> kore_i2c::Counts {
    let khz = input_clock_hz / 1_000;
    let (t_high, t_low) = match speed {
        kore_i2c::Speed::Standard => (SS_T_HIGH_NS, SS_T_LOW_NS),
        kore_i2c::Speed::Fast => (FS_T_HIGH_NS, FS_T_LOW_NS),
    };
    kore_i2c::Counts {
        hcnt: clamp_count(scl_hcnt(khz, t_high, DEFAULT_FALL_NS)),
        lcnt: clamp_count(scl_lcnt(khz, t_low, DEFAULT_FALL_NS)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_i2c::Speed;

    #[test]
    fn derived_counts_match_the_synopsys_formula_at_100mhz() {
        // 100 MHz = 100_000 kHz.
        // fast hcnt: round(100000 * 900 / 1e6) - 3 = 90 - 3 = 87
        // fast lcnt: round(100000 * 1600 / 1e6) - 1 = 160 - 1 = 159
        let fs = derive(Speed::Fast, 100_000_000);
        assert_eq!((fs.hcnt, fs.lcnt), (87, 159));
        // std hcnt: round(100000 * 4300 / 1e6) - 3 = 430 - 3 = 427
        // std lcnt: round(100000 * 5000 / 1e6) - 1 = 500 - 1 = 499
        let ss = derive(Speed::Standard, 100_000_000);
        assert_eq!((ss.hcnt, ss.lcnt), (427, 499));
    }

    #[test]
    fn a_different_integration_clock_gives_different_counts() {
        // The whole reason the clock is per-integration data: AMD's core
        // is clocked differently from Intel's, and a hardcoded constant
        // would mis-time one of them while looking correct.
        let intel = derive(Speed::Fast, 100_000_000);
        let amd = derive(Speed::Fast, 133_000_000);
        assert_ne!(intel, amd);
        // 133000 kHz: hcnt round(133000*900/1e6)=120 -> 117
        //             lcnt round(133000*1600/1e6)=213 -> 212
        assert_eq!((amd.hcnt, amd.lcnt), (117, 212));
    }

    #[test]
    fn the_low_period_always_exceeds_the_high_period() {
        // A structural property of the I2C spec's timings that must
        // survive any clock: tLOW > tHIGH in both modes.
        for hz in [
            25_000_000,
            100_000_000,
            120_000_000,
            133_000_000,
            216_000_000,
        ] {
            for speed in [Speed::Standard, Speed::Fast] {
                let c = derive(speed, hz);
                assert!(c.lcnt > c.hcnt, "{speed:?} at {hz} Hz: {c:?}");
            }
        }
    }

    #[test]
    fn a_pathologically_slow_clock_clamps_instead_of_wrapping() {
        // 1 kHz input: every term underflows the offset. The counts must
        // floor at 1, never wrap to 0xFFFF_FFFD.
        let c = derive(Speed::Fast, 1_000);
        assert_eq!((c.hcnt, c.lcnt), (1, 1));
    }

    #[test]
    fn the_register_field_clamp_saturates_rather_than_wrapping() {
        // The clamp's upper arm is not reachable from a `u32` input
        // clock — the fastest expressible clock still divides down to
        // ~18k counts — so it is exercised directly rather than through
        // a `derive` call that would assert a false property.
        assert_eq!(clamp_count(u16::MAX as u32 + 1), u16::MAX);
        assert_eq!(clamp_count(u32::MAX), u16::MAX);
        assert_eq!(clamp_count(0), 1, "a zero count would stop the clock");
        assert_eq!(clamp_count(1), 1);
        assert_eq!(clamp_count(u16::MAX as u32), u16::MAX);

        // What the fastest expressible clock actually produces: in range,
        // and still ordered correctly.
        let c = derive(Speed::Standard, u32::MAX);
        assert_eq!((c.hcnt, c.lcnt), (18465, 21474));
        assert!(c.lcnt > c.hcnt);
    }
}

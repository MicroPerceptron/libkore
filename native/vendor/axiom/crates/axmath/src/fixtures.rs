//! Test-only helpers for validating approximation accuracy over ranges.

#[derive(Clone, Copy, Debug, Default)]
struct RangeStatsF32 {
    samples: usize,
    max_abs_err: f32,
    max_abs_x: f32,
    max_rel_err: f32,
    max_rel_x: f32,
    first_failure: Option<FailureF32>,
}

#[derive(Clone, Copy, Debug)]
struct FailureF32 {
    x: f32,
    got: f32,
    want: f32,
    abs_err: f32,
    rel_err: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RangeStatsF64 {
    samples: usize,
    max_abs_err: f64,
    max_abs_x: f64,
    max_rel_err: f64,
    max_rel_x: f64,
    first_failure: Option<FailureF64>,
}

#[derive(Clone, Copy, Debug)]
struct FailureF64 {
    x: f64,
    got: f64,
    want: f64,
    abs_err: f64,
    rel_err: f64,
}

fn scan_unary_f32_range(
    approx: fn(f32) -> f32,
    reference: fn(f32) -> f32,
    start: f32,
    end: f32,
    steps: usize,
    abs_tol: f32,
    rel_tol: f32,
) -> RangeStatsF32 {
    let mut stats = RangeStatsF32::default();

    for i in 0..=steps {
        let t = (i as f32) / (steps as f32);
        let x = start + (end - start) * t;
        let got = approx(x);
        let want = reference(x);
        stats.samples += 1;

        if want.is_nan() {
            if !got.is_nan() && stats.first_failure.is_none() {
                stats.first_failure = Some(FailureF32 {
                    x,
                    got,
                    want,
                    abs_err: f32::NAN,
                    rel_err: f32::NAN,
                });
            }
            continue;
        }

        if want.is_infinite() {
            if got != want && stats.first_failure.is_none() {
                stats.first_failure = Some(FailureF32 {
                    x,
                    got,
                    want,
                    abs_err: f32::INFINITY,
                    rel_err: f32::INFINITY,
                });
            }
            continue;
        }

        let abs_err = (got - want).abs();
        let rel_err = abs_err / want.abs().max(1.0);

        if abs_err > stats.max_abs_err {
            stats.max_abs_err = abs_err;
            stats.max_abs_x = x;
        }
        if rel_err > stats.max_rel_err {
            stats.max_rel_err = rel_err;
            stats.max_rel_x = x;
        }

        if abs_err > abs_tol && rel_err > rel_tol && stats.first_failure.is_none() {
            stats.first_failure = Some(FailureF32 {
                x,
                got,
                want,
                abs_err,
                rel_err,
            });
        }
    }

    stats
}

fn scan_unary_f64_range(
    approx: fn(f64) -> f64,
    reference: fn(f64) -> f64,
    start: f64,
    end: f64,
    steps: usize,
    abs_tol: f64,
    rel_tol: f64,
) -> RangeStatsF64 {
    let mut stats = RangeStatsF64::default();

    for i in 0..=steps {
        let t = (i as f64) / (steps as f64);
        let x = start + (end - start) * t;
        let got = approx(x);
        let want = reference(x);
        stats.samples += 1;

        if want.is_nan() {
            if !got.is_nan() && stats.first_failure.is_none() {
                stats.first_failure = Some(FailureF64 {
                    x,
                    got,
                    want,
                    abs_err: f64::NAN,
                    rel_err: f64::NAN,
                });
            }
            continue;
        }

        if want.is_infinite() {
            if got != want && stats.first_failure.is_none() {
                stats.first_failure = Some(FailureF64 {
                    x,
                    got,
                    want,
                    abs_err: f64::INFINITY,
                    rel_err: f64::INFINITY,
                });
            }
            continue;
        }

        let abs_err = (got - want).abs();
        let rel_err = abs_err / want.abs().max(1.0);

        if abs_err > stats.max_abs_err {
            stats.max_abs_err = abs_err;
            stats.max_abs_x = x;
        }
        if rel_err > stats.max_rel_err {
            stats.max_rel_err = rel_err;
            stats.max_rel_x = x;
        }

        if abs_err > abs_tol && rel_err > rel_tol && stats.first_failure.is_none() {
            stats.first_failure = Some(FailureF64 {
                x,
                got,
                want,
                abs_err,
                rel_err,
            });
        }
    }

    stats
}

#[track_caller]
pub(crate) fn assert_unary_f32_range(
    label: &str,
    approx: fn(f32) -> f32,
    reference: fn(f32) -> f32,
    start: f32,
    end: f32,
    steps: usize,
    abs_tol: f32,
    rel_tol: f32,
) {
    let stats = scan_unary_f32_range(approx, reference, start, end, steps, abs_tol, rel_tol);

    if let Some(failure) = stats.first_failure {
        panic!(
            "{label}: first failure at x={x} got={got:e} want={want:e} abs_err={abs_err:e} rel_err={rel_err:e}; max_abs_err={max_abs_err:e} at x={max_abs_x}; max_rel_err={max_rel_err:e} at x={max_rel_x}; samples={samples}",
            x = failure.x,
            got = failure.got,
            want = failure.want,
            abs_err = failure.abs_err,
            rel_err = failure.rel_err,
            max_abs_err = stats.max_abs_err,
            max_abs_x = stats.max_abs_x,
            max_rel_err = stats.max_rel_err,
            max_rel_x = stats.max_rel_x,
            samples = stats.samples,
        );
    }
}

#[track_caller]
pub(crate) fn assert_unary_f64_range(
    label: &str,
    approx: fn(f64) -> f64,
    reference: fn(f64) -> f64,
    start: f64,
    end: f64,
    steps: usize,
    abs_tol: f64,
    rel_tol: f64,
) {
    let stats = scan_unary_f64_range(approx, reference, start, end, steps, abs_tol, rel_tol);

    if let Some(failure) = stats.first_failure {
        panic!(
            "{label}: first failure at x={x} got={got:e} want={want:e} abs_err={abs_err:e} rel_err={rel_err:e}; max_abs_err={max_abs_err:e} at x={max_abs_x}; max_rel_err={max_rel_err:e} at x={max_rel_x}; samples={samples}",
            x = failure.x,
            got = failure.got,
            want = failure.want,
            abs_err = failure.abs_err,
            rel_err = failure.rel_err,
            max_abs_err = stats.max_abs_err,
            max_abs_x = stats.max_abs_x,
            max_rel_err = stats.max_rel_err,
            max_rel_x = stats.max_rel_x,
            samples = stats.samples,
        );
    }
}

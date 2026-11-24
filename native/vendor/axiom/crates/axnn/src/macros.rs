/// Create a tensor on a backend.
///
/// # Forms
///
/// **Zeros with shape and dtype:**
/// ```ignore
/// let x = tensor!(&be, F32, shape![2, 3])?;
/// let w = tensor!(&be, Q4, shape![4096])?;
/// ```
///
/// **1D f32 tensor from values:**
/// ```ignore
/// let x = tensor!(&be, [1.0, 2.0, 3.0])?;
/// ```
///
/// **Upload raw bytes with dtype and shape:**
/// ```ignore
/// let x = tensor!(&be, Q4, shape![256], &raw_bytes)?;
/// ```
///
/// The dtype argument accepts any `DType` value. Constants like
/// `F32`, `Q4`, etc. are re-exported from `axnn` for convenience.
#[macro_export]
macro_rules! tensor {
    // ── 1D from f32 literal values: tensor!(be, [v0, v1, ...]) ──
    // Always produces an F32 tensor. For other dtypes, use the
    // shape + bytes form.
    ($be:expr, [$($v:expr),+ $(,)?]) => {{
        const N: usize = $crate::_count!($($v),+);
        let values: [f32; N] = [$($v as f32),+];
        let mut bytes = [0u8; N * 4];
        let mut i = 0;
        while i < N {
            let b = values[i].to_le_bytes();
            bytes[i * 4] = b[0];
            bytes[i * 4 + 1] = b[1];
            bytes[i * 4 + 2] = b[2];
            bytes[i * 4 + 3] = b[3];
            i += 1;
        }
        let desc = $crate::TensorDesc::new($crate::shape![N], $crate::DType::F32);
        $crate::Backend::upload($be, &bytes, &desc)
    }};

    // ── Upload raw bytes: tensor!(be, DTYPE, shape, &bytes) ──
    ($be:expr, $dt:expr, $shape:expr, $bytes:expr) => {{
        let desc = $crate::TensorDesc::new($shape, $dt);
        $crate::Backend::upload($be, $bytes, &desc)
    }};

    // ── Zeros: tensor!(be, DTYPE, shape) ──
    ($be:expr, $dt:expr, $shape:expr) => {{
        let desc = $crate::TensorDesc::new($shape, $dt);
        $crate::Backend::zeros($be, &desc)
    }};
}

/// Count the number of expressions in a macro repetition.
#[doc(hidden)]
#[macro_export]
macro_rules! _count {
    () => { 0usize };
    ($head:expr $(, $tail:expr)*) => { 1usize + $crate::_count!($($tail),*) };
}

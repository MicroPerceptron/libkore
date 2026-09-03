//! Unsigned integer traits.

pub trait IntoUsize: Copy + Ord + Sized {
    fn into_usize(self) -> usize;
}

pub trait FromUsize: Copy + Ord + Sized {
    fn from_usize(value: usize) -> Self;
}

pub trait Unsigned: Copy + Ord + Sized + IntoUsize + FromUsize {
    const MAX: usize;

    fn align_up(self, alignment: usize) -> Option<Self> {
        if alignment == 0 {
            return Some(self);
        }
        let mask = alignment - 1;
        let value = self.into_usize();
        let aligned = value.checked_add(mask)?;
        let aligned = aligned & !mask;
        if aligned > Self::MAX {
            return None;
        }
        Some(Self::from_usize(aligned))
    }

    fn align_down(self, alignment: usize) -> Option<Self> {
        if alignment == 0 {
            return Some(self);
        }
        let mask = alignment - 1;
        let value = self.into_usize();
        let aligned = value.checked_sub(value & mask)?;
        if aligned > Self::MAX {
            return None;
        }
        Some(Self::from_usize(aligned))
    }

    fn checked_add(self, offset: usize) -> Option<Self>;
    fn checked_sub(self, offset: usize) -> Option<Self>;
    fn checked_mul(self, offset: usize) -> Option<Self>;
    fn checked_div(self, offset: usize) -> Option<Self>;
}

macro_rules! impl_unsigned {
    ($t:ty) => {
        impl IntoUsize for $t {
            fn into_usize(self) -> usize {
                self as usize
            }
        }

        impl FromUsize for $t {
            fn from_usize(value: usize) -> Self {
                assert!(value <= <$t>::MAX as usize);
                value as $t
            }
        }

        impl Unsigned for $t {
            const MAX: usize = <$t>::MAX as usize;

            fn checked_add(self, offset: usize) -> Option<Self> {
                offset.try_into().ok().and_then(|o| self.checked_add(o))
            }

            fn checked_sub(self, offset: usize) -> Option<Self> {
                offset.try_into().ok().and_then(|o| self.checked_sub(o))
            }

            fn checked_mul(self, offset: usize) -> Option<Self> {
                offset.try_into().ok().and_then(|o| self.checked_mul(o))
            }

            fn checked_div(self, offset: usize) -> Option<Self> {
                offset.try_into().ok().and_then(|o| self.checked_div(o))
            }
        }
    };
}

impl_unsigned!(u8);
impl_unsigned!(u16);
impl_unsigned!(u32);
#[cfg(target_pointer_width = "64")]
impl_unsigned!(u64);
impl_unsigned!(usize);

#[cfg(test)]
mod tests {
    use super::*;

    mod tests_into_usize_and_from_usize {
        use super::*;

        #[test]
        fn test_into_usize_and_from_usize_u8() {
            let value: u8 = 42;
            let usize_value = value.into_usize();
            let value_back = u8::from_usize(usize_value);
            assert_eq!(value, value_back);
        }

        #[test]
        fn test_into_usize_and_from_usize_u16() {
            let value: u16 = 42;
            let usize_value = value.into_usize();
            let value_back = u16::from_usize(usize_value);
            assert_eq!(value, value_back);
        }

        #[test]
        fn test_into_usize_and_from_usize_u32() {
            let value: u32 = 42;
            let usize_value = value.into_usize();
            let value_back = u32::from_usize(usize_value);
            assert_eq!(value, value_back);
        }

        #[test]
        fn test_into_usize_and_from_usize_usize() {
            let value: usize = 42;
            let usize_value = value.into_usize();
            let value_back = usize::from_usize(usize_value);
            assert_eq!(value, value_back);
        }

        #[test]
        #[should_panic]
        fn test_from_usize_panics_when_value_exceeds_u8_max() {
            let _ = u8::from_usize(usize::from(u8::MAX) + 1);
        }

        #[test]
        #[should_panic]
        fn test_from_usize_panics_when_value_exceeds_u16_max() {
            let _ = u16::from_usize(usize::from(u16::MAX) + 1);
        }

        #[test]
        #[should_panic]
        fn test_from_usize_panics_when_value_exceeds_u32_max() {
            let _ = u32::from_usize(u32::MAX.into_usize() + 1);
        }
    }

    mod tests_checked_add {
        use super::*;

        #[test]
        fn test_checked_add_returns_none_on_u8_overflow() {
            let result = <u8 as Unsigned>::checked_add(u8::MAX, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_add_returns_none_on_u16_overflow() {
            let result = <u16 as Unsigned>::checked_add(u16::MAX, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_add_returns_none_on_u32_overflow() {
            let result = <u32 as Unsigned>::checked_add(u32::MAX, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_add_returns_none_on_usize_overflow() {
            let result = <usize as Unsigned>::checked_add(usize::MAX, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_add_returns_none_when_offset_does_not_fit_target_type() {
            let result = <u8 as Unsigned>::checked_add(0, usize::from(u8::MAX) + 1);
            assert_eq!(result, None);
        }
    }

    mod tests_checked_sub {
        use super::*;

        #[test]
        fn test_checked_sub_returns_none_on_u8_underflow() {
            let result = <u8 as Unsigned>::checked_sub(0, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_sub_returns_none_on_u16_underflow() {
            let result = <u16 as Unsigned>::checked_sub(0, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_sub_returns_none_on_u32_underflow() {
            let result = <u32 as Unsigned>::checked_sub(0, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_sub_returns_none_on_usize_underflow() {
            let result = <usize as Unsigned>::checked_sub(0, 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_sub_returns_none_when_offset_does_not_fit_target_type() {
            let result = <u8 as Unsigned>::checked_sub(u8::MAX, usize::from(u8::MAX) + 1);
            assert_eq!(result, None);
        }
    }

    mod tests_checked_mul {
        use super::*;

        #[test]
        fn test_checked_mul_returns_none_on_u8_overflow() {
            let result = <u8 as Unsigned>::checked_mul(u8::MAX, 2);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_mul_returns_none_on_u16_overflow() {
            let result = <u16 as Unsigned>::checked_mul(u16::MAX, 2);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_mul_returns_none_on_u32_overflow() {
            let result = <u32 as Unsigned>::checked_mul(u32::MAX, 2);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_mul_returns_none_on_usize_overflow() {
            let result = <usize as Unsigned>::checked_mul(usize::MAX, 2);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_mul_returns_none_when_offset_does_not_fit_target_type() {
            let result = <u8 as Unsigned>::checked_mul(1, usize::from(u8::MAX) + 1);
            assert_eq!(result, None);
        }
    }

    mod tests_checked_div {
        use super::*;

        #[test]
        fn test_checked_div_returns_none_on_u8_divide_by_zero() {
            let result = <u8 as Unsigned>::checked_div(42, 0);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_div_returns_none_on_u16_divide_by_zero() {
            let result = <u16 as Unsigned>::checked_div(42, 0);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_div_returns_none_on_u32_divide_by_zero() {
            let result = <u32 as Unsigned>::checked_div(42, 0);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_div_returns_none_on_usize_divide_by_zero() {
            let result = <usize as Unsigned>::checked_div(42, 0);
            assert_eq!(result, None);
        }

        #[test]
        fn test_checked_div_returns_none_when_offset_does_not_fit_target_type() {
            let result = <u8 as Unsigned>::checked_div(u8::MAX, usize::from(u8::MAX) + 1);
            assert_eq!(result, None);
        }
    }

    mod tests_align {
        use super::*;

        #[test]
        fn test_align_up_returns_none_when_offset_does_not_fit_target_type() {
            let result = <u8 as Unsigned>::align_up(u8::MAX, usize::from(u8::MAX) + 1);
            assert_eq!(result, None);
        }

        #[test]
        fn test_align_down_returns_zero_when_alignment_exceeds_target_type() {
            let result = <u8 as Unsigned>::align_down(u8::MAX, usize::from(u8::MAX) + 1);
            assert_eq!(result, Some(0));
        }
    }
}

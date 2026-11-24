//! Storage for the existing scene engine: inline for const scenes, or a
//! caller-owned named arena for runtime-sized scenes. No global allocator.
use core::ops::{Deref, DerefMut};
#[cfg(feature = "graph")]
pub type Arena = &'static (dyn core::alloc::Allocator + Sync);

pub(crate) struct Storage<T, const N: usize> {
    inline: [T; N],
    #[cfg(feature = "graph")]
    dynamic: Option<alloc::vec::Vec<T, Arena>>,
}
impl<T, const N: usize> Storage<T, N> {
    pub const fn inline(inline: [T; N]) -> Self {
        Self {
            inline,
            #[cfg(feature = "graph")]
            dynamic: None,
        }
    }
    #[cfg(feature = "graph")]
    pub fn in_arena(inline: [T; N], arena: Arena) -> Self {
        Self {
            inline,
            dynamic: Some(alloc::vec::Vec::new_in(arena)),
        }
    }
    #[cfg(feature = "graph")]
    pub fn reserve_to(&mut self, len: usize) -> Result<(), ()> {
        match self.dynamic.as_mut() {
            Some(values) => values
                .try_reserve_exact(len.saturating_sub(values.len()))
                .map_err(|_| ()),
            None if len <= N => Ok(()),
            None => Err(()),
        }
    }
    #[cfg(feature = "graph")]
    pub fn resize_reserved(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        if let Some(values) = self.dynamic.as_mut() {
            assert!(len <= values.capacity());
            values.resize(len, value);
        }
    }
}
impl<T, const N: usize> Deref for Storage<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        #[cfg(feature = "graph")]
        if let Some(values) = &self.dynamic {
            return values;
        }
        &self.inline
    }
}
impl<T, const N: usize> DerefMut for Storage<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        #[cfg(feature = "graph")]
        if let Some(values) = &mut self.dynamic {
            return values;
        }
        &mut self.inline
    }
}

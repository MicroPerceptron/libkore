//! Hash collections backed by hashbrown, with its upstream allocator parameters.
//!
//! Default constructors use the allocator installed by the application. This
//! crate installs no heap. Use `new_in` for an explicit allocator and
//! `try_reserve` when allocation failure must be recoverable.
//!
//! The default fast hasher does not provide the standard library's HashDoS
//! protection. Supply a suitable keyed hasher for attacker-controlled keys.
pub use hashbrown::{
    DefaultHashBuilder, Equivalent, HashMap, HashSet, HashTable, TryReserveError, hash_map,
    hash_set, hash_table,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn map_set_and_prehashed_table_use_the_application_heap() {
        let mut map = HashMap::new();
        assert_eq!(map.insert("key", 1), None);
        assert_eq!(map.insert("key", 2), Some(1));
        assert_eq!(map.remove("key"), Some(2));
        let mut set = HashSet::new();
        assert!(set.insert(7));
        assert!(!set.insert(7));
        let mut table = HashTable::new();
        table.insert_unique(42, 7, |_| 42);
        assert_eq!(table.find(42, |v| *v == 7), Some(&7));
        assert!(table.try_reserve(usize::MAX, |_| 42).is_err());
        assert_eq!(table.find(42, |v| *v == 7), Some(&7));
    }
}

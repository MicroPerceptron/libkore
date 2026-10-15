//! AArch64 guest architecture vocabulary.
//!
//! The kernel's EL2 guest backend is not implemented yet. No register
//! indices are exported until that backend defines its wire conversion;
//! in particular, this module must not inherit x86-64's register numbering.
//! Shared records and semantic role positions remain in the parent module.

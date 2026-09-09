//! Resolved presentation preference, never authority to modify desktop settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorScheme {
    Light,
    #[default]
    Dark,
}

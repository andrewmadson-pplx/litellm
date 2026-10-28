mod name_index;

use std::borrow::Cow;

pub use name_index::*;

use crate::wire::Fields;

/// A store `resolve` reads rows from. The Python host implements it over the bound dict, tests over a `BTreeMap`
pub trait CatalogView {
    fn contains(&self, key: &str) -> bool;
    fn row(&self, key: &str) -> Option<RowRef<'_>>;
    /// The actual key for a case-insensitive name, from the store's `NameIndex`
    fn by_normalized(&self, name: &str) -> Option<String>;
}

/// A row handed back by `CatalogView`, decoded into the typed fields
pub type RowRef<'a> = Cow<'a, Fields>;

use std::collections::HashMap;

/// The store's only secondary index, built from keys and each key's raw `litellm_provider`. It never holds a row,
/// so only adding or removing keys makes it stale
#[derive(Clone, Debug, Default)]
pub struct NameIndex {
    by_lower: HashMap<String, String>,
    by_provider: HashMap<String, Vec<String>>,
}

impl NameIndex {
    pub fn new<'a>(keys: impl IntoIterator<Item = (&'a str, Option<&'a str>)>) -> Self {
        keys.into_iter()
            .fold(Self::default(), |mut index, (key, provider)| {
                index.by_lower.insert(key.to_lowercase(), key.to_owned());
                if let Some(provider) = provider {
                    index
                        .by_provider
                        .entry(provider.to_owned())
                        .or_default()
                        .push(key.to_owned());
                }
                index
            })
    }

    pub fn key_for(&self, name: &str) -> Option<&str> {
        self.by_lower.get(&name.to_lowercase()).map(String::as_str)
    }

    pub fn provider_keys(&self, provider: &str) -> &[String] {
        self.by_provider.get(provider).map_or(&[], Vec::as_slice)
    }
}

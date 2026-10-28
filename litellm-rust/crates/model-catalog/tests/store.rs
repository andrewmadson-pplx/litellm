use litellm_model_catalog::NameIndex;
use rstest::rstest;

fn index() -> NameIndex {
    NameIndex::new([
        ("gpt-4o", Some("openai")),
        ("MyCorp/Custom-Model", Some("openai")),
        ("claude-sonnet", Some("anthropic")),
        ("no-provider", None),
    ])
}

#[rstest]
#[case::exact("gpt-4o", Some("gpt-4o"))]
#[case::upper("GPT-4O", Some("gpt-4o"))]
#[case::mixed_case_key("mycorp/custom-model", Some("MyCorp/Custom-Model"))]
#[case::providerless("No-Provider", Some("no-provider"))]
#[case::absent("gpt-5", None)]
fn key_for_finds_the_actual_key_in_any_case(#[case] name: &str, #[case] expected: Option<&str>) {
    assert_eq!(index().key_for(name), expected);
}

#[rstest]
#[case::two_keys("openai", &["gpt-4o", "MyCorp/Custom-Model"])]
#[case::one_key("anthropic", &["claude-sonnet"])]
#[case::raw_name_only("OpenAI", &[])]
#[case::unknown("azure", &[])]
fn provider_keys_lists_keys_by_raw_provider(#[case] provider: &str, #[case] expected: &[&str]) {
    assert_eq!(index().provider_keys(provider), expected);
}

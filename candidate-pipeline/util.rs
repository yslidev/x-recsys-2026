pub fn short_type_name(full: &'static str) -> &'static str {
    full.rsplit("::").next().unwrap_or(full)
}

//! Shared ProviderDecl helpers for ACP-backed provider sidecars.

use serde_json::{json, Value};

/// The relay token field the manifest + host agree on.
pub const RELAY_SECRET: &str = "relay_token";
/// Placeholder base URL — the real URL arrives per turn in
/// `provider/chat`'s `relay_url`.
pub const PLACEHOLDER_BASE: &str = "https://127.0.0.1:1/";

/// Build the standard provider declaration for an ACP-backed plugin.
///
/// `cache_ttl_secs`: honest prompt-cache lifetime — an ACP child keeps
/// the vendor's own thread cache warm for ~minutes.
pub fn provider_decl(
    provider_id: &str,
    display_name: &str,
    auth_method_id: &str,
    auth_method_name: &str,
    auth_kind: &str, // "oauth" | "api_key"
    operations: &[&str],
    cache_ttl_secs: u64,
) -> Value {
    json!({
        "id": provider_id,
        "name": display_name,
        "transport": {
            "kind": "openai-responses",
            "base_url": PLACEHOLDER_BASE,
            "authorization": {"kind": "bearer", "secret_name": RELAY_SECRET},
            "request": {
                "prompt_cache_key": false,
                "store": false,
                "include_reasoning_encrypted": true,
                "previous_response_id": false,
                "tool_choice": "auto",
                "parallel_tool_calls": true,
                "text_verbosity": "low",
                "cache_ttl_secs": cache_ttl_secs,
            },
            "headers": [
                {"name": "session-id", "required": true,
                 "source": {"kind": "session_id"}},
            ],
        },
        "auth_methods": [{
            "id": auth_method_id,
            "name": auth_method_name,
            "kind": auth_kind,
            "operations": operations,
        }],
    })
}

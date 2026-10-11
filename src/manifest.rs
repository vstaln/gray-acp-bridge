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
/// the vendor's own thread cache warm for ~minutes. 0 = unknown: the
/// host then never schedules a warming replay.
///
/// Leaves `request.warm_replay` off: a sidecar that spawns a CLI child
/// per turn would turn a host replay into a duplicate prompt. Plain
/// HTTPS relays want [`provider_decl_http`].
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

/// [`provider_decl`] for a sidecar that only forwards each request to a
/// stateless vendor HTTP API (no CLI child, no server-side thread): the
/// host may verbatim-replay a request to keep the vendor's prefix cache
/// warm through a long tool phase (`request.warm_replay`), and may
/// rewrite already-sent history on a cold cache. One more HTTPS request
/// capped at a single output token is all a replay costs here. The
/// warmer still only fires when `cache_ttl_secs` > 0 and the prompt is
/// big enough to pay for itself.
pub fn provider_decl_http(
    provider_id: &str,
    display_name: &str,
    auth_method_id: &str,
    auth_method_name: &str,
    auth_kind: &str,
    operations: &[&str],
    cache_ttl_secs: u64,
) -> Value {
    let mut decl = provider_decl(
        provider_id,
        display_name,
        auth_method_id,
        auth_method_name,
        auth_kind,
        operations,
        cache_ttl_secs,
    );
    decl["transport"]["request"]["warm_replay"] = Value::Bool(true);
    decl
}

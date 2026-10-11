//! acp_bridge — shared plumbing for gray provider sidecars that drive an
//! ACP-speaking agent CLI (cursor-agent, grok, opencode acp).
//!
//! Pieces:
//! - `acp`: JSON-RPC-over-stdio ACP client (initialize, authenticate,
//!   session/new|load, session/prompt, session/cancel, set_model,
//!   set_config_option, set_mode), with an update/notification dispatch.
//! - `relay`: single-admission loopback HTTP relay — the host POSTs one
//!   OpenAI Responses request; we stream Responses SSE back.
//! - `mcp`: minimal MCP server (streamable-HTTP style POST/GET) that
//!   exposes host tools as `mcp__gray__*` and parks `tools/call` until a
//!   later turn's `function_call_output` answers it.
//! - `chat`: Responses-request → ACP turn translation and the Responses
//!   SSE event fold (created → output_item/delta/done → completed).
//! - `manifest`: ProviderDecl builders shared by the thin provider bins.
//! - `session`: pooled ACP children keyed by conversation, parked-call
//!   resume, idle reaper.

pub mod acp;
pub mod chat;
pub mod guard;
pub mod manifest;
pub mod mcp;
pub mod relay;
pub mod session;

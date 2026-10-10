# gray-acp-bridge

Shared plumbing for gray **provider sidecars** — the crate every
`gray-*-sub` provider builds on. Not a plugin itself; a library in the
[gray](https://github.com/vstaln/gray) plugin workspace.

## Pieces

| Module | Role |
|---|---|
| `acp` | JSON-RPC-over-stdio client for ACP-speaking agent CLIs (`cursor-agent`, `grok`, `opencode acp`): `initialize`, `authenticate`, `session/new\|load`, `session/prompt`, `session/cancel`, `set_model`, `set_config_option`, `set_mode`, with update/notification dispatch. |
| `relay` | **Single-admission loopback HTTP relay** — the host POSTs one OpenAI-Responses request; the relay streams Responses SSE back and refuses repeats (one response per turn, by contract). |
| `chat` | Turn preparation (`prepare_turn`) and the Responses SSE fold (`fold_result`: `response.created` → item/delta events → `response.completed` → `[DONE]`). |
| `mcp` | Minimal streamable-HTTP MCP server exposing host tools as `mcp__gray__*`; parks `tools/call` until a later turn's `function_call_output` answers it. |
| `manifest` | Provider-manifest helpers shared by the sidecars. |
| `session` | Session bookkeeping for ACP-driven providers. |

## The contract it serves

Gray's host only accepts `openai-responses` provider transports. A
provider plugin therefore doesn't need a streaming client of its own:
it mints a per-turn relay (`relay::mint_turn` → `{relay_url,
relay_token}`), the host POSTs one request, the sidecar answers it by
whatever means (ACP session, OpenAI-compat upstream, OAuth API), and the
relay emits canonical Responses SSE back. Credentials arrive wrapped in
`CredentialEnvelope`; secret fields are redacted in `Debug`.

Used by: `gray-claude-sub`, `gray-antigravity-sub`, `gray-copilot-sub`,
`gray-cursor-sub`, `gray-devin-sub`, `gray-muse-sub`, and the
`gray-openai-compat` family.

## License

MIT

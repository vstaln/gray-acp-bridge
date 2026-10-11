//! Pooled ACP children + the turn engine.
//!
//! One pooled session = one spawned agent process + one ACP session.
//! Continuation matching (chat::continuation) routes a Responses request
//! to the session that already absorbed its history. A turn suspended on
//! parked MCP calls stays pooled; the next request's
//! `function_call_output` tail resolves the calls and the SAME prompt
//! turn resumes streaming.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::acp::{AcpClient, SpawnSpec, Wire};
use crate::chat::{self, OutItem, PreparedTurn, TurnResult};
use crate::guard::{self, TurnGuard};
use crate::mcp::{McpPark, PendingCall};

const MAX_SESSIONS: usize = 4;
/// A parked call freezes its ACP turn: after this much silence we end the
/// Responses turn with the emitted function_calls and suspend the session.
const PARK_QUIESCE: Duration = Duration::from_secs(10);
/// Idle pooled session is killed after this.
const IDLE_TIMEOUT: Duration = Duration::from_secs(45 * 60);
/// Reaper cadence.
const REAPER_TICK: Duration = Duration::from_secs(30);
/// Budget for a single update wait while a turn streams.
const UPDATE_WAIT: Duration = Duration::from_millis(250);
/// Give a cancelled turn this long to settle.

/// Everything a provider binary must supply.
pub trait Driver: Send + Sync {
    /// `agent` argv/env for one staged child.
    fn spawn_spec(&self, stage: &std::path::Path) -> SpawnSpec;
    /// ACP `authenticate` methodId, if the agent needs one before sessions.
    fn auth_method_id(&self) -> Option<&'static str>;
    /// clientInfo {name, version} for initialize.
    fn client_info(&self) -> (&'static str, &'static str);
    /// `session/set_mode` to apply right after session/new (e.g. "ask").
    fn session_mode(&self) -> Option<&'static str> {
        None
    }
    /// Extra clientCapabilities merged into initialize.
    fn client_capabilities(&self) -> Value {
        json!({"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false})
    }
    /// Model selection: set_model (+ config options / effort meta).
    /// `session_new_result` carries the session/new response (models,
    /// configOptions) so the driver can resolve config-option ids.
    fn apply_model(
        &self,
        client: &mut AcpClient,
        session_id: &str,
        session_new_result: &Value,
        model: &str,
        effort: Option<&str>,
        overrides: &[(String, String)],
    ) -> Result<(), String>;
    /// `_meta` merged into session/prompt params (grok needs promptId).
    fn prompt_meta(&self, seq: u64) -> Value {
        let _ = seq;
        Value::Null
    }
    /// Notification names that complete a prompt turn besides the RPC
    /// response (grok: `_x.ai/session/prompt_complete`).
    fn prompt_complete_methods(&self) -> &'static [&'static str] {
        &[]
    }
    /// Handle an agent→client extension request (cursor askQuestion,
    /// grok ask_user_question). Return the result value, or None to answer
    /// -32601.
    fn ext_request(&self, _method: &str, _params: &Value) -> Option<Value> {
        None
    }
    /// Pick a permission optionId for session/request_permission, or None
    /// to refuse the tool.
    fn permission_option(&self, params: &Value) -> PermissionDecision {
        let _ = params;
        PermissionDecision::Deny
    }
}

pub enum PermissionDecision {
    /// Respond outcome.selected with this optionId.
    Allow(String),
    /// Respond outcome.cancelled.
    Deny,
}

struct Parked {
    call: PendingCall,
}

pub struct Session {
    client: AcpClient,
    mcp: Arc<McpPark>,
    session_id: String,
    /// The exact Responses `input` this session last fully consumed —
    /// verbatim, including our own echoed answer items (strict prefix).
    absorbed: Vec<Value>,
    /// Echo-zone signature of the last completed turn's answer.
    reply_call_ids: Vec<String>,
    reply_text: String,
    parked: HashMap<String, Parked>,
    mid_turn: bool,
    model: String,
    /// stdout EOF observed — drop from the pool after this drive call.
    dead: bool,
    prompt_seq: u64,
    last_used: Instant,
    prior_cut: bool,
    _stage: tempfile::TempDir,
}

pub struct Pool {
    sessions: Mutex<HashMap<String, Session>>,
    driver: Arc<dyn Driver>,
}

impl Pool {
    pub fn new(driver: Arc<dyn Driver>) -> Arc<Self> {
        let pool = Arc::new(Self {
            sessions: Mutex::new(HashMap::new()),
            driver,
        });
        let p = pool.clone();
        std::thread::Builder::new()
            .name("acp-sub-reaper".into())
            .spawn(move || loop {
                std::thread::sleep(REAPER_TICK);
                if let Ok(mut sessions) = p.sessions.lock() {
                    sessions.retain(|_, s| {
                        if s.mid_turn {
                            return true; // suspended turns outlive idle timeout
                        }
                        s.last_used.elapsed() < IDLE_TIMEOUT
                    });
                }
            })
            .ok();
        pool
    }
}

/// The full relay turn: `body` is the raw Responses request bytes.
/// Returns the complete SSE byte stream.
pub fn run_turn(
    pool: &Arc<Pool>,
    raw: &[u8],
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    let body: Value = serde_json::from_slice(raw).map_err(|e| format!("bad request json: {e}"))?;
    let model_hint = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let mut prep = chat::prepare_turn(&body, &model_hint)?;

    let mut guard = pool.sessions.lock().map_err(|_| "pool lock".to_string())?;

    // 1) A suspended session whose parked calls this tail answers wins:
    // verbatim absorbed prefix + any tail function_call_output hitting a
    // parked call_id. The echo items replayed in between are ignored —
    // the child holds the originals.
    let mut chosen: Option<String> = None;
    let mut resume_tail: Vec<Value> = Vec::new();
    'outer: for (key, s) in guard.iter() {
        if s.model != prep.model || !s.mid_turn || !prep.input.starts_with(&s.absorbed) {
            continue;
        }
        let region = &prep.input[s.absorbed.len()..];
        if region.iter().any(|i| {
            chat_item_type(i) == "function_call_output"
                && i.get("call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|c| s.parked.contains_key(c))
        }) {
            resume_tail = region.to_vec();
            chosen = Some(key.clone());
            break 'outer;
        }
    }

    // 2) Continuation match against absorbed + last-reply echo.
    let mut tail: Vec<Value> = Vec::new();
    if chosen.is_none() {
        for (key, s) in guard.iter() {
            if s.model != prep.model || s.mid_turn {
                continue;
            }
            if let Ok((t, _undelivered)) = chat::continuation(
                &s.absorbed,
                &s.reply_call_ids,
                &s.reply_text,
                &prep.input,
            ) {
                tail = t.to_vec();
                chosen = Some(key.clone());
                break;
            }
        }
    } else {
        tail = resume_tail;
    }

    let fresh = chosen.is_none();
    let key = match chosen {
        Some(k) => k,
        None => {
            if guard.len() >= MAX_SESSIONS {
                // evict oldest idle
                if let Some(oldest) = guard
                    .iter()
                    .filter(|(_, s)| !s.mid_turn)
                    .min_by_key(|(_, s)| s.last_used)
                    .map(|(k, _)| k.clone())
                {
                    guard.remove(&oldest);
                }
            }
            let key = format!("sess-{}", chat::rand_hex(8));
            let session = spawn_session(&*pool.driver, &prep)?;
            guard.insert(key.clone(), session);
            tail = prep.input.clone();
            key
        }
    };

    let session = guard.get_mut(&key).ok_or("session vanished")?;
    session.last_used = Instant::now();
    session.mcp.set_tools(to_mcp_tools(&prep.tools));
    prep.tail = tail;
    chat::split_tail(&mut prep);

    let result = drive(session, &*pool.driver, &mut prep, cancel, fresh);
    if session.dead {
        guard.remove(&key);
    }
    result.map(|r| {
        let response_id = format!("resp_{}", chat::rand_hex(24));
        chat::fold_result(&r, &prep.model, &response_id)
    })
}

fn chat_item_type(i: &Value) -> &str {
    match i.get("type").and_then(Value::as_str) {
        Some(t) => t,
        None if i.get("role").is_some() => "message",
        None => "",
    }
}

fn spawn_session(
    driver: &dyn Driver,
    prep: &PreparedTurn,
) -> Result<Session, String> {
    let mcp = McpPark::start()?;
    let stage = tempfile::TempDir::new().map_err(|e| format!("stage dir: {e}"))?;
    let spec = driver.spawn_spec(stage.path());
    let mut client = AcpClient::spawn(&spec).map_err(|e| {
        format!(
            "spawn {} {}: {e}",
            spec.command,
            spec.args.join(" ")
        )
    })?;
    let caps = driver.client_capabilities();
    let (name, ver) = driver.client_info();
    let init = client
        .initialize((name, ver), caps)
        .map_err(|e| format!("initialize: {e}"))?;
    if let Some(method) = driver.auth_method_id() {
        client
            .authenticate(method)
            .map_err(|e| format!("authenticate: {e}"))?;
    }
    let mcp_servers = json!([mcp.server_entry("gray")]);
    let mcp = mcp;
    let sess = client
        .session_new(&stage.path().to_string_lossy(), mcp_servers)
        .map_err(|e| format!("session/new: {e}"))?;
    let session_id = sess
        .get("sessionId")
        .and_then(Value::as_str)
        .ok_or("session/new: no sessionId")?
        .to_string();
    if let Some(mode) = driver.session_mode() {
        // Best-effort: some agents lack set_mode.
        let _ = client.set_mode(&session_id, mode);
    }
    if !prep.model.is_empty() {
        driver.apply_model(&mut client, &session_id, &sess, &prep.model, prep.effort.as_deref(), &prep.overrides)?;
    }
    let _ = init;
    Ok(Session {
        client,
        mcp,
        session_id,
        absorbed: Vec::new(),
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        parked: HashMap::new(),
        mid_turn: false,
        model: prep.model.clone(),
        dead: false,
        prompt_seq: 0,
        last_used: Instant::now(),
        prior_cut: false,
        _stage: stage,
    })
}

/// Convert host Responses `tools[]` ({type:"function",name,description,
/// parameters}) to MCP tool defs ({name,description,inputSchema}).
pub fn to_mcp_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|t| {
            let (name, desc, params) = if t.get("type").and_then(Value::as_str) == Some("function") {
                // Responses shape: {type:"function", name, description, parameters}
                (
                    t.get("name"),
                    t.get("description"),
                    t.get("parameters"),
                )
            } else {
                // already MCP-ish?
                (
                    t.get("name"),
                    t.get("description"),
                    t.get("inputSchema").or_else(|| t.get("parameters")),
                )
            };
            let name = name?.as_str()?;
            if name.is_empty() || name.len() > 50 {
                return None;
            }
            let mut schema = params.cloned().unwrap_or_else(|| json!({"type":"object","properties":{}}));
            normalize_schema(&mut schema);
            Some(json!({
                "name": name,
                "description": desc.and_then(Value::as_str).unwrap_or(""),
                "inputSchema": schema,
            }))
        })
        .collect()
}

fn normalize_schema(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        for key in ["oneOf", "allOf", "anyOf"] {
            obj.remove(key);
        }
        obj.entry("type".to_string())
            .or_insert(Value::String("object".to_string()));
        if obj.get("type").and_then(Value::as_str) == Some("object")
            && !matches!(obj.get("properties"), Some(Value::Object(_)))
        {
            obj.insert("properties".to_string(), json!({}));
        }
    }
}

fn drive(
    session: &mut Session,
    driver: &dyn Driver,
    prep: &mut PreparedTurn,
    cancel: &AtomicBool,
    fresh: bool,
) -> Result<TurnResult, String> {
    let mut turn_guard = TurnGuard::default();
    let mut r = drive_inner(session, driver, prep, cancel, fresh, &mut turn_guard)?;
    guard::clean_items(&mut r.items);
    if !session.mid_turn {
        let (call_ids, text) = chat::reply_signature(&r);
        session.reply_call_ids = call_ids;
        session.reply_text = text;
    }
    Ok(r)
}

/// Run/continue one turn on `session`.
fn drive_inner(
    session: &mut Session,
    driver: &dyn Driver,
    prep: &mut PreparedTurn,
    cancel: &AtomicBool,
    fresh: bool,
    turn_guard: &mut TurnGuard,
) -> Result<TurnResult, String> {
    let mut result = TurnResult::default();
    result.stop_reason = "end_turn".to_string();
    let mut prompt_rx: Option<std::sync::mpsc::Receiver<Value>> = None;
    // User-side tail items that arrived while a turn was suspended get
    // queued as a follow-up prompt once the current one completes.
    let mut queued_prompt: Vec<Value> = Vec::new();

    // Resume path: answer parked calls from the tail's outputs; leftover
    // user-side tail (a message sent alongside) queues as a next prompt.
    if session.mid_turn {
        let mut saw_output = false;
        for item in &prep.tail {
            if chat_item_type(item) == "function_call_output" {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let output = item.get("output").map(|o| match o {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }).unwrap_or_default();
                if let Some(p) = session.parked.remove(&call_id) {
                    p.call.resolve(output);
                    saw_output = true;
                } else if !chat::assistant_side(item) {
                    queued_prompt.push(json!({"type":"text",
                        "text": format!("Tool result for {call_id}: {output}")}));
                }
            } else if !chat::assistant_side(item) {
                if let Some(text) = chat::item_to_transcript_pub(item) {
                    queued_prompt.push(json!({"type": "text", "text": text}));
                }
            }
        }
        let _ = saw_output;
    } else {
        // New prompt on this session.
        let mut prompt_blocks: Vec<Value> = Vec::new();
        if std::mem::take(&mut session.prior_cut) && !fresh {
            prompt_blocks.push(json!({"type": "text", "text": guard::CUT_NOTE}));
        }
        if fresh && !prep.system.is_empty() {
            prompt_blocks.push(json!({"type": "text",
                "text": format!("System instructions:\n{}\n\n{}", prep.system, guard::ONE_TURN_RULE)}));
        } else if fresh {
            prompt_blocks.push(json!({"type": "text",
                "text": format!("System instructions:\n{}", guard::ONE_TURN_RULE)}));
        }
        if prep.tail_prompt.is_empty() && prep.outputs.is_empty() && !fresh {
            // Nothing new — nudge the model rather than send an empty prompt.
            prompt_blocks.push(json!({"type": "text", "text": "Continue."}));
        } else {
            prompt_blocks.extend(prep.tail_prompt.clone());
        }
        // function_call_output items that own no parked call still carry
        // information the model must see — render them as transcript text.
        for (call_id, output) in prep.outputs.drain(..) {
            prompt_blocks.push(json!({"type": "text",
                "text": format!("Tool result for {call_id}: {output}")}));
        }
        if prompt_blocks.is_empty() && session.mid_turn == false {
            // Nothing renderable — nudge rather than send an empty prompt
            // (which would hang the turn forever).
            prompt_blocks.push(json!({"type": "text", "text": "Continue."}));
        }
        if !prompt_blocks.is_empty() {
            session.prompt_seq += 1;
            let meta = driver.prompt_meta(session.prompt_seq);
            let (_id, rx) = session
                .client
                .prompt(&session.session_id.clone(), Value::Array(prompt_blocks), meta)
                .map_err(|e| format!("session/prompt: {e}"))?;
            prompt_rx = Some(rx);
            session.mid_turn = true;
        }
    }

    // Stream updates until the prompt resolves or we suspend on parks.
    let mut quiesce: Option<Instant> = None;
    let mut text_acc = String::new();
    let mut thought_acc = String::new();
    let flush_text = |result: &mut TurnResult, text_acc: &mut String| {
        if !text_acc.is_empty() {
            result.items.push(OutItem::Message {
                text: std::mem::take(text_acc),
            });
        }
    };
    let flush_thought = |result: &mut TurnResult, thought_acc: &mut String| {
        if !thought_acc.is_empty() {
            result.items.push(OutItem::Reasoning {
                text: std::mem::take(thought_acc),
            });
        }
    };

    let absorbed_mark = session.absorbed.len();
    let emit_mark = result.items.len();
    let _ = (absorbed_mark, emit_mark);

    loop {
        if cancel.load(Ordering::SeqCst) {
            session.client.cancel(&session.session_id);
            session.mid_turn = false;
            fail_parked(session, "interrupted");
            return Err("turn cancelled by client".to_string());
        }

        // Parked MCP calls arrived?
        for pending in session.mcp.drain() {
            if turn_guard.cut().is_some() {
                pending.fail("reply was cut at a fabricated transcript record; this call never ran");
                continue;
            }
            flush_text(&mut result, &mut text_acc);
            let (call_id, name) = (pending.call_id.clone(), pending.name.clone());
            let arguments = pending.arguments.to_string();
            session.parked.insert(call_id.clone(), Parked { call: pending });
            result.items.push(OutItem::FunctionCall {
                call_id,
                name,
                arguments,
            });
            quiesce = None;
        }

        // Suspension check: parked calls + quiesced stream → end turn.
        if !session.parked.is_empty() {
            match quiesce {
                Some(t) if t.elapsed() >= PARK_QUIESCE => {
                    flush_text(&mut result, &mut text_acc);
                    flush_thought(&mut result, &mut thought_acc);
                    result.suspended = true;
                    session.last_used = Instant::now();
                    return Ok(result);
                }
                None => quiesce = Some(Instant::now()),
                _ => {}
            }
        }

        // Did the prompt RPC resolve? That ends the turn for real —
        // parked MCP calls still open would have kept it pending.
        if let Some(rx) = &prompt_rx {
            match rx.try_recv() {
                Ok(msg) => {
                    if let Some(err) = msg.get("error") {
                        session.mid_turn = false;
                        fail_parked(session, "prompt failed");
                        return Err(format!(
                            "session/prompt: {}",
                            err.get("message").and_then(Value::as_str).unwrap_or("?")
                        ));
                    }
                    let stop = msg
                        .get("result")
                        .and_then(|r| r.get("stopReason"))
                        .and_then(Value::as_str)
                        .unwrap_or("end_turn")
                        .to_string();
                    if let Some(u) = msg.get("result").and_then(|r| r.get("usage")).and_then(chat::Usage::from_acp) {
                        result.usage = Some(u);
                    }
                    result.stop_reason = stop;
                    let _ = prompt_rx.take();
                    if !queued_prompt.is_empty() {
                        // A user-side tail rode along with the tool
                        // outputs — ask it now, keep streaming.
                        session.prompt_seq += 1;
                        let meta = driver.prompt_meta(session.prompt_seq);
                        match session.client.prompt(
                            &session.session_id.clone(),
                            Value::Array(std::mem::take(&mut queued_prompt)),
                            meta,
                        ) {
                            Ok((_id, rx)) => {
                                prompt_rx = Some(rx);
                                continue;
                            }
                            Err(e) => {
                                session.mid_turn = false;
                                fail_parked(session, "prompt failed");
                                return Err(format!("session/prompt: {e}"));
                            }
                        }
                    }
                    flush_text(&mut result, &mut text_acc);
                    flush_thought(&mut result, &mut thought_acc);
                    finish_turn(session, &mut result, prep);
                    return Ok(result);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    prompt_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }

        let wire = session.client.recv(UPDATE_WAIT);
        let Some(w) = wire else { continue };
        match w {
            Wire::Eof => {
                session.mid_turn = false;
                session.dead = true;
                fail_parked(session, "agent exited");
                flush_text(&mut result, &mut text_acc);
                flush_thought(&mut result, &mut thought_acc);
                if result.items.is_empty() {
                    let tail = session.stderr_tail_join();
                    return Err(format!("agent exited mid-turn{tail}"));
                }
                result.suspended = true; // keep whatever we emitted
                session.last_used = Instant::now();
                return Ok(result);
            }
            Wire::Bad => {}
            Wire::Notification { method, params } => {
                if method == "session/update" {
                    let cut = handle_update(&params, &mut result, &mut text_acc, &mut thought_acc,
                                  &flush_text, &flush_thought, turn_guard);
                    if let Some(m) = cut {
                        eprintln!(
                            "acp-bridge: reply cut at fabricated {} record (offset {}); {}",
                            guard::marker_name(m),
                            text_acc.len(),
                            if session.parked.is_empty() { "upstream cancelled" } else { "draining (calls parked)" }
                        );
                        session.prior_cut = true;
                        if session.parked.is_empty() {
                            session.client.cancel(&session.session_id);
                        }
                    }
                    if !session.parked.is_empty() {
                        quiesce = Some(Instant::now());
                    }
                } else if driver.prompt_complete_methods().contains(&method.as_str()) {
                    flush_text(&mut result, &mut text_acc);
                    flush_thought(&mut result, &mut thought_acc);
                    finish_turn(session, &mut result, prep);
                    return Ok(result);
                }
            }
            Wire::Request { id, method, params } => {
                handle_agent_request(session, driver, &id, &method, &params);
                quiesce = None;
            }
        }
    }
}

fn finish_turn(session: &mut Session, result: &mut TurnResult, prep: &mut PreparedTurn) {
    session.mid_turn = false;
    session.last_used = Instant::now();
    // The whole input — verbatim, echo items included — is now absorbed.
    session.absorbed = prep.input.clone();
    let (call_ids, text) = chat::reply_signature(result);
    session.reply_call_ids = call_ids;
    session.reply_text = text;
    fail_parked(session, "turn ended");
}

fn fail_parked(session: &mut Session, message: &str) {
    for (_, p) in session.parked.drain() {
        p.call.fail(message);
    }
}

impl Session {
    fn stderr_tail_join(&self) -> String {
        let tail = self.client.stderr_tail();
        if tail.is_empty() {
            String::new()
        } else {
            format!(" (stderr: {})", tail.join(" | "))
        }
    }
}

fn handle_agent_request(
    session: &mut Session,
    driver: &dyn Driver,
    id: &Value,
    method: &str,
    params: &Value,
) {
    match method {
        "session/request_permission" => {
            match driver.permission_option(params) {
                PermissionDecision::Allow(option_id) => {
                    let _ = session.client.respond(
                        id,
                        json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
                    );
                }
                PermissionDecision::Deny => {
                    let _ = session.client.respond(
                        id,
                        json!({"outcome": {"outcome": "cancelled"}}),
                    );
                }
            }
        }
        "fs/read_text_file" | "fs/readTextFile" | "fs/write_text_file"
        | "fs/writeTextFile" | "terminal/create" | "terminal/output"
        | "terminal/release" | "terminal/kill" | "terminal/wait_for_exit" => {
            let _ = session
                .client
                .respond_error(id, -32601, "client capability not supported");
        }
        other => match driver.ext_request(other, params) {
            Some(result) => {
                let _ = session.client.respond(id, result);
            }
            None => {
                let _ = session
                    .client
                    .respond_error(id, -32601, "unsupported client method");
            }
        },
    }
}

fn handle_update(
    params: &Value,
    result: &mut TurnResult,
    text_acc: &mut String,
    thought_acc: &mut String,
    flush_text: &dyn Fn(&mut TurnResult, &mut String),
    _flush_thought: &dyn Fn(&mut TurnResult, &mut String),
    turn_guard: &mut TurnGuard,
) -> Option<&'static str> {
    let update = params.get("update")?;
    let kind = update.get("sessionUpdate").and_then(Value::as_str).unwrap_or("");
    if turn_guard.cut().is_some() && kind != "usage_update" {
        return None;
    }
    match kind {
        "agent_message_chunk" => {
            if let Some(t) = content_text(update.get("content")) {
                if text_acc.is_empty() {
                    turn_guard.new_segment();
                }
                text_acc.push_str(&t);
                return turn_guard.feed(text_acc);
            }
        }
        "agent_thought_chunk" => {
            if let Some(t) = content_text(update.get("content")) {
                thought_acc.push_str(&t);
            }
        }
        "tool_call" | "tool_call_update" => {
            // Agent's own tool narrations — surfaced as reasoning context
            // so the host UI shows activity without faking host tool calls.
            let title = update.get("title").and_then(Value::as_str).unwrap_or("");
            if kind == "tool_call" && !title.is_empty() {
                flush_text(result, text_acc);
                thought_acc.push_str(&format!("[tool: {title}]\n"));
            }
        }
        "usage_update" => {
            if let Some(u) = update.get("usage").or_else(|| Some(update)) {
                if let Some(usage) = chat::Usage::from_acp(u) {
                    result.usage = Some(usage);
                }
            }
        }
        "plan" | "available_commands_update" | "current_mode_update"
        | "config_option_update" | "session_info_update" | "user_message_chunk" => {}
        _ => {}
    }
    None
}

fn content_text(content: Option<&Value>) -> Option<String> {
    let c = content?;
    if c.get("type").and_then(Value::as_str) == Some("text") {
        return c.get("text").and_then(Value::as_str).map(str::to_string);
    }
    None
}

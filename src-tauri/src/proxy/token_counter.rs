//! Local token estimation for Anthropic `count_tokens`.
//!
//! Used when the selected upstream cannot serve
//! `POST /v1/messages/count_tokens` (OpenAI Chat / Responses / Gemini native
//! gateways, or Anthropic-compatible relays that 404 the endpoint).
//!
//! This is intentionally a fast heuristic — good enough for Claude Desktop's
//! context-window UI so it stops falling back to `max_tokens=1` probes. It is
//! not a byte-identical Anthropic tokenizer.

use serde_json::Value;

/// Anthropic-compatible count_tokens response body.
pub fn count_tokens_response(input_tokens: u64) -> Value {
    serde_json::json!({ "input_tokens": input_tokens })
}

/// Estimate input tokens for an Anthropic Messages-shaped count_tokens body.
pub fn estimate_anthropic_count_tokens(body: &Value) -> u64 {
    let mut total: u64 = 0;

    if let Some(system) = body.get("system") {
        total = total.saturating_add(estimate_system(system));
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            total = total.saturating_add(estimate_message(message));
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            total = total.saturating_add(estimate_tool(tool));
        }
        // Tool-choice / framing overhead observed on real Anthropic counts.
        total = total.saturating_add(12);
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        total = total.saturating_add(estimate_json_value(tool_choice));
    }

    // Request framing (roles / message separators). Keep small so empty bodies
    // still return a non-zero, Desktop-friendly number.
    total = total.saturating_add(3);
    total.max(1)
}

/// Whether this Claude api_format is expected to own a real Anthropic
/// `/v1/messages/count_tokens` upstream. Transform formats almost never do.
pub fn api_format_supports_upstream_count_tokens(api_format: &str) -> bool {
    matches!(api_format.trim(), "anthropic" | "")
}

fn estimate_system(system: &Value) -> u64 {
    match system {
        Value::String(text) => estimate_text_tokens(text),
        Value::Array(blocks) => blocks.iter().map(estimate_content_block).sum(),
        other => estimate_json_value(other),
    }
}

fn estimate_message(message: &Value) -> u64 {
    let mut total: u64 = 4; // role + message framing
    match message.get("content") {
        Some(Value::String(text)) => total = total.saturating_add(estimate_text_tokens(text)),
        Some(Value::Array(blocks)) => {
            for block in blocks {
                total = total.saturating_add(estimate_content_block(block));
            }
        }
        Some(other) => total = total.saturating_add(estimate_json_value(other)),
        None => {}
    }
    total
}

fn estimate_content_block(block: &Value) -> u64 {
    let block_type = block
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match block_type {
        "text" => block
            .get("text")
            .and_then(Value::as_str)
            .map(estimate_text_tokens)
            .unwrap_or(0),
        "image" | "input_image" => estimate_image_block(block),
        "tool_use" => {
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .map(estimate_text_tokens)
                .unwrap_or(0);
            let input = block.get("input").map(estimate_json_value).unwrap_or(0);
            name.saturating_add(input).saturating_add(8)
        }
        "tool_result" => {
            let mut total: u64 = 8;
            match block.get("content") {
                Some(Value::String(text)) => {
                    total = total.saturating_add(estimate_text_tokens(text))
                }
                Some(Value::Array(blocks)) => {
                    for nested in blocks {
                        total = total.saturating_add(estimate_content_block(nested));
                    }
                }
                Some(other) => total = total.saturating_add(estimate_json_value(other)),
                None => {}
            }
            total
        }
        "thinking" | "redacted_thinking" => block
            .get("thinking")
            .or_else(|| block.get("data"))
            .and_then(Value::as_str)
            .map(estimate_text_tokens)
            .unwrap_or(32),
        "document" | "file" => {
            // Documents vary wildly; prefer explicit text, otherwise a fixed pad.
            if let Some(text) = block
                .get("source")
                .and_then(|s| s.get("data"))
                .and_then(Value::as_str)
            {
                // Base64 document — rough compressed estimate.
                ((text.len() as u64) / 8).max(256)
            } else {
                512
            }
        }
        _ => estimate_json_value(block),
    }
}

fn estimate_image_block(block: &Value) -> u64 {
    // Anthropic bills images in fixed-ish buckets. Without pixel dims, use a
    // mid-size default so the context meter stays conservative.
    let source = block.get("source").or_else(|| block.get("image_url"));
    if let Some(data) = source
        .and_then(|s| s.get("data").or_else(|| s.get("url")))
        .and_then(Value::as_str)
    {
        if data.starts_with("data:") || data.len() > 256 {
            // Large inline payloads → high-res-ish cost.
            return 1600;
        }
    }
    800
}

fn estimate_tool(tool: &Value) -> u64 {
    let mut total: u64 = 12; // tool framing
    if let Some(name) = tool.get("name").and_then(Value::as_str) {
        total = total.saturating_add(estimate_text_tokens(name));
    }
    if let Some(desc) = tool.get("description").and_then(Value::as_str) {
        total = total.saturating_add(estimate_text_tokens(desc));
    }
    if let Some(schema) = tool.get("input_schema").or_else(|| tool.get("parameters")) {
        total = total.saturating_add(estimate_json_value(schema));
    }
    total
}

fn estimate_json_value(value: &Value) -> u64 {
    match value {
        Value::Null | Value::Bool(_) => 1,
        Value::Number(n) => estimate_text_tokens(&n.to_string()),
        Value::String(s) => estimate_text_tokens(s),
        Value::Array(items) => items
            .iter()
            .map(estimate_json_value)
            .fold(2u64, |acc, n| acc.saturating_add(n)),
        Value::Object(map) => map.iter().fold(2u64, |acc, (k, v)| {
            acc.saturating_add(estimate_text_tokens(k))
                .saturating_add(estimate_json_value(v))
        }),
    }
}

/// Mixed ASCII / CJK heuristic approximating Claude-style BPE cost:
/// - ASCII runs ≈ 1 token / 4 chars
/// - non-ASCII (CJK, emoji, …) ≈ 1 token / char
fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }

    let mut tokens: u64 = 0;
    let mut ascii_run: u64 = 0;

    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_run = ascii_run.saturating_add(1);
        } else {
            if ascii_run > 0 {
                tokens = tokens.saturating_add(ascii_run.saturating_add(3) / 4);
                ascii_run = 0;
            }
            tokens = tokens.saturating_add(1);
        }
    }

    if ascii_run > 0 {
        tokens = tokens.saturating_add(ascii_run.saturating_add(3) / 4);
    }

    tokens.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn emptyish_body_returns_at_least_one() {
        assert!(estimate_anthropic_count_tokens(&json!({})) >= 1);
    }

    #[test]
    fn counts_plain_text_message() {
        let body = json!({
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "Hello world"}
            ]
        });
        let tokens = estimate_anthropic_count_tokens(&body);
        // "Hello world" ≈ 3 ascii tokens + framing
        assert!(tokens >= 4, "tokens={tokens}");
        assert!(tokens < 40, "tokens={tokens}");
    }

    #[test]
    fn cjk_costs_more_per_char_than_ascii() {
        let ascii = estimate_text_tokens("abcd"); // 1
        let cjk = estimate_text_tokens("你好世界"); // 4
        assert_eq!(ascii, 1);
        assert_eq!(cjk, 4);
    }

    #[test]
    fn tools_add_nontrivial_tokens() {
        let bare = estimate_anthropic_count_tokens(&json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let with_tools = estimate_anthropic_count_tokens(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "Bash",
                "description": "Run a shell command in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "command to run"}
                    },
                    "required": ["command"]
                }
            }]
        }));
        assert!(
            with_tools > bare + 10,
            "bare={bare} with_tools={with_tools}"
        );
    }

    #[test]
    fn api_format_gate() {
        assert!(api_format_supports_upstream_count_tokens("anthropic"));
        assert!(!api_format_supports_upstream_count_tokens(
            "openai_responses"
        ));
        assert!(!api_format_supports_upstream_count_tokens("openai_chat"));
        assert!(!api_format_supports_upstream_count_tokens("gemini_native"));
    }
}

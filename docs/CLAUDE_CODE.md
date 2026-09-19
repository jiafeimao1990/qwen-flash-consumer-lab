# Claude Code validation

The 96K profile was tested behind a local Anthropic-Messages-to-OpenAI
compatibility gateway. This document records the context policy and measured
behavior; it is not a claim that stock Claude Code can talk directly to an
OpenAI-compatible endpoint.

## Context policy

The Claude Code process used these environment values:

```text
CLAUDE_CODE_MAX_CONTEXT_TOKENS=98304
CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=72
```

The 72% compaction threshold deliberately leaves room for tool output and the
next assistant turn before the backend's hard 98,304-token limit. Claude Code
reads these values at process startup, so an already-running session must be
closed and reopened after changing them.

The compatibility gateway also normalized Claude Code's message shape for the
Qwen chat template and kept volatile reminder text out of the stable leading
system prefix. This was important for prefix-cache reuse. Do not infer that the
three environment values alone are a complete compatibility layer.

## Measured behavior

- Warm long tool turns produced 33.5-41.8 output tok/s, with a 38.8 tok/s
  weighted wall-rate mean.
- One 68,635-token-input turn reused 97.7% of the prefix and produced 5,745
  tokens in 145 seconds, or 39.6 output tok/s by wall time.
- A post-compaction cold turn had 48,730 input tokens, no prefix-cache hit and
  produced 1,340 tokens in 89.6 seconds. Its apparent end-to-end rate was about
  15 output tok/s because cold prefill dominated; this was not a 15 tok/s
  decode result.
- Prefix reuse in normal warm turns was generally 96-99%.

These numbers explain why a fixed backend benchmark near 48 tok/s does not
translate to a constant 48 tok/s interactive-agent experience. Tool latency,
cold prefill, compaction, gateway work and smaller generations all affect the
wall-clock rate.

## Long-session lesson

Auto-compaction prevents a single request from crossing the backend limit, but
it does not make an indefinitely long autonomous session free. Repeated
summaries, tool schemas, reminders and task state still consume context. For
multi-hour jobs, persist progress outside the conversation and resume in a
fresh session at bounded checkpoints.

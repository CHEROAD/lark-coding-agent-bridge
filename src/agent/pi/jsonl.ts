import type { AgentEvent } from '../types';

/**
 * Translate pi's `--mode json` stdout events into bridge `AgentEvent`s.
 *
 * pi emits a JSON object per line (see `toJsonEvent` in pi's json-event.ts):
 *  - `message_update` → streaming deltas (`assistantMessageEvent.type`):
 *      text_delta / thinking_delta / toolcall_* (tool call args)
 *  - `tool_execution_start` / `tool_execution_end` → tool lifecycle
 *  - `message_end` (assistant) → final answer text
 *  - `agent_end` → terminal event (normal / error)
 *  - `turn_end` → end of a turn (carries the authoritative assistant message)
 */
export class PiJsonlTranslator {
  private toolCallNames = new Map<string, string>();
  private finalTextEmitted = false;
  private terminalEmitted = false;

  /** True once a terminal event (`done` / `error`) has been emitted. */
  isTerminalEmitted(): boolean {
    return this.terminalEmitted;
  }

  /** Translate one parsed JSON line into zero or more bridge events. */
  *translate(line: unknown): Generator<AgentEvent> {
    if (!line || typeof line !== 'object') return;
    const obj = line as Record<string, unknown>;
    switch (obj.type) {
      case 'message_update': {
        const usage = obj.usage as Record<string, number> | undefined;
        if (usage) yield* this.usageEvent(usage);
        const evt = obj.assistantMessageEvent as Record<string, unknown> | undefined;
        if (evt) yield* this.assistantMessageEvent(evt);
        return;
      }
      case 'tool_execution_start': {
        const id = asString(obj.toolCallId) ?? asString(obj.id);
        const name = asString(obj.toolName);
        if (id && name) {
          this.toolCallNames.set(id, name);
          yield { type: 'tool_use', id, name, input: obj.args };
        }
        return;
      }
      case 'tool_execution_end': {
        const id = asString(obj.toolCallId) ?? asString(obj.id);
        const name = asString(obj.toolName) ?? (id ? this.toolCallNames.get(id) : undefined);
        if (id) {
          yield {
            type: 'tool_result',
            id,
            output: extractToolResultOutput(obj.result),
            isError: obj.isError === true,
          };
        }
        return;
      }
      case 'message_end': {
        yield* this.messageEnd(obj);
        return;
      }
      case 'turn_end': {
        yield* this.messageEnd(obj);
        return;
      }
      case 'agent_end': {
        yield* this.agentEnd(obj);
        return;
      }
      default:
        return;
    }
  }

  private *assistantMessageEvent(evt: Record<string, unknown>): Generator<AgentEvent> {
    switch (evt.type) {
      case 'text_delta':
        yield { type: 'text', delta: asString(evt.delta) ?? '' };
        return;
      case 'thinking_delta':
        yield { type: 'thinking', delta: asString(evt.delta) ?? '' };
        return;
      case 'toolcall_start': {
        const id = asString(evt.id);
        const name = asString(evt.toolName);
        if (id && name) this.toolCallNames.set(id, name);
        // The authoritative tool invocation arrives as `tool_execution_start`;
        // `toolcall_start` only streams the model's raw args, so emit nothing here.
        return;
      }
      case 'toolcall_end': {
        const toolCall = evt.toolCall as Record<string, unknown> | undefined;
        const id = asString(toolCall?.id) ?? asString(evt.id);
        const name = asString(toolCall?.name) ?? (id ? this.toolCallNames.get(id) : undefined);
        if (id && name) this.toolCallNames.set(id, name);
        return;
      }
      case 'toolcall_delta': {
        // Arguments stream in as raw JSON string deltas; too noisy to forward
        // every delta. The authoritative tool input arrives at toolcall_end
        // (or tool_execution_start). Silently accumulate nothing here.
        return;
      }
      default:
        return;
    }
  }

  private *messageEnd(obj: Record<string, unknown>): Generator<AgentEvent> {
    const message = obj.message as Record<string, unknown> | undefined;
    if (message?.role !== 'assistant') return;
    if (this.finalTextEmitted) return;
    const content = message.content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== 'object') continue;
        const b = block as Record<string, unknown>;
        if (b.type === 'text' && typeof b.text === 'string' && b.text.length > 0) {
          this.finalTextEmitted = true;
          yield { type: 'final_text', content: b.text };
        }
      }
    }
  }

  private *agentEnd(obj: Record<string, unknown>): Generator<AgentEvent> {
    if (this.terminalEmitted) return;
    if (obj.willRetry === true) return;
    this.terminalEmitted = true;
    // Inspect the last message's stopReason to distinguish error/aborted runs.
    const messages = obj.messages;
    if (Array.isArray(messages) && messages.length > 0) {
      const last = messages[messages.length - 1] as Record<string, unknown> | undefined;
      const stopReason = asString(last?.stopReason);
      if (stopReason === 'error' || stopReason === 'aborted') {
        yield {
          type: 'error',
          message: asString(last?.errorMessage) ?? `pi run ${stopReason}`,
          terminationReason: 'failed',
        };
        return;
      }
    }
    yield { type: 'done', terminationReason: 'normal' };
  }

  private *usageEvent(usage: Record<string, number>): Generator<AgentEvent> {
    const num = (v: unknown): number | undefined =>
      typeof v === 'number' && Number.isFinite(v) ? v : undefined;
    yield {
      type: 'usage',
      inputTokens: num(usage.input),
      outputTokens: num(usage.output),
      cachedInputTokens: num(usage.cacheRead),
      reasoningOutputTokens: num(usage.reasoning),
    };
  }
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

/**
 * pi reports tool results as a nested `{ content: [{ type: 'text', text }] }`
 * structure. Flatten text blocks into a single string; fall back to the raw
 * string or a JSON dump for non-text results (images, structured data, etc.).
 */
function extractToolResultOutput(result: unknown): string {
  const direct = asString(result);
  if (direct !== undefined) return direct;
  if (result && typeof result === 'object') {
    const obj = result as Record<string, unknown>;
    if (Array.isArray(obj.content)) {
      const text = obj.content
        .filter((block): block is Record<string, unknown> => !!block && typeof block === 'object')
        .map((block) => (block.type === 'text' ? asString(block.text) : undefined))
        .filter((t): t is string => t !== undefined)
        .join('');
      if (text) return text;
    }
  }
  return JSON.stringify(result ?? '');
}

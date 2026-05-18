import { apiClient } from './api';

export type ScriptArgs = (string | number | boolean)[] | Record<string, string | number | boolean | null | undefined>;

export interface ScriptResult {
  ok: boolean;
  exitCode: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
  result: unknown;
  timedOut: boolean;
  error?: string;
}

export interface StreamEvent {
  type: 'stdout' | 'stderr' | 'exit' | 'error';
  data?: string;
  message?: string;
  exitCode?: number | null;
  signal?: string | null;
  ok?: boolean;
  timedOut?: boolean;
  result?: unknown;
  stderr?: string;
}

export interface CallScriptOptions {
  args?: ScriptArgs;
  signal?: AbortSignal;
  // Match the API's 20-minute ceiling by default so axios doesn't bail early.
  timeoutMs?: number;
}

export interface StreamCallScriptOptions extends CallScriptOptions {
  onEvent?: (event: StreamEvent) => void;
  onStdout?: (chunk: string) => void;
  onStderr?: (chunk: string) => void;
}

const DEFAULT_TIMEOUT_MS = 20 * 60 * 1000;

// Buffered call: resolves with full stdout/stderr after the script exits.
export const callScript = async (
  script: string,
  options: CallScriptOptions = {},
): Promise<ScriptResult> => {
  const response = await apiClient.post<ScriptResult>(
    '/api/scripts',
    { script, args: options.args },
    {
      timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      signal: options.signal,
      // Don't throw on non-2xx so callers can inspect the structured failure.
      validateStatus: () => true,
    },
  );
  return response.data;
};

// Streaming call: invokes the callbacks for each NDJSON event as it arrives,
// then resolves with the final exit event.
export const callScriptStream = async (
  script: string,
  options: StreamCallScriptOptions = {},
): Promise<StreamEvent | null> => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (options.signal) {
    if (options.signal.aborted) controller.abort();
    else options.signal.addEventListener('abort', onAbort);
  }

  const timeout = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeout);

  let finalEvent: StreamEvent | null = null;

  try {
    const response = await fetch('/api/scripts', {
      method: 'POST',
      headers,
      body: JSON.stringify({ script, args: options.args, stream: true }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      const text = await response.text().catch(() => '');
      throw new Error(`Script stream failed: HTTP ${response.status} ${text}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (!line) continue;
        try {
          const event = JSON.parse(line) as StreamEvent;
          options.onEvent?.(event);
          if (event.type === 'stdout' && event.data) options.onStdout?.(event.data);
          if (event.type === 'stderr' && event.data) options.onStderr?.(event.data);
          if (event.type === 'exit' || event.type === 'error') finalEvent = event;
        } catch {
          // Ignore malformed lines and keep streaming.
        }
      }
    }
  } finally {
    clearTimeout(timer);
    if (options.signal) options.signal.removeEventListener('abort', onAbort);
  }

  return finalEvent;
};

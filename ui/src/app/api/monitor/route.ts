import { startMonitor } from '@/server/monitor';
import { MonitorSample } from '@/types';

// SSE stream of system stats. On connect: an `init` event with the 2-minute
// rolling history plus the latest full sample; then a `sample` event every
// MONITOR_TICK_MS. Auth is the normal middleware bearer check, so the client
// uses fetch (EventSource can't send headers).
export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const monitor = startMonitor();
  const encoder = new TextEncoder();
  let unsubscribe: (() => void) | null = null;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      };
      const cleanup = () => {
        unsubscribe?.();
        unsubscribe = null;
        try {
          controller.close();
        } catch {
          // already closed
        }
      };
      try {
        send('init', monitor.getInit());
      } catch {
        cleanup();
        return;
      }
      unsubscribe = monitor.subscribe((sample: MonitorSample, serialized: string) => {
        // desiredSize goes ever more negative when the consumer stopped
        // reading. If the abort for a vanished client never propagates (dev
        // HMR reloads, proxy quirks), this is the backstop that keeps dead
        // subscribers from accumulating and eating CPU forever.
        if (controller.desiredSize !== null && controller.desiredSize < -120) {
          cleanup();
          return;
        }
        try {
          controller.enqueue(encoder.encode(`event: sample\ndata: ${serialized}\n\n`));
        } catch {
          // Client is gone but abort hasn't fired yet
          cleanup();
        }
      });
      request.signal.addEventListener('abort', cleanup);
    },
    cancel() {
      unsubscribe?.();
      unsubscribe = null;
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      // Belt-and-braces against any buffering proxy in front of us
      'X-Accel-Buffering': 'no',
    },
  });
}

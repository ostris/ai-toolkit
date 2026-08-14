/**
 * Next.js instrumentation hook — runs once when the server process starts
 * (not during build). Starts the always-on system monitor so the rolling
 * stats history is already populated when the first client connects.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { startMonitor } = await import('@/server/monitor');
    startMonitor();
  }
}

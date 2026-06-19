import { NextResponse } from 'next/server';

/**
 * Restart the Next.js server. Works in production because the launcher
 * (`npm run start`) wraps it with `concurrently --restart-tries -1
 * --restart-after 1000`, which re-spawns the process within ~1 second
 * after we exit. The dev server (`npm run dev`) does NOT auto-restart,
 * so this endpoint is only meaningful in the production-style launch
 * used by Start-AI-Toolkit.bat.
 */
export async function POST() {
  // Schedule the exit AFTER we return the response so the client gets
  // the OK and can start polling for the server to come back online.
  setTimeout(() => {
    console.log('[restart] User requested restart. Exiting process so the supervisor relaunches it.');
    process.exit(0);
  }, 400);
  return NextResponse.json({ ok: true });
}

// Allow GET too for easy curl/manual testing.
export const GET = POST;

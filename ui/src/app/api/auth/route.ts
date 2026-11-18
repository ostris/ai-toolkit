import { NextResponse } from 'next/server';
import { getReadKeys, isReadProtectionDisabled, secondsUntilReadKeyRoll } from '@/server/readKey';

// the read key changes over time; never prerender this
export const dynamic = 'force-dynamic';

export async function GET() {
  // if this gets hit, auth has already been verified
  // no read key without a password or with read protection disabled; URLs then carry none
  if (!process.env.AI_TOOLKIT_AUTH || isReadProtectionDisabled()) {
    return NextResponse.json({ isAuthenticated: true });
  }
  const [readKey = ''] = await getReadKeys();
  return NextResponse.json({ isAuthenticated: true, readKey, readKeyRefreshIn: secondsUntilReadKeyRoll() });
}

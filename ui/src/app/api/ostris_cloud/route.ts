import { NextResponse } from 'next/server';

export async function GET() {
  const appUrl = process.env.OSTRIS_CLOUD_APP_URL;
  const apiKey = process.env.OSTRIS_CLOUD_API_KEY;

  if (!appUrl || !apiKey) {
    return NextResponse.json({ enabled: false });
  }

  try {
    const res = await fetch(`${appUrl}/api/machine/me`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: 'no-store',
    });

    if (!res.ok) {
      return NextResponse.json({
        enabled: true,
        appUrl,
        error: `Ostris Cloud responded with ${res.status}`,
      });
    }

    const data = await res.json();
    return NextResponse.json({
      enabled: true,
      appUrl,
      balance: data.balance ?? null,
    });
  } catch (error) {
    return NextResponse.json({
      enabled: true,
      appUrl,
      error: `Failed to fetch Ostris Cloud balance: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
}

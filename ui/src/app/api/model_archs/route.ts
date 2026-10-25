import { NextResponse } from 'next/server';
import { listExtensionUiModules } from '@/server/extensionUi';

// Extension UI modules (AI_TOOLKIT_UI_MODELS) as transpiled CommonJS source,
// evaluated client-side by useModelArchs. Read from disk per request so new
// models appear without a rebuild.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const result = await listExtensionUiModules();
    return NextResponse.json(result, { headers: { 'Cache-Control': 'no-store' } });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 });
  }
}

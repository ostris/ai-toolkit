import { NextResponse } from 'next/server';

export async function GET() {
  // if this gets hit, auth has already been verified
  return NextResponse.json({ isAuthenticated: true });
}

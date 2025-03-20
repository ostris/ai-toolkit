// middleware.ts (at the root of your project)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// if route starts with these, approve
const publicRoutes = ['/api/img/', '/api/files/'];

export function middleware(request: NextRequest) {
  // check env var for AI_TOOLKIT_AUTH, if not set, approve all requests
  // if it is set make sure bearer token matches
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) {
    return NextResponse.next();
  }

  // Get the token from the headers
  const token = request.headers.get('Authorization')?.split(' ')[1];

  // allow public routes to pass through
  if (publicRoutes.some(route => request.nextUrl.pathname.startsWith(route))) {
    return NextResponse.next();
  }

  // Check if the route should be protected
  // This will apply to all API routes that start with /api/
  if (request.nextUrl.pathname.startsWith('/api/')) {
    if (!token || token !== tokenToUse) {
      // Return a JSON response with 401 Unauthorized
      return new NextResponse(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // For authorized users, continue
    return NextResponse.next();
  }

  // For non-API routes, just continue
  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};

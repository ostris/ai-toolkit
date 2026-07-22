import { PrismaClient } from '@prisma/client';

// Single shared PrismaClient for the whole server. Route files must import
// this instead of constructing their own client — each PrismaClient spawns
// its own query engine + connection pool, and a pile of them contending for
// the same sqlite file just adds lock pressure. The globalThis stash keeps
// dev hot-reload from leaking a new client on every recompile.
const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient };

const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

export default prisma;

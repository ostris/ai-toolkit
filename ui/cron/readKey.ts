// Rolling read-only key for media/file GETs, which cannot send the bearer token.
// src/server/readKey.ts must derive identical keys.
import crypto from 'crypto';
import type http from 'http';

const WINDOW_SECONDS = 12 * 60 * 60;

let cache: { windowIndex: number; secret: string; keys: string[] } | null = null;

// [current, previous]; empty without a launch secret, so nothing validates
const getReadKeys = (): string[] => {
  const secret = process.env.AI_TOOLKIT_READ_SECRET;
  if (!secret) return [];
  const windowIndex = Math.floor(Date.now() / 1000 / WINDOW_SECONDS);
  if (cache && cache.windowIndex === windowIndex && cache.secret === secret) return cache.keys;
  const sign = (index: number) =>
    crypto.createHmac('sha256', secret).update(`aitk-read-key:${index}`).digest('hex').slice(0, 32);
  const keys = [sign(windowIndex), sign(windowIndex - 1)];
  cache = { windowIndex, secret, keys };
  return keys;
};

const safeEqual = (a: string, b: string) => {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  return bufA.length === bufB.length && crypto.timingSafeEqual(bufA, bufB);
};

const isReadProtectionDisabled = () =>
  ['1', 'true'].includes((process.env.AI_TOOLKIT_DISABLE_READ_PROTECTION || '').toLowerCase());

// open when no password is set or protection is disabled; otherwise the bearer token or a live read key
export const isReadAuthorized = (req: http.IncomingMessage): boolean => {
  const password = process.env.AI_TOOLKIT_AUTH;
  if (!password || isReadProtectionDisabled()) return true;
  const bearer = req.headers.authorization?.split(' ')[1];
  if (bearer && safeEqual(bearer, password)) return true;
  let key: string | null = null;
  try {
    key = new URL(req.url || '', 'http://localhost').searchParams.get('k');
  } catch {
    return false;
  }
  return !!key && getReadKeys().some(readKey => safeEqual(key, readKey));
};

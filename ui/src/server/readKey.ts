// Rolling read-only key for media/file GETs, which cannot send the bearer token.
// Web Crypto only: the edge-runtime middleware imports this.
// cron/readKey.ts must derive identical keys.
export const READ_KEY_PARAM = 'k';
const WINDOW_SECONDS = 12 * 60 * 60;

// opt out: reads under the public routes need no key or token, as before read keys existed
export const isReadProtectionDisabled = () =>
  ['1', 'true'].includes((process.env.AI_TOOLKIT_DISABLE_READ_PROTECTION || '').toLowerCase());

let cache: { windowIndex: number; secret: string; keys: string[] } | null = null;

const sign = async (cryptoKey: CryptoKey, windowIndex: number) => {
  const message = new TextEncoder().encode(`aitk-read-key:${windowIndex}`);
  const signature = await crypto.subtle.sign('HMAC', cryptoKey, message);
  return Array.from(new Uint8Array(signature), byte => byte.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 32);
};

// [current, previous]; empty without a launch secret, so nothing validates
export const getReadKeys = async (): Promise<string[]> => {
  const secret = process.env.AI_TOOLKIT_READ_SECRET;
  if (!secret) return [];
  const windowIndex = Math.floor(Date.now() / 1000 / WINDOW_SECONDS);
  if (cache && cache.windowIndex === windowIndex && cache.secret === secret) return cache.keys;
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const keys = [await sign(cryptoKey, windowIndex), await sign(cryptoKey, windowIndex - 1)];
  cache = { windowIndex, secret, keys };
  return keys;
};

const safeEqual = (a: string, b: string) => {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
};

export const isValidReadKey = async (key: string | null | undefined) => {
  if (!key) return false;
  let valid = false;
  for (const readKey of await getReadKeys()) {
    if (safeEqual(key, readKey)) valid = true;
  }
  return valid;
};

export const secondsUntilReadKeyRoll = () => WINDOW_SECONDS - (Math.floor(Date.now() / 1000) % WINDOW_SECONDS);

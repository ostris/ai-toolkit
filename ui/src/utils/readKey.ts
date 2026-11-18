// Read key for /api/img, /api/files and /api/audio/art URLs when a password is set.
// Empty otherwise. AuthWrapper sets it from /api/auth.
let readKey = '';

export const setReadKey = (key: string) => {
  readKey = key || '';
};

// for URLs built without encodeFilePathForUrl, which appends the key itself
export const withReadKey = (url: string) => {
  if (!readKey) return url;
  return `${url}${url.includes('?') ? '&' : '?'}k=${readKey}`;
};

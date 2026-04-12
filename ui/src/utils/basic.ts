export const objectCopy = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj)) as T;
};

export const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const imgExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'];
export const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
export const audioExtensions = ['.mp3', '.wav', '.flac', '.ogg'];

export const isVideo = (filePath: string) => videoExtensions.includes(filePath.toLowerCase().slice(-4));
export const isImage = (filePath: string) => imgExtensions.includes(filePath.toLowerCase().slice(-4));
export const isAudio = (filePath: string) => audioExtensions.includes(filePath.toLowerCase().slice(-4));

export const tagsToObj = (tagStr: string): Record<string, any> => {
  const result: Record<string, any> = {};
  const regex = /<([A-Z_][A-Z0-9_]*)>([\s\S]*?)<\/\1>/g;
  let match;
  while ((match = regex.exec(tagStr)) !== null) {
    const value = match[2].trim();
    try {
      result[match[1]] = JSON.parse(value);
    } catch {
      result[match[1]] = value;
    }
  }
  return result;
};

export const objToTags = (obj: Record<string, any>): string => {
  return Object.entries(obj)
    .map(([key, value]) => {
      const content = typeof value === 'string' ? value : JSON.stringify(value);
      return `<${key}>${content}</${key}>`;
    })
    .join('\n');
};

export const pathJoin = (...parts: string[]) => {
  const sep = parts.length > 0 && parts[0].includes('\\') ? '\\' : '/';
  const leadingTrailing = sep === '\\' ? /^\\+|\\+$/g : /^\/+|\/+$/g;
  const trailing = sep === '\\' ? /\\+$/ : /\/+$/;
  return parts
    .map((part, index) => {
      if (index === 0) {
        return part.replace(trailing, '');
      } else {
        return part.replace(leadingTrailing, '');
      }
    })
    .filter(part => part.length > 0)
    .join(sep);
}
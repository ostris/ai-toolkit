export const objectCopy = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj)) as T;
};

export const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const imgExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'];
export const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
export const audioExtensions = ['.mp3', '.wav'];

export const isVideo = (filePath: string) => videoExtensions.includes(filePath.toLowerCase().slice(-4));
export const isImage = (filePath: string) => imgExtensions.includes(filePath.toLowerCase().slice(-4));
export const isAudio = (filePath: string) => audioExtensions.includes(filePath.toLowerCase().slice(-4));

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

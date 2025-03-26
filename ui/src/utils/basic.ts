export const objectCopy = <T>(obj: T): T => {
  return JSON.parse(JSON.stringify(obj)) as T;
};

export const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const imgExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'];
export const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];

export const isVideo = (filePath: string) => videoExtensions.includes(filePath.toLowerCase().slice(-4));
export const isImage = (filePath: string) => imgExtensions.includes(filePath.toLowerCase().slice(-4));

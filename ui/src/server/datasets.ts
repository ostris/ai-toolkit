import fs from 'fs';
import path from 'path';

const datasetItemExtensions = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
]);

export function isDatasetItem(filename: string) {
  return datasetItemExtensions.has(path.extname(filename).toLowerCase());
}

export function findDatasetItemsRecursively(dir: string): string[] {
  let results: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;

    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      results = results.concat(findDatasetItemsRecursively(itemPath));
    } else if (entry.isFile() && isDatasetItem(name)) {
      results.push(itemPath);
    }
  }

  return results;
}

export function countDatasetItemsRecursively(dir: string): number {
  let count = 0;
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;

    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      count += countDatasetItemsRecursively(itemPath);
    } else if (entry.isFile() && isDatasetItem(name)) {
      count += 1;
    }
  }

  return count;
}

export function isPathInRoot(filePath: string, rootPath: string) {
  const resolvedPath = path.resolve(filePath);
  const resolvedRoot = path.resolve(rootPath);

  if (process.platform === 'win32') {
    const lowerPath = resolvedPath.toLowerCase();
    const lowerRoot = resolvedRoot.toLowerCase();
    return lowerPath === lowerRoot || lowerPath.startsWith(lowerRoot + path.sep);
  }

  return resolvedPath === resolvedRoot || resolvedPath.startsWith(resolvedRoot + path.sep);
}

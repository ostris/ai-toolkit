import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

interface CheckpointInfo {
  path: string;
  exists: boolean;
  format: 'safetensors' | 'pt' | 'directory' | 'unknown';
  size: number;
}

export async function getLatestCheckpoint(jobOutputDir: string): Promise<CheckpointInfo | null> {
  try {
    if (!fs.existsSync(jobOutputDir)) {
      return null;
    }

    const files = fs.readdirSync(jobOutputDir);
    let latestSafetensors: { file: string; mtime: number } | null = null;
    let latestPt: { file: string; mtime: number } | null = null;

    // Look for all safetensors and .pt files, find the newest
    for (const file of files) {
      if (file.endsWith('.safetensors')) {
        const fullPath = path.join(jobOutputDir, file);
        try {
          const stats = fs.statSync(fullPath);
          if (!stats.isDirectory()) {
            const mtime = stats.mtimeMs;
            if (!latestSafetensors || mtime > latestSafetensors.mtime) {
              latestSafetensors = { file, mtime };
            }
          }
        } catch (e) {
          // Skip files we can't stat
        }
      } else if (file.endsWith('.pt')) {
        const fullPath = path.join(jobOutputDir, file);
        try {
          const stats = fs.statSync(fullPath);
          if (!stats.isDirectory()) {
            const mtime = stats.mtimeMs;
            if (!latestPt || mtime > latestPt.mtime) {
              latestPt = { file, mtime };
            }
          }
        } catch (e) {
          // Skip files we can't stat
        }
      }
    }

    // Return latest safetensors if available (preferred format)
    if (latestSafetensors) {
      const fullPath = path.join(jobOutputDir, latestSafetensors.file);
      const stats = fs.statSync(fullPath);
      return {
        path: fullPath,
        exists: true,
        format: 'safetensors',
        size: stats.size,
      };
    }

    // Fallback to latest .pt file
    if (latestPt) {
      const fullPath = path.join(jobOutputDir, latestPt.file);
      const stats = fs.statSync(fullPath);
      return {
        path: fullPath,
        exists: true,
        format: 'pt',
        size: stats.size,
      };
    }

    // Look for checkpoint directories as last resort
    for (const file of files) {
      if (file.includes('checkpoint')) {
        const fullPath = path.join(jobOutputDir, file);
        const stats = fs.statSync(fullPath);
        if (stats.isDirectory()) {
          return {
            path: fullPath,
            exists: true,
            format: 'directory',
            size: getDirectorySize(fullPath),
          };
        }
      }
    }

    return null;
  } catch (error) {
    console.error('Error getting latest checkpoint:', error);
    return null;
  }
}

export async function copyCheckpoint(
  sourceCheckpointPath: string,
  destJobOutputDir: string,
  newJobName?: string,
): Promise<boolean> {
  try {
    if (!fs.existsSync(sourceCheckpointPath)) {
      console.error('Source checkpoint does not exist:', sourceCheckpointPath);
      return false;
    }

    if (!fs.existsSync(destJobOutputDir)) {
      fs.mkdirSync(destJobOutputDir, { recursive: true });
    }

    const stats = fs.statSync(sourceCheckpointPath);

    if (stats.isDirectory()) {
      copyDirectoryRecursive(sourceCheckpointPath, destJobOutputDir);
    } else if (stats.isFile()) {
      const fileName = path.basename(sourceCheckpointPath);
      
      let destFileName = fileName;
      if (newJobName) {
        // Rename: "parent_name_000001000.safetensors" → "fork_name_000001000.safetensors"
        const nameMatch = fileName.match(/^(.+?)(_\d+)?(\.\w+)$/);
        if (nameMatch) {
          const stepSuffix = nameMatch[2] || '';
          const extension = nameMatch[3];
          destFileName = `${newJobName}${stepSuffix}${extension}`;
        }
      }
      
      const destPath = path.join(destJobOutputDir, destFileName);
      fs.copyFileSync(sourceCheckpointPath, destPath);
    }

    return true;
  } catch (error) {
    console.error('Error copying checkpoint:', error);
    return false;
  }
}

function copyDirectoryRecursive(source: string, dest: string): void {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const items = fs.readdirSync(source);
  for (const item of items) {
    const sourcePath = path.join(source, item);
    const destPath = path.join(dest, item);
    const stats = fs.statSync(sourcePath);

    if (stats.isDirectory()) {
      copyDirectoryRecursive(sourcePath, destPath);
    } else {
      fs.copyFileSync(sourcePath, destPath);
    }
  }
}

function getDirectorySize(dirPath: string): number {
  let size = 0;
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stats = fs.statSync(filePath);

    if (stats.isDirectory()) {
      size += getDirectorySize(filePath);
    } else {
      size += stats.size;
    }
  }

  return size;
}

export function getCheckpointFormattedSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

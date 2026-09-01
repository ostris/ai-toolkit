/**
 * Turn a `[...param]` catch-all route param into the requested filesystem path.
 *
 * Next.js hands catch-all params over as an array of URL segments with the
 * percent-encoding of each segment already decoded, so both URL forms map to
 * the same path:
 *   /api/files/%2Fmnt%2Fout%2Fjob%2Ffile.safetensors        (legacy: one segment)
 *   /api/files/%2Fmnt%2Fout%2Fjob/file.safetensors          (folder / filename)
 * The second form gives downloaders (wget, browsers) the real filename as the
 * last URL segment. Windows paths arrive with their backslashes intact inside
 * a segment (`C:\out\job` + `/` + `file.safetensors`); path.resolve on the
 * caller side normalizes the mixed separators.
 *
 * Mirrors the per-segment decode in cron/fileServer.ts serveFile().
 */
export const catchAllToFilePath = (param: string | string[] | undefined): string => {
  if (param === undefined) return '';
  return (Array.isArray(param) ? param : [param]).join('/');
};

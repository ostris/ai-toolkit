type MacstatsModule = {
  getGpuDataSync?: () => { temperature?: number; usage?: number };
  getFanDataSync?: () => Record<string, { rpm?: number }>;
  getPowerDataSync?: () => { gpu?: number };
  getRAMUsageSync?: () => { used: number; total: number; free: number };
  getCpuDataSync?: () => { temperature?: number };
};

let macstatsPromise: Promise<MacstatsModule | null> | undefined;

export async function getMacstats(): Promise<MacstatsModule | null> {
  if (!macstatsPromise) {
    macstatsPromise = (async () => {
      try {
        const mod = await import('macstats');
        return (mod.default ?? mod) as MacstatsModule;
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === 'MODULE_NOT_FOUND' || code === 'ERR_MODULE_NOT_FOUND') {
          return null;
        }
        throw error;
      }
    })();
  }

  return macstatsPromise;
}

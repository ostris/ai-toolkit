export const parseEnvBool = (value: string | undefined | null, defaultValue: boolean): boolean => {
  if (value === undefined || value === null) return defaultValue;
  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return defaultValue;
};

export const isLtxOnlyMode = (): boolean => {
  if (parseEnvBool(process.env.AITK_ALLOW_NON_LTX, false)) return false;
  return parseEnvBool(process.env.AITK_LTX_ONLY_MODE, true);
};

const isLtxModel = (model: any): boolean => {
  const arch = String(model?.arch ?? '').toLowerCase().trim();
  const nameOrPath = String(model?.name_or_path ?? '').toLowerCase().trim();
  return arch.startsWith('ltx2') || nameOrPath.includes('lightricks/ltx-2.3') || nameOrPath.includes('ltx-2.3');
};

export const isLtxJobConfig = (jobConfig: any): boolean => {
  const processList = jobConfig?.config?.process;
  if (!Array.isArray(processList)) return true;
  for (const processCfg of processList) {
    const type = String(processCfg?.type ?? '').toLowerCase().trim();
    if (!['sd_trainer', 'diffusion_trainer', 'ui_trainer'].includes(type)) {
      continue;
    }
    if (!isLtxModel(processCfg?.model)) {
      return false;
    }
  }
  return true;
};

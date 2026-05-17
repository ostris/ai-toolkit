import { modelArchs } from '@/app/jobs/new/options';

export function getVotingInputImageMode(arch: string | null | undefined): 'none' | 'single' | 'multi' {
  if (!arch) return 'none';
  const model = modelArchs.find(candidate => candidate.name === arch);
  if (!model?.additionalSections) return 'none';
  if (model.additionalSections.includes('sample.multi_ctrl_imgs')) return 'multi';
  if (model.additionalSections.includes('sample.ctrl_img')) return 'single';
  return 'none';
}


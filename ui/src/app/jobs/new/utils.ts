import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { modelArchs, ModelArch } from './options';
import { objectCopy } from '@/utils/basic';

const expandDatasetDefaults = (
  defaults: { [key: string]: any },
  numDatasets: number,
): { [key: string]: any } => {
  // expands the defaults for datasets[x] to datasets[0], datasets[1], etc.
  const expandedDefaults: { [key: string]: any } = { ...defaults };
  for (const key in defaults) {
    if (key.includes('datasets[x].')) {
      for (let i = 0; i < numDatasets; i++) {
        const datasetKey = key.replace('datasets[x].', `datasets[${i}].`);
        const v = defaults[key];
        expandedDefaults[datasetKey] = Array.isArray(v) ? [...v] : objectCopy(v);
      }
      delete expandedDefaults[key];
    }
  }
  return expandedDefaults;
};

export const handleModelArchChange = (
  currentArchName: string,
  newArchName: string,
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const currentArch = modelArchs.find(a => a.name === currentArchName);
  if (!currentArch || currentArch.name === newArchName) {
    return;
  }

  // update the defaults when a model is selected
  const newArch = modelArchs.find(model => model.name === newArchName);

  // update vram setting
  if (!newArch?.additionalSections?.includes('model.low_vram')) {
    setJobConfig(false, 'config.process[0].model.low_vram');
  }

  // handle layer offloading setting
  if (!newArch?.additionalSections?.includes('model.layer_offloading')) {
    if ('layer_offloading' in jobConfig.config.process[0].model) {
      const newModel = objectCopy(jobConfig.config.process[0].model);
      delete newModel.layer_offloading;
      delete newModel.layer_offloading_text_encoder_percent;
      delete newModel.layer_offloading_transformer_percent;
      setJobConfig(newModel, 'config.process[0].model');
    }
  } else {
    // set to false if not set
    if (!('layer_offloading' in jobConfig.config.process[0].model)) {
      setJobConfig(false, 'config.process[0].model.layer_offloading');
      setJobConfig(1.0, 'config.process[0].model.layer_offloading_text_encoder_percent');
      setJobConfig(1.0, 'config.process[0].model.layer_offloading_transformer_percent');
    }
  }

  const numDatasets = jobConfig.config.process[0].datasets.length;

  let currentDefaults = expandDatasetDefaults(currentArch.defaults || {}, numDatasets);
  let newDefaults = expandDatasetDefaults(newArch?.defaults || {}, numDatasets);

  // set new model
  setJobConfig(newArchName, 'config.process[0].model.arch');

  // update datasets
  const hasControlPath = newArch?.additionalSections?.includes('datasets.control_path') || false;
  const hasMultiControlPaths = newArch?.additionalSections?.includes('datasets.multi_control_paths') || false;
  const hasNumFrames = newArch?.additionalSections?.includes('datasets.num_frames') || false;
  const controls = newArch?.controls ?? [];
  const datasets = jobConfig.config.process[0].datasets.map(dataset => {
    const newDataset = objectCopy(dataset);
    newDataset.controls = controls;
    if (hasMultiControlPaths) {
      // make sure the config has the multi control paths
      newDataset.control_path_1 = newDataset.control_path_1 || null;
      newDataset.control_path_2 = newDataset.control_path_2 || null;
      newDataset.control_path_3 = newDataset.control_path_3 || null;
      // if we previously had a single control path and now
      // we selected a multi control model
      if (newDataset.control_path && newDataset.control_path !== '') {
        // only set if not overwriting
        if (!newDataset.control_path_1) {
          newDataset.control_path_1 = newDataset.control_path;
        }
      }
      delete newDataset.control_path; // remove single control path
    } else if (hasControlPath) {
      newDataset.control_path = newDataset.control_path || null;
      if (newDataset.control_path_1 && newDataset.control_path_1 !== '') {
        newDataset.control_path = newDataset.control_path_1;
      }
      if ('control_path_1' in newDataset) {
        delete newDataset.control_path_1;
      }
      if ('control_path_2' in newDataset) {
        delete newDataset.control_path_2;
      }
      if ('control_path_3' in newDataset) {
        delete newDataset.control_path_3;
      }
    } else {
      // does not have control images
      if ('control_path' in newDataset) {
        delete newDataset.control_path;
      }
      if ('control_path_1' in newDataset) {
        delete newDataset.control_path_1;
      }
      if ('control_path_2' in newDataset) {
        delete newDataset.control_path_2;
      }
      if ('control_path_3' in newDataset) {
        delete newDataset.control_path_3;
      }
    }
    if (!hasNumFrames) {
      newDataset.num_frames = 1; // reset num_frames if not applicable
    }
    return newDataset;
  });
  setJobConfig(datasets, 'config.process[0].datasets');

  // update samples
  const hasSampleCtrlImg = newArch?.additionalSections?.includes('sample.ctrl_img') || false;
  const samples = jobConfig.config.process[0].sample.samples.map(sample => {
    const newSample = objectCopy(sample);
    if (!hasSampleCtrlImg) {
      delete newSample.ctrl_img; // remove ctrl_img if not applicable
    }
    return newSample;
  });
  setJobConfig(samples, 'config.process[0].sample.samples');

  // revert defaults from previous model
  for (const key in currentDefaults) {
    setJobConfig(currentDefaults[key][1], key);
  }

  for (const key in newDefaults) {
    setJobConfig(newDefaults[key][0], key);
  }
};

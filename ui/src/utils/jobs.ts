import { JobConfig } from '@/types';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/start`)
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop`)
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};

export const duplicateJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs?id=${jobID}`)
      .then(res => res.data)
      .then(data => {
        const jobConfig = JSON.parse(data.job_config);
        const gpuIDs = data.gpu_ids;

        jobConfig.config.name = `${jobConfig.config.name}_copy`;

        apiClient
          .post('/api/jobs', {
            id: '',
            name: jobConfig.config.name,
            gpu_ids: gpuIDs,
            job_config: jobConfig,
          })
          .then(() => {
            console.log('Job duplicated');
            resolve();
          })
          .catch(error => {
            console.log('Error duplicating job:', error);
            reject(error);
          })
      })
      .catch(error => {
        console.log('Error duplicating job:', error);
        reject(error);
      })
  });
}

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/delete`)
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};

export const getJobConfig = (job: Job) => {
  return JSON.parse(job.job_config) as JobConfig;
};

export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  const isStopping = job.stop && job.status === 'running';
  const canDelete = ['completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canEdit = ['completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canStop = job.status === 'running' && !isStopping;
  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  if (job.status === 'completed' && jobConfig.config.process[0].train.steps > job.step && !isStopping) {
    canStart = true;
  }
  return { canDelete, canEdit, canStop, canStart };
};

export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].sample?.prompts?.length || 0;
};

export const getTotalSteps = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].train.steps;
};

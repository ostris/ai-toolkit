import { JobConfig } from '@/types';
import { Job } from '@prisma/client';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    fetch(`/api/jobs/${jobID}/start`)
      .then(res => res.json())
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
    fetch(`/api/jobs/${jobID}/stop`)
      .then(res => res.json())
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

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    fetch(`/api/jobs/${jobID}/delete`)
      .then(res => res.json())
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
}

export const getTotalSteps = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].train.steps;
}

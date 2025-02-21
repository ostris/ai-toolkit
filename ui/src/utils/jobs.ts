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

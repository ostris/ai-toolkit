import axios from 'axios';

interface ForkJobRequest {
  newJobName: string;
  gpuIds?: string;
  copyCheckpoint: boolean;
}

interface ForkJobResponse {
  success: boolean;
  jobId?: string;
  error?: string;
  message?: string;
}

export async function forkJob(
  parentJobId: string,
  request: ForkJobRequest,
): Promise<ForkJobResponse> {
  try {
    const response = await axios.post<ForkJobResponse>(
      `/api/jobs/${parentJobId}/fork`,
      request,
    );

    return response.data;
  } catch (error) {
    console.error('Error forking job:', error);

    if (axios.isAxiosError(error) && error.response?.data?.error) {
      return {
        success: false,
        error: error.response.data.error,
      };
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fork job',
    };
  }
}

export async function getJobWithForkInfo(jobId: string) {
  try {
    const response = await axios.get(`/api/jobs/${jobId}/fork-info`);
    return response.data;
  } catch (error) {
    console.error('Error fetching fork info:', error);
    return null;
  }
}

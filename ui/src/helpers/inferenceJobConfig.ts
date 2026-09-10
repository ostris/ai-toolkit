// Job config for the resident inference engine (extensions_built_in/inference_engine).
// The worker fills engine.job_folder in when it launches the job.
export const defaultInferenceJobConfig = {
  job: 'extension',
  config: {
    name: 'inference_engine',
    process: [
      {
        type: 'InferenceEngine',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        engine: {
          dtype: 'bf16',
        },
      },
    ],
  },
};

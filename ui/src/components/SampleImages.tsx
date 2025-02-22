import { useMemo } from 'react';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      return sampleConfig.prompts.length;
    }
    return 10;
  }, [job]);

  // Use direct Tailwind class without string interpolation
  // This way Tailwind can properly generate the class
  // I hate this, but it's the only way to make it work
  const gridColsClass = useMemo(() => {
    const cols = Math.min(numSamples, 20);

    switch (cols) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-2';
      case 3:
        return 'grid-cols-3';
      case 4:
        return 'grid-cols-4';
      case 5:
        return 'grid-cols-5';
      case 6:
        return 'grid-cols-6';
      case 7:
        return 'grid-cols-7';
      case 8:
        return 'grid-cols-8';
      case 9:
        return 'grid-cols-9';
      case 10:
        return 'grid-cols-10';
      case 11:
        return 'grid-cols-11';
      case 12:
        return 'grid-cols-12';
      case 13:
        return 'grid-cols-13';
      case 14:
        return 'grid-cols-14';
      case 15:
        return 'grid-cols-15';
      case 16:
        return 'grid-cols-16';
      case 17:
        return 'grid-cols-17';
      case 18:
        return 'grid-cols-18';
      case 19:
        return 'grid-cols-19';
      case 20:
        return 'grid-cols-20';
      default:
        return 'grid-cols-1';
    }
  }, [numSamples]);

  return (
    <div>
      <div className="pb-4">
        {status === 'loading' && sampleImages.length === 0 && <p>Loading...</p>}
        {status === 'error' && <p>Error fetching sample images</p>}
        {sampleImages && (
          <div className={`grid ${gridColsClass} gap-1`}>
            {sampleImages.map((sample: string) => (
              <SampleImageCard
                key={sample}
                imageUrl={sample}
                numSamples={numSamples}
                sampleImages={sampleImages}
                alt="Sample Image"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

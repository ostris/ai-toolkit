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
      if (sampleConfig.prompts) {
        return sampleConfig.prompts.length;
      } else {
        return sampleConfig.samples.length;
      }
    }
    return 10;
  }, [job]);

  // Use direct Tailwind class without string interpolation
  // This way Tailwind can properly generate the class
  // I hate this, but it's the only way to make it work
  const gridColsClass = useMemo(() => {
    const cols = Math.min(numSamples, 40);

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
      case 21:
        return 'grid-cols-21';
      case 22:
        return 'grid-cols-22';
      case 23:
        return 'grid-cols-23';
      case 24:
        return 'grid-cols-24';
      case 25:
        return 'grid-cols-25';
      case 26:
        return 'grid-cols-26';
      case 27:
        return 'grid-cols-27';
      case 28:
        return 'grid-cols-28';
      case 29:
        return 'grid-cols-29';
      case 30:
        return 'grid-cols-30';
      case 31:
        return 'grid-cols-31';
      case 32:
        return 'grid-cols-32';
      case 33:
        return 'grid-cols-33';
      case 34:
        return 'grid-cols-34';
      case 35:
        return 'grid-cols-35';
      case 36:
        return 'grid-cols-36';
      case 37:
        return 'grid-cols-37';
      case 38:
        return 'grid-cols-38';
      case 39:
        return 'grid-cols-39';
      case 40:
        return 'grid-cols-40';
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

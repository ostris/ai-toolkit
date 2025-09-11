import { useMemo, useState } from 'react';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import classNames from 'classnames';

interface SampleImagesMenuProps {
  job?: Job | null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeURIComponent(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <Button
      onClick={downloadZip}
      className={classNames(`px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700`, {
        'opacity-50 cursor-not-allowed': isZipping,
      })}
    >
      {isZipping ? (
        <LuLoader className="animate-spin inline-block mr-2" />
      ) : (
        <FaDownload className="inline-block mr-2" />
      )}
      {isZipping ? 'Preparing' : 'Download'}
    </Button>
  );
};

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

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

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
        {PageInfoContent}
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

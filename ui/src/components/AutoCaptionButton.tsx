import React, { use, useEffect } from 'react';
import { Button } from '@headlessui/react';
import { CaptionDatasetModal, openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useJobByRef from '@/hooks/useJobByRef';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';

type AutoCaptionButtonProps = {
  datasetPath: string;
  setIsAutoCaptioning?: (isAutoCaptioning: boolean) => void;
};

export default function AutoCaptionButton({ datasetPath, setIsAutoCaptioning }: AutoCaptionButtonProps) {
  const { job, status, refreshJob } = useJobByRef(datasetPath, 5000);
  useEffect(() => {
    if (setIsAutoCaptioning) {
      setIsAutoCaptioning(!!(job && job.status === 'running'));
    }
  }, [job, setIsAutoCaptioning]);

  if (job && (job.status === 'running' || job.status === 'queued')) {
    return (
      <Link
        href={`/jobs/${job.id}`}
        className="text-white bg-gray-400 px-2 sm:px-3 py-1 rounded-md mr-1 sm:mr-2 inline-flex items-center gap-1 sm:gap-1.5 text-sm sm:text-base whitespace-nowrap"
      >
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="hidden sm:inline">Auto Captioning...</span>
        <span className="sm:hidden">Captioning</span>
      </Link>
    );
  }
  return (
    <Button
      className="text-white bg-blue-600 px-2 sm:px-3 py-1 rounded-md mr-1 sm:mr-2 text-sm sm:text-base whitespace-nowrap"
      onClick={() =>
        openCaptionDatasetModal(datasetPath, () => {
          refreshJob();
        })
      }
    >
      <span className="hidden sm:inline">Auto Caption</span>
      <span className="sm:hidden">Caption</span>
    </Button>
  );
}

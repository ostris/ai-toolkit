import React, {use, useEffect} from 'react';
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
      setIsAutoCaptioning(!!(job && (job.status === 'running')));
    }
  }, [job, setIsAutoCaptioning]);
  
  if (job && (job.status === 'running' || job.status === 'queued')) {
    return (
      <Link href={`/jobs/${job.id}`} className="text-white bg-gray-400 px-3 py-1 rounded-md mr-2 inline-flex items-center gap-1.5">
        <Loader2 className="w-4 h-4 animate-spin" />
        Auto Captioning...
      </Link>
    );
  }
  return (
    <Button
      className="text-white bg-blue-600 px-3 py-1 rounded-md mr-2"
      onClick={() => openCaptionDatasetModal(datasetPath, () => {
        refreshJob();
      })}
    >
      Auto Caption
    </Button>
  );
}

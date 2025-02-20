'use client';

import { useEffect, useState, use } from 'react';
import Card from '@/components/Card';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import { useRouter } from 'next/router';
import DatasetImageCard from '@/components/DatasetImageCard';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    fetch('/api/datasets/listImages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ datasetName: dbName }),
    })
      .then(res => res.json())
      .then(data => {
        console.log('Images:', data.images);
        // sort
        data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);
  return (
    <>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold mb-8">Dataset: {datasetName}</h1>
          </div>
        </div>
        <Card title={`Images (${imgList.length})`}>
          {status === 'loading' && <p>Loading...</p>}
          {status === 'error' && <p>Error fetching images</p>}
          {status === 'success' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {imgList.length === 0 && <p>No images found</p>}
              {imgList.map(img => (
                <DatasetImageCard
                  key={img.img_path}
                  alt="image"
                  imageUrl={img.img_path}
                />
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

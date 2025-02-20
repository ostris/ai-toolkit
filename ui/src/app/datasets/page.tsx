'use client';

import { useEffect, useState } from 'react';
import Card from '@/components/Card';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';

export default function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshDatasets = () => {
    setStatus('loading');
    fetch('/api/datasets/list')
      .then(res => res.json())
      .then(data => {
        console.log('Datasets:', data);
        // sort
        data.sort((a: string, b: string) => a.localeCompare(b));
        setDatasets(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshDatasets();
  }, []);
  return (
    <>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold mb-8">Datasets</h1>
          </div>
          <div>
            <button
              onClick={() => {
                setIsNewDatasetModalOpen(true);
              }}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
              New Dataset
            </button>
          </div>
        </div>
        <Card title={`Datasets (${datasets.length})`}>
          {status === 'loading' && <p>Loading...</p>}
          {status === 'error' && <p>Error fetching datasets</p>}
          {status === 'success' && (
            <div className="space-y-1">
              {datasets.length === 0 && <p>No datasets found</p>}
              {datasets.map((dataset: string) => (
                <Link href={`/datasets/${dataset}`} className="bg-gray-800 hover:bg-gray-700 py-2 px-4 rounded-lg cursor-pointer block" key={dataset}>
                  {dataset}
                </Link>
              ))}
            </div>
          )}
        </Card>
      </div>
      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form
            onSubmit={e => {
              e.preventDefault();
              console.log('Creating new dataset');
              // make post with name to create new dataset
              fetch('/api/datasets/create', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: newDatasetName }),
              })
                .then(res => res.json())
                .then(data => {
                  console.log('New dataset created:', data);
                  refreshDatasets();
                  setNewDatasetName('');
                  setIsNewDatasetModalOpen(false);
                })
                .catch(error => {
                  console.error('Error creating new dataset:', error);
                });
            }}
          >
            <div className="text-sm text-gray-600">
              This will create a new folder with the name below in your dataset folder.
            </div>
            <div>
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                type="submit"
              >
                Confirm
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}

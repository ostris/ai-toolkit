'use client';

import { useEffect, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';

export default function Datasets() {
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => setIsNewDatasetModalOpen(true)}
          >
            New Dataset
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {status === 'loading' && <p>Loading...</p>}
        {status === 'error' && <p>Error fetching datasets</p>}
        {status === 'success' && (
          <div className="space-y-1">
            {datasets.length === 0 && <p>No datasets found</p>}
            {datasets.map((dataset: string) => (
              <div className="flex justify-between bg-gray-900 hover:bg-gray-800 transition-colors" key={dataset}>
                <div>
                  <Link href={`/datasets/${dataset}`} className="cursor-pointer block px-4 py-2" key={dataset}>
                    {dataset}
                  </Link>
                </div>
                <div className="flex-1"></div>
                <div>
                  <button
                    className="text-gray-200 hover:bg-red-600 px-2 py-2 mt-1 mr-1 rounded-full transition-colors"
                    onClick={() => {
                      openConfirm({
                        title: 'Delete Dataset',
                        message: `Are you sure you want to delete the dataset "${dataset}"? This action cannot be undone.`,
                        type: 'warning',
                        confirmText: 'Delete',
                        onConfirm: () => {
                          fetch('/api/datasets/delete', {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ name: dataset }),
                          })
                            .then(res => res.json())
                            .then(data => {
                              console.log('Dataset deleted:', data);
                              refreshDatasets();
                            })
                            .catch(error => {
                              console.error('Error deleting dataset:', error);
                            });
                        },
                      });
                    }}
                  >
                    <FaRegTrashAlt />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </MainContent>
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

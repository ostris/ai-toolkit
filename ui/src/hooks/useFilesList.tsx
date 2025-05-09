'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';

interface FileObject {
  path: string;
  size: number;
}

export default function useFilesList(jobID: string, reloadInterval: null | number = null) {
  const [files, setFiles] = useState<FileObject[]>([]);
  const didInitialLoadRef = useRef(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refreshFiles = () => {
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    apiClient
      .get(`/api/jobs/${jobID}/files`)
      .then(res => res.data)
      .then(data => {
        console.log('Fetched files:', data);
        if (data.files) {
          setFiles(data.files);
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshFiles();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshFiles();
      }, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID]);

  return { files, setFiles, status, refreshFiles };
}

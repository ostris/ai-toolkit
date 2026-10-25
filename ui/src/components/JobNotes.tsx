'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

interface Props {
  job: Job;
}

const AUTOSAVE_INTERVAL = 5000;

export default function JobNotes({ job }: Props) {
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  // refs so the unmount flush and interval always see the latest values
  const contentRef = useRef('');
  const savedRef = useRef('');
  const isSavingRef = useRef(false);
  const loadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    apiClient
      .get(`/api/jobs/${job.id}/notes`)
      .then(res => {
        if (cancelled) return;
        const loaded = res.data?.content ?? '';
        contentRef.current = loaded;
        savedRef.current = loaded;
        loadedRef.current = true;
        setContent(loaded);
        setIsLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  const save = useCallback(async () => {
    if (!loadedRef.current || isSavingRef.current) return;
    const toSave = contentRef.current;
    if (toSave === savedRef.current) return;
    isSavingRef.current = true;
    try {
      await apiClient.post(`/api/jobs/${job.id}/notes`, { content: toSave });
      savedRef.current = toSave;
    } catch (error) {
      console.error('Error saving notes:', error);
    } finally {
      isSavingRef.current = false;
    }
  }, [job.id]);

  // periodic autosave
  useEffect(() => {
    const interval = setInterval(() => {
      save();
    }, AUTOSAVE_INTERVAL);
    return () => clearInterval(interval);
  }, [save]);

  // flush on unmount (tab change / navigation) and on page unload. keepalive so
  // the request survives the component/page going away before it completes.
  useEffect(() => {
    const flush = () => {
      if (!loadedRef.current || contentRef.current === savedRef.current) return;
      const token = localStorage.getItem('AI_TOOLKIT_AUTH');
      fetch(`/api/jobs/${job.id}/notes`, {
        method: 'POST',
        keepalive: true,
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ content: contentRef.current }),
      }).catch(() => {});
      savedRef.current = contentRef.current;
    };
    window.addEventListener('beforeunload', flush);
    return () => {
      window.removeEventListener('beforeunload', flush);
      flush();
    };
  }, [job.id]);

  return (
    <textarea
      value={content}
      onChange={e => {
        contentRef.current = e.target.value;
        setContent(e.target.value);
      }}
      onBlur={() => save()}
      onKeyDown={e => {
        // ctrl/cmd+s saves immediately instead of hitting the browser save dialog
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
          e.preventDefault();
          save();
        }
      }}
      disabled={isLoading}
      spellCheck={false}
      placeholder="Notes for this job (markdown)..."
      className="block w-full h-full resize-none bg-gray-900 text-gray-100 font-mono text-sm p-3 outline-none disabled:opacity-50"
    />
  );
}

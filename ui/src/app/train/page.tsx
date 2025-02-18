'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { options } from './options';

interface TrainingData {
  modelConfig: {
    name_or_path: string;
    steps: number;
    batchSize: number;
    learningRate: number;
  };
}

const defaultTrainingData: TrainingData = {
  modelConfig: {
    name_or_path: 'ostris/Flex.1-alpha',
    steps: 100,
    batchSize: 32,
    learningRate: 0.001,
  },
};

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');

  const [name, setName] = useState('');
  const [trainingData, setTrainingData] = useState<TrainingData>(defaultTrainingData);
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  useEffect(() => {
    if (runId) {
      fetch(`/api/training?id=${runId}`)
        .then(res => res.json())
        .then(data => {
          setName(data.name);
          setTrainingData(JSON.parse(data.run_data));
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [runId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id: runId,
          name,
          run_data: trainingData,
        }),
      });

      if (!response.ok) throw new Error('Failed to save training');

      setStatus('success');
      if (!runId) {
        const data = await response.json();
        router.push(`/training?id=${data.id}`);
      }
      setTimeout(() => setStatus('idle'), 2000);
    } catch (error) {
      console.error('Error saving training:', error);
      setStatus('error');
      setTimeout(() => setStatus('idle'), 2000);
    }
  };

  const updateSection = (section: keyof TrainingData, data: any) => {
    setTrainingData(prev => ({
      ...prev,
      [section]: { ...prev[section], ...data },
    }));
  };

  const modelOptions = options.model.map(model => model.name_or_path);

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <h1 className="text-3xl font-bold mb-8">{runId ? 'Edit Training Run' : 'New Training Run'}</h1>

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="space-y-4">
          <label htmlFor="name" className="block text-sm font-medium mb-2">
            Training Name
          </label>
          <input
            type="text"
            id="name"
            value={name}
            onChange={e => setName(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
            placeholder="Enter training name"
            required
          />
        </div>

        {/* Model Configuration Section */}
        <section className="space-y-4 p-6 bg-gray-900 rounded-lg">
          <h2 className="text-xl font-bold mb-4">Model Configuration</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Model</label>
              <select
                value={trainingData.modelConfig.name_or_path}
                onChange={e => updateSection('modelConfig', { name_or_path: e.target.value })}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg"
              >
                {modelOptions.map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Epochs</label>
              <input
                type="number"
                value={trainingData.modelConfig.steps}
                onChange={e => updateSection('modelConfig', { steps: Number(e.target.value) })}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg"
              />
            </div>
          </div>
        </section>

        <button
          type="submit"
          disabled={status === 'saving'}
          className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {status === 'saving' ? 'Saving...' : runId ? 'Update Training' : 'Create Training'}
        </button>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
    </div>
  );
}

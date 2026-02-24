'use client';

import { useState, useRef, useCallback } from 'react';
import { MainContent, TopBar } from '@/components/layout';
import { Button } from '@headlessui/react';
import { TextInput, SliderInput } from '@/components/formInputs';
import { apiClient } from '@/utils/api';
import Card from '@/components/Card';

type MergeStatus = 'idle' | 'starting' | 'running' | 'completed' | 'error';

export default function MergeLoRAsPage() {
  const [lora1Path, setLora1Path] = useState('');
  const [lora2Path, setLora2Path] = useState('');
  const [outputName, setOutputName] = useState('merged_characters');
  const [dareDropRate, setDareDropRate] = useState(0.5);
        <div className="max-w-4xl mx-auto py-8 space-y-6">

          {status !== 'idle' && (
  const [status, setStatus] = useState<MergeStatus>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback((statusFile: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const res = await apiClient.get('/api/merge/status', { params: { statusFile } });
        const data = res.data;
        if (data.status === 'completed') {
          setStatus('completed');
          setStatusMessage('Merge complete! Output saved.');
          stopPolling();
        } else if (data.status === 'error') {
          setStatus('error');
          setStatusMessage(data.error || data.output || 'Merge failed. Check server logs.');
          stopPolling();
        }
      } catch {
        // keep polling
      }
    }, 3000);
  }, [stopPolling]);

  const handleMerge = async () => {
    if (!lora1Path || !lora2Path || !outputName) {
      alert('Please fill in both LoRA paths and an output name.');
      return;
    }

    setStatus('starting');
    setStatusMessage('Launching merge process...');
    setOutputPath('');

    try {
      const res = await apiClient.post('/api/merge/run', {
        lora_1_path: lora1Path,
        lora_2_path: lora2Path,
        output_name: outputName,
        dare_drop_rate: dareDropRate,
      });

      const data = res.data;
      if (data.status === 'started') {
        setStatus('running');
        setStatusMessage(`Merge running (PID ${data.pid})...`);
        setOutputPath(data.outputPath || '');
        startPolling(data.statusFile);
      } else {
        setStatus('error');
        setStatusMessage(data.error || 'Unexpected response from server.');
      }
    } catch (error: any) {
      const msg = error.response?.data?.error || error.message || 'Failed to start merge.';
      setStatus('error');
      setStatusMessage(msg);
    }
  };

  const isRunning = status === 'starting' || status === 'running';

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Orthogonal Character Merge (OCR-Merge)</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-green-800 hover:bg-green-700 disabled:opacity-50 px-4 py-2 rounded-md"
            onClick={handleMerge}
            disabled={isRunning}
          >
            {isRunning ? 'Merging...' : 'Run Merge'}
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <div className="max-w-4xl mx-auto py-8 space-y-6">

          {status !== 'idle' && (
            <div className={`px-4 py-3 rounded-lg text-sm ${
              status === 'completed' ? 'bg-green-900/50 text-green-300' :
              status === 'error' ? 'bg-red-900/50 text-red-300' :
              'bg-blue-900/50 text-blue-300'
            }`}>
              <div className="font-medium">
                {status === 'starting' && 'Starting...'}
                {status === 'running' && 'Merge Running'}
                {status === 'completed' && 'Merge Complete'}
                {status === 'error' && 'Merge Failed'}
              </div>
              <div className="mt-1 text-xs opacity-80">{statusMessage}</div>
              {outputPath && status === 'completed' && (
                <div className="mt-1 text-xs">Output: <span className="font-mono">{outputPath}</span></div>
              )}
            </div>
          )}

          <Card title="Input Character LoRAs">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Character A LoRA Path</label>
                <TextInput
                  value={lora1Path}
                  onChange={setLora1Path}
                  placeholder="C:\path\to\char_a.safetensors"
                  disabled={isRunning}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Character B LoRA Path</label>
                <TextInput
                  value={lora2Path}
                  onChange={setLora2Path}
                  placeholder="C:\path\to\char_b.safetensors"
                  disabled={isRunning}
                />
              </div>
            </div>
          </Card>

          <Card title="Output Settings">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Output Name</label>
                <TextInput
                  value={outputName}
                  onChange={setOutputName}
                  placeholder="merged_characters"
                  disabled={isRunning}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Saves to <span className="font-mono">output/{outputName}.safetensors</span> in the ai-toolkit folder.
                </p>
              </div>
            </div>
          </Card>

          <Card title="Omni-Merge Options (DO-Merge 2026)">
            <div className="space-y-6 pt-2">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="block text-sm font-medium">Bilateral Orthogonalization Strength</label>
                  <span className="text-xs text-gray-400">{dareDropRate}</span>
                </div>
                <SliderInput
                  value={dareDropRate}
                  onChange={setDareDropRate}
                  min={0.01}
                  max={0.99}
                  step={0.01}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Controls how strictly the triggers, AUDIO VOICES, and TEMPORAL MOTION are symmetrically isolated (BSO). Structural (Spatial) layers use Magnitude/Direction Decoupling (DO-Merge) to prevent one LoRA from overpowering the other's face/body.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </MainContent>
    </>
  );
}

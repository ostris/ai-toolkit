import React, { useState } from 'react';
import { Modal } from './Modal';
import { Job } from '@prisma/client';
import { getTotalSteps } from '@/utils/jobs';

interface ContinueTrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
  job: Job;
  onContinue: (mode: 'resume' | 'clone', newSteps: number, newName?: string) => void;
}

export const ContinueTrainingModal: React.FC<ContinueTrainingModalProps> = ({
  isOpen,
  onClose,
  job,
  onContinue,
}) => {
  const [mode, setMode] = useState<'resume' | 'clone'>('resume');
  const currentSteps = getTotalSteps(job);
  const [newSteps, setNewSteps] = useState(currentSteps + 2000);
  const [newName, setNewName] = useState(`${job.name}_continued`);

  const handleContinue = () => {
    onContinue(mode, newSteps, mode === 'clone' ? newName : undefined);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Continue Training" size="lg">
      <div className="space-y-6">
        {/* Mode Selection */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-200">Continue Mode</label>

          {/* Resume Option */}
          <div
            className={`cursor-pointer rounded-lg border-2 p-4 transition-colors ${
              mode === 'resume'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 bg-gray-800 hover:border-gray-600'
            }`}
            onClick={() => setMode('resume')}
          >
            <div className="flex items-start">
              <input
                type="radio"
                name="mode"
                checked={mode === 'resume'}
                onChange={() => setMode('resume')}
                className="mt-1 h-4 w-4 text-blue-500"
              />
              <div className="ml-3">
                <h4 className="text-base font-semibold text-gray-100">Resume Training</h4>
                <p className="mt-1 text-sm text-gray-400">
                  Continue from the last checkpoint with the same job name. Training will resume from
                  step {job.step} and continue to the new step count.
                </p>
                <div className="mt-2 flex items-center space-x-2 text-xs text-gray-500">
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>Keeps same name and continues from checkpoint</span>
                </div>
              </div>
            </div>
          </div>

          {/* Clone Option */}
          <div
            className={`cursor-pointer rounded-lg border-2 p-4 transition-colors ${
              mode === 'clone'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 bg-gray-800 hover:border-gray-600'
            }`}
            onClick={() => setMode('clone')}
          >
            <div className="flex items-start">
              <input
                type="radio"
                name="mode"
                checked={mode === 'clone'}
                onChange={() => setMode('clone')}
                className="mt-1 h-4 w-4 text-blue-500"
              />
              <div className="ml-3">
                <h4 className="text-base font-semibold text-gray-100">Start Fresh from Weights</h4>
                <p className="mt-1 text-sm text-gray-400">
                  Create a new job with a different name, using the final checkpoint as starting weights.
                  Training will start from step 0 with the loaded weights.
                </p>
                <div className="mt-2 flex items-center space-x-2 text-xs text-gray-500">
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  <span>Creates new job with pretrained weights</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* New Name (only for clone mode) */}
        {mode === 'clone' && (
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">New Job Name</label>
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-gray-100 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter new job name"
            />
          </div>
        )}

        {/* New Steps */}
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            {mode === 'resume' ? 'Total Steps (increase to continue)' : 'Total Steps for New Training'}
          </label>
          <div className="flex items-center space-x-3">
            <input
              type="number"
              value={newSteps}
              onChange={e => setNewSteps(parseInt(e.target.value) || 0)}
              className="flex-1 rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-gray-100 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              min={mode === 'resume' ? job.step : 0}
            />
            {mode === 'resume' && (
              <div className="text-sm text-gray-400">
                Current: {job.step} / {currentSteps}
              </div>
            )}
          </div>
          {mode === 'resume' && newSteps <= job.step && (
            <p className="mt-1 text-xs text-red-400">
              Steps must be greater than current step ({job.step})
            </p>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
          <button
            onClick={onClose}
            className="rounded-lg border border-gray-600 px-4 py-2 text-sm font-medium text-gray-300 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Cancel
          </button>
          <button
            onClick={handleContinue}
            disabled={mode === 'resume' && newSteps <= job.step}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {mode === 'resume' ? 'Resume Training' : 'Create & Start'}
          </button>
        </div>
      </div>
    </Modal>
  );
};

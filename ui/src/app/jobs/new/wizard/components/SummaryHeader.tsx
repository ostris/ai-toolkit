'use client';

import { ConfigSummary } from '../utils/types';
import { FaExclamationTriangle, FaCheck, FaMicrochip, FaRuler, FaClock, FaMemory } from 'react-icons/fa';

interface Props {
  summary: ConfigSummary;
}

export default function SummaryHeader({ summary }: Props) {
  return (
    <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 p-3">
      <div className="flex flex-wrap gap-4 items-center">
        {/* Model Badge */}
        <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
          <FaMicrochip className="text-blue-500" />
          <span className="text-sm font-medium">
            {summary.model || <span className="text-gray-400 italic">No model</span>}
          </span>
        </div>

        {/* Resolution Badge */}
        <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
          <FaRuler className="text-green-500" />
          <span className="text-sm font-medium">
            {summary.resolution || <span className="text-gray-400 italic">Not set</span>}
          </span>
        </div>

        {/* Steps Badge */}
        <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
          <FaClock className="text-purple-500" />
          <span className="text-sm font-medium">
            {summary.steps ? `${summary.steps} steps` : <span className="text-gray-400 italic">Steps not set</span>}
          </span>
        </div>

        {/* VRAM Badge */}
        <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
          <FaMemory className="text-orange-500" />
          <span className="text-sm font-medium">{summary.estimatedVRAM}</span>
        </div>

        {/* Warnings */}
        {summary.warnings.length > 0 && (
          <div className="ml-auto flex items-center gap-1 text-yellow-600 dark:text-yellow-400">
            <FaExclamationTriangle />
            <span className="text-sm">{summary.warnings.length} warning{summary.warnings.length !== 1 ? 's' : ''}</span>
          </div>
        )}

        {/* All Good Indicator */}
        {summary.warnings.length === 0 && summary.model && summary.resolution && summary.steps && (
          <div className="ml-auto flex items-center gap-1 text-green-600 dark:text-green-400">
            <FaCheck />
            <span className="text-sm">Config looks good</span>
          </div>
        )}
      </div>
    </div>
  );
}

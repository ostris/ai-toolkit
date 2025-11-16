'use client';

import { useState, useEffect } from 'react';
import { Job } from '@prisma/client';
import { FaExclamationTriangle, FaInfoCircle, FaCheckCircle, FaSpinner, FaRedo } from 'react-icons/fa';
import { Button } from '@headlessui/react';

interface AnalysisSummary {
  peak_memory_gb: number;
  total_memory_gb: number;
  avg_memory_utilization: number;
  steps_completed: number;
  duration_minutes: number;
  training_completed: boolean;
  oom_count: number;
  worker_count: number;
  swap_usage_detected: boolean;
}

interface Recommendation {
  severity: 'critical' | 'warning' | 'info';
  title: string;
  action: string;
  impact: string;
  config_path: string | null;
  current_value: any;
  recommended_value: any;
  priority: number;
}

interface LogError {
  line_number: number;
  message: string;
  type: string;
  timestamp?: number;
  traceback?: string;
}

interface AnalysisData {
  summary: AnalysisSummary | null;
  recommendations: Recommendation[];
  log_errors: LogError[];
  health_score: 'good' | 'warning' | 'critical' | 'unknown';
  metrics_timeline: any[];
}

interface Props {
  job: Job;
}

export default function PerformanceAnalysis({ job }: Props) {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/jobs/${job.id}/analysis`);
      if (!response.ok) {
        throw new Error('Failed to fetch analysis');
      }
      const data = await response.json();
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const generateAnalysis = async () => {
    setGenerating(true);
    setError(null);
    try {
      const config = JSON.parse(job.job_config || '{}');
      const jobName = config?.config?.name || '';

      const response = await fetch(`/api/jobs/${job.id}/analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobName }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate analysis');
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    fetchAnalysis();
  }, [job.id]);

  const getHealthScoreBadge = (score: string) => {
    switch (score) {
      case 'critical':
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
            <FaExclamationTriangle className="mr-1" /> Critical
          </span>
        );
      case 'warning':
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
            <FaExclamationTriangle className="mr-1" /> Warning
          </span>
        );
      case 'good':
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
            <FaCheckCircle className="mr-1" /> Good
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
            <FaInfoCircle className="mr-1" /> Unknown
          </span>
        );
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <FaExclamationTriangle className="text-red-500" />;
      case 'warning':
        return <FaExclamationTriangle className="text-yellow-500" />;
      default:
        return <FaInfoCircle className="text-blue-500" />;
    }
  };

  const getSeverityBorder = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500';
      case 'warning':
        return 'border-yellow-500';
      default:
        return 'border-blue-500';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <FaSpinner className="animate-spin text-2xl text-gray-500" />
        <span className="ml-2">Loading analysis...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-300 rounded-md">
        <p className="text-red-700 dark:text-red-300">Error: {error}</p>
        <Button
          onClick={fetchAnalysis}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
        >
          Retry
        </Button>
      </div>
    );
  }

  const hasNoData = !analysis?.summary && analysis?.recommendations.length === 0;

  return (
    <div className="space-y-6 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Performance Analysis</h2>
        <div className="flex items-center gap-2">
          {analysis && getHealthScoreBadge(analysis.health_score)}
          <Button
            onClick={generateAnalysis}
            disabled={generating}
            className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-1"
          >
            {generating ? (
              <>
                <FaSpinner className="animate-spin" /> Analyzing...
              </>
            ) : (
              <>
                <FaRedo /> Re-analyze
              </>
            )}
          </Button>
        </div>
      </div>

      {hasNoData ? (
        <div className="p-6 bg-gray-50 dark:bg-gray-800 rounded-lg text-center">
          <FaInfoCircle className="text-4xl text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            No monitoring data available for this job.
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500">
            Monitoring data is collected automatically during training when enabled in the config.
          </p>
        </div>
      ) : (
        <>
          {/* Summary Card */}
          {analysis?.summary && (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 className="font-semibold mb-3">Training Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Peak Memory</p>
                  <p className="font-medium">
                    {analysis.summary.peak_memory_gb.toFixed(1)}GB / {analysis.summary.total_memory_gb.toFixed(0)}GB
                    <span className="text-sm text-gray-500 ml-1">
                      ({((analysis.summary.peak_memory_gb / analysis.summary.total_memory_gb) * 100).toFixed(0)}%)
                    </span>
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Duration</p>
                  <p className="font-medium">
                    {analysis.summary.duration_minutes >= 60
                      ? `${(analysis.summary.duration_minutes / 60).toFixed(1)} hours`
                      : `${Math.round(analysis.summary.duration_minutes)} min`}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Steps Completed</p>
                  <p className="font-medium">{analysis.summary.steps_completed}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Status</p>
                  <p className="font-medium">
                    {analysis.summary.training_completed ? (
                      <span className="text-green-600">Completed</span>
                    ) : (
                      <span className="text-yellow-600">Incomplete</span>
                    )}
                  </p>
                </div>
              </div>
              {(analysis.summary.oom_count > 0 || analysis.summary.swap_usage_detected) && (
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex gap-4 text-sm">
                    {analysis.summary.oom_count > 0 && (
                      <span className="text-red-600">
                        <FaExclamationTriangle className="inline mr-1" />
                        {analysis.summary.oom_count} OOM error(s)
                      </span>
                    )}
                    {analysis.summary.swap_usage_detected && (
                      <span className="text-yellow-600">
                        <FaExclamationTriangle className="inline mr-1" />
                        Swap usage detected
                      </span>
                    )}
                    {analysis.summary.worker_count > 0 && (
                      <span className="text-gray-600 dark:text-gray-400">
                        {analysis.summary.worker_count} worker(s)
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recommendations */}
          {analysis?.recommendations && analysis.recommendations.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 className="font-semibold mb-3">Optimization Recommendations</h3>
              <div className="space-y-3">
                {analysis.recommendations.map((rec, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-md border-l-4 ${getSeverityBorder(rec.severity)} bg-gray-50 dark:bg-gray-900`}
                  >
                    <div className="flex items-start gap-2">
                      {getSeverityIcon(rec.severity)}
                      <div className="flex-1">
                        <h4 className="font-medium">{rec.title}</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{rec.action}</p>
                        <p className="text-sm text-gray-500 mt-1">{rec.impact}</p>
                        {rec.config_path && (
                          <p className="text-xs text-gray-400 mt-1 font-mono">{rec.config_path}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Log Errors */}
          {analysis?.log_errors && analysis.log_errors.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 className="font-semibold mb-3">
                Errors & Warnings
                <span className="text-sm font-normal text-gray-500 ml-2">
                  ({analysis.log_errors.length} found)
                </span>
              </h3>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {analysis.log_errors.slice(0, 20).map((err, index) => (
                  <div
                    key={index}
                    className="p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm"
                  >
                    <p className="font-medium text-red-700 dark:text-red-300">
                      Line {err.line_number}: {err.type}
                    </p>
                    <p className="text-gray-700 dark:text-gray-300 font-mono text-xs mt-1 whitespace-pre-wrap">
                      {err.message}
                    </p>
                  </div>
                ))}
                {analysis.log_errors.length > 20 && (
                  <p className="text-sm text-gray-500 text-center">
                    ... and {analysis.log_errors.length - 20} more
                  </p>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

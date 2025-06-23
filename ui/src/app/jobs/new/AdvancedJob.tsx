'use client';
import { JobConfig } from '@/types';
import Editor, { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { useEffect, useRef, useState } from 'react';
import YAML from 'yaml';

interface Props {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  settings: any;
  // Multi-GPU props
  useMultiGPU: boolean;
  setUseMultiGPU: (value: boolean) => void;
  numGPUs: number;
  setNumGPUs: (value: number) => void;
  accelerateConfig: any;
  setAccelerateConfig: (value: any) => void;
}

const isDev = process.env.NODE_ENV === 'development';

const yamlConfig: YAML.DocumentOptions &
  YAML.SchemaOptions &
  YAML.ParseOptions &
  YAML.CreateNodeOptions &
  YAML.ToStringOptions = {
  indent: 2,
  lineWidth: 999999999999,
  defaultStringType: 'QUOTE_DOUBLE',
  defaultKeyType: 'PLAIN',
  directives: true,
};

export default function AdvancedJob({ jobConfig, setJobConfig, settings }: Props) {
  const [editorValue, setEditorValue] = useState<string>('');
  const lastJobConfigUpdateStringRef = useRef('');
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  // Track if the editor has been mounted
  const isEditorMounted = useRef(false);

  // Handler for editor mounting
  const handleEditorDidMount: OnMount = editor => {
    editorRef.current = editor;
    isEditorMounted.current = true;

    // Initial content setup
    try {
      const yamlContent = YAML.stringify(jobConfig, yamlConfig);
      setEditorValue(yamlContent);
      lastJobConfigUpdateStringRef.current = JSON.stringify(jobConfig);
    } catch (e) {
      console.warn(e);
    }
  };

  useEffect(() => {
    const lastUpdate = lastJobConfigUpdateStringRef.current;
    const currentUpdate = JSON.stringify(jobConfig);

    // Skip if no changes or editor not yet mounted
    if (lastUpdate === currentUpdate || !isEditorMounted.current) {
      return;
    }

    try {
      // Preserve cursor position and selection
      const editor = editorRef.current;
      if (editor) {
        // Save current editor state
        const position = editor.getPosition();
        const selection = editor.getSelection();
        const scrollTop = editor.getScrollTop();

        // Update content
        const yamlContent = YAML.stringify(jobConfig, yamlConfig);

        // Only update if the content is actually different
        if (yamlContent !== editor.getValue()) {
          // Set value directly on the editor model instead of using React state
          editor.getModel()?.setValue(yamlContent);

          // Restore cursor position and selection
          if (position) editor.setPosition(position);
          if (selection) editor.setSelection(selection);
          editor.setScrollTop(scrollTop);
        }

        lastJobConfigUpdateStringRef.current = currentUpdate;
      }
    } catch (e) {
      console.warn(e);
    }
  }, [jobConfig]);

  const handleChange = (value: string | undefined) => {
    if (value === undefined) return;

    try {
      const parsed = YAML.parse(value);
      // Don't update jobConfig if the change came from the editor itself
      // to avoid a circular update loop
      if (JSON.stringify(parsed) !== lastJobConfigUpdateStringRef.current) {
        lastJobConfigUpdateStringRef.current = JSON.stringify(parsed);

        // We have to ensure certain things are always set
        try {
          parsed.config.process[0].type = 'ui_trainer';
          parsed.config.process[0].sqlite_db_path = './aitk_db.db';
          parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
          parsed.config.process[0].device = 'cuda';
          parsed.config.process[0].performance_log_every = 10;
        } catch (e) {
          console.warn(e);
        }
        setJobConfig(parsed, 'config');
      }
    } catch (e) {
      // Don't update on parsing errors
      console.warn(e);
    }
  };

  return (
    <>
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme="vs-dark"
        onChange={handleChange}
        onMount={handleEditorDidMount}
        options={{
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          automaticLayout: true,
        }}
      />
    </>
  );
}

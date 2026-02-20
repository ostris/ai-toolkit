'use client';
import { useEffect, useState } from 'react';
import YAML from 'yaml';
import Editor from '@monaco-editor/react';

import { Job } from '@prisma/client';

interface Props {
  job: Job;
}

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

export default function JobConfigViewer({ job }: Props) {
  const [editorValue, setEditorValue] = useState<string>('');
  useEffect(() => {
    if (job?.job_config) {
      const yamlContent = YAML.stringify(JSON.parse(job.job_config), yamlConfig);
      setEditorValue(yamlContent);
    }
  }, [job]);
  return (
    <>
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme="vs-dark"
        options={{
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          automaticLayout: true,
          readOnly: true,
        }}
      />
    </>
  );
}

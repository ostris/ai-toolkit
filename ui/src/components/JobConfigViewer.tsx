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

function toYaml(obj: any): string {
  const doc = new YAML.Document(obj, yamlConfig);
  YAML.visit(doc, {
    Scalar(_key, node) {
      if (typeof node.value === 'string' && node.value.includes('\n')) {
        node.type = YAML.Scalar.BLOCK_LITERAL;
      }
    },
  });
  return doc.toString(yamlConfig);
}

export default function JobConfigViewer({ job }: Props) {
  const [editorValue, setEditorValue] = useState<string>('');
  useEffect(() => {
    if (job?.job_config) {
      const yamlContent = toYaml(JSON.parse(job.job_config));
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

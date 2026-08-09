'use client';
import { useEffect, useState, useRef } from 'react';
import YAML from 'yaml';
import Editor, { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { useTheme } from '@/components/ThemeProvider';

type Props<T> = {
  config: T;
  setConfig: (value: any, key?: string) => void;
  transformOnParse?: (parsed: any) => any;
};

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

export default function AdvancedConfigEditor<T>({ config, setConfig, transformOnParse }: Props<T>) {
  const { theme } = useTheme();
  const [editorValue, setEditorValue] = useState<string>('');
  const [hasError, setHasError] = useState(false);
  const lastConfigUpdateStringRef = useRef('');
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<any>(null);

  // Track if the editor has been mounted
  const isEditorMounted = useRef(false);

  // Handler for editor mounting
  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    isEditorMounted.current = true;

    // Initial content setup
    try {
      const yamlContent = toYaml(config);
      setEditorValue(yamlContent);
      lastConfigUpdateStringRef.current = JSON.stringify(config);
    } catch (e) {
      console.warn(e);
    }
  };

  useEffect(() => {
    const lastUpdate = lastConfigUpdateStringRef.current;
    const currentUpdate = JSON.stringify(config);

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
        const yamlContent = toYaml(config);

        // Only update if the content is actually different
        if (yamlContent !== editor.getValue()) {
          // Set value directly on the editor model instead of using React state
          editor.getModel()?.setValue(yamlContent);

          // Restore cursor position and selection
          if (position) editor.setPosition(position);
          if (selection) editor.setSelection(selection);
          editor.setScrollTop(scrollTop);
        }

        lastConfigUpdateStringRef.current = currentUpdate;
      }
    } catch (e) {
      console.warn(e);
    }
  }, [config]);

  const setMarkers = (errors: { message: string; line: number }[]) => {
    const monaco = monacoRef.current;
    const model = editorRef.current?.getModel();
    if (!monaco || !model) return;
    const markers = errors.map(err => ({
      severity: monaco.MarkerSeverity.Error,
      message: err.message,
      startLineNumber: err.line,
      startColumn: 1,
      endLineNumber: err.line,
      endColumn: model.getLineMaxColumn(err.line),
    }));
    monaco.editor.setModelMarkers(model, 'yaml', markers);
  };

  const handleChange = (value: string | undefined) => {
    if (value === undefined) return;

    try {
      let parsed = YAML.parse(value);
      setHasError(false);
      setMarkers([]);

      // Don't update config if the change came from the editor itself
      // to avoid a circular update loop
      if (JSON.stringify(parsed) !== lastConfigUpdateStringRef.current) {
        if (transformOnParse) {
          parsed = transformOnParse(parsed);
        }
        lastConfigUpdateStringRef.current = JSON.stringify(parsed);
        setConfig(parsed);
      }
    } catch (e: any) {
      setHasError(true);
      const line = e?.linePos?.[0]?.line ?? e?.linePos?.line ?? 1;
      setMarkers([{ message: e?.message ?? 'Invalid YAML', line }]);
    }
  };

  return (
    <div className="relative h-full w-full">
      {hasError && (
        <div
          className="absolute inset-0 z-10 pointer-events-none rounded-sm"
          style={{ boxShadow: 'inset 0 0 12px 2px rgba(239, 68, 68, 0.5)' }}
        />
      )}
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme={theme === 'dark' ? 'vs-dark' : 'light'}
        className="z-0"
        onChange={handleChange}
        onMount={handleEditorDidMount}
        options={{
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          automaticLayout: true,
        }}
      />
    </div>
  );
}

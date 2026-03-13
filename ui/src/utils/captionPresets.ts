export interface PartialOption {
  filename: string;
  content: string;
}

export interface TemplateVariable {
  name: string;
  options: PartialOption[];
}

export interface CaptionPreset {
  name: string;
  rawContent: string;
  renderedContent: string;
  variables: TemplateVariable[];
}

export const VARIABLE_PATTERN = /\$\{(\w+)=\[[^\]]+\]\}/g;

export function renderTemplate(rawContent: string, selections: Record<string, string>): string {
  return rawContent.replace(VARIABLE_PATTERN, (_match, varName) => selections[varName] ?? '');
}

export function applySelections(preset: CaptionPreset, selections: Record<string, number>): string {
  const contentSelections: Record<string, string> = {};
  for (const variable of preset.variables) {
    const idx = selections[variable.name] ?? 0;
    contentSelections[variable.name] = variable.options[idx]?.content ?? '';
  }
  return renderTemplate(preset.rawContent, contentSelections);
}

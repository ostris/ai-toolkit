export interface PartialOption {
  filename: string;
  content: string;
  rawContent: string;
  source: 'default' | 'user';
  scope: string;
  variables: TemplateVariable[];
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
  source: 'default' | 'user';
}

export const VARIABLE_PATTERN = /\$\{(\w+)=(\w[\w-]*\/)\}/g;

export function renderTemplate(rawContent: string, selections: Record<string, string>): string {
  return rawContent.replace(VARIABLE_PATTERN, (_match, varName) => selections[varName] ?? '');
}

/**
 * Recursively resolve an option's rawContent using nested variable selections.
 */
function resolveOptionContent(option: PartialOption, selections: Record<string, number>): string {
  if (!option.variables || option.variables.length === 0) {
    return option.rawContent;
  }

  let content = option.rawContent;
  for (const variable of option.variables) {
    const idx = selections[variable.name] ?? 0;
    const nestedOption = variable.options[idx];
    const resolved = nestedOption ? resolveOptionContent(nestedOption, selections) : '';
    content = content.replace(
      new RegExp(`\\$\\{${variable.name}=\\w[\\w-]*\\/\\}`, 'g'),
      resolved,
    );
  }
  return content;
}

export function applySelections(preset: CaptionPreset, selections: Record<string, number>): string {
  const contentSelections: Record<string, string> = {};
  for (const variable of preset.variables) {
    const idx = selections[variable.name] ?? 0;
    const option = variable.options[idx];
    if (!option) continue;
    contentSelections[variable.name] = resolveOptionContent(option, selections);
  }
  return renderTemplate(preset.rawContent, contentSelections);
}

/**
 * Collect all active variables by walking selections recursively.
 * Returns a flat list of variables that should have dropdowns rendered.
 */
export function getActiveVariables(variables: TemplateVariable[], selections: Record<string, number>): TemplateVariable[] {
  const result: TemplateVariable[] = [];
  for (const variable of variables) {
    result.push(variable);
    const idx = selections[variable.name] ?? 0;
    const option = variable.options[idx];
    if (option?.variables?.length) {
      result.push(...getActiveVariables(option.variables, selections));
    }
  }
  return result;
}

import * as nodeModule from 'module';
import os from 'os';
import path from 'path';

/**
 * macstats is a native, macOS-only optional dependency.
 *
 * It must not be bundled. Webpack special-cases `createRequire(...)`: with a literal base it
 * resolves and inlines the module (a hard MODULE_NOT_FOUND throw when it can't), and with a
 * computed base it replaces the whole call with `void 0` — both leave macstats unreachable at
 * runtime even when it is installed, regardless of serverExternalPackages/externals. So look
 * createRequire up dynamically, where the bundler can't recognize it, and base the require on
 * process.cwd() (the ui/ directory the Next server runs in) so it resolves against
 * ui/node_modules at runtime.
 */

// undefined = not tried yet, null = unavailable on this machine
let cachedModule: any | null | undefined;

export function loadMacstats(): any | null {
  if (cachedModule !== undefined) return cachedModule;

  if (os.platform() !== 'darwin') {
    cachedModule = null;
    return cachedModule;
  }

  try {
    const createRequire = (nodeModule as any)['create' + 'Require'] as typeof nodeModule.createRequire;
    const nativeRequire = createRequire(path.join(process.cwd(), 'package.json'));
    cachedModule = nativeRequire('macstats');
  } catch (error) {
    console.warn('macstats not available:', error);
    cachedModule = null;
  }

  return cachedModule;
}

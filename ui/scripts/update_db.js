const { spawnSync } = require('child_process');

// Keep local UI startup backwards-compatible while allowing Beam to point
// Prisma at a persistent Volume through DATABASE_URL.
const env = {
  ...process.env,
  DATABASE_URL: process.env.DATABASE_URL || 'file:../../aitk_db.db',
};
const npmCommand = process.platform === 'win32' ? 'npx.cmd' : 'npx';

for (const args of [
  ['prisma', 'generate'],
  ['prisma', 'db', 'push'],
]) {
  const result = spawnSync(npmCommand, args, {
    env,
    stdio: 'inherit',
    shell: false,
  });
  if (result.error) {
    console.error(`Failed to run ${npmCommand} ${args.join(' ')}: ${result.error.message}`);
    process.exit(1);
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

import processQueue from './actions/processQueue';
import prisma from './prisma';

// Journal mode for the main sqlite db. WAL keeps readers from blocking while
// the trainer/worker write, which is what we want on a local disk. Users on
// setups where WAL can't work (e.g. db on a network filesystem) can override
// with AI_TOOLKIT_DB_JOURNAL_MODE=DELETE (or any other valid sqlite mode).
const DEFAULT_JOURNAL_MODE = 'WAL';
const VALID_JOURNAL_MODES = ['DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF'];

async function ensureJournalMode() {
  const envMode = process.env.AI_TOOLKIT_DB_JOURNAL_MODE;
  let targetMode = (envMode || DEFAULT_JOURNAL_MODE).toUpperCase();
  if (!VALID_JOURNAL_MODES.includes(targetMode)) {
    console.warn(
      `Invalid AI_TOOLKIT_DB_JOURNAL_MODE "${envMode}", expected one of ${VALID_JOURNAL_MODES.join(', ')}. Using ${DEFAULT_JOURNAL_MODE}.`,
    );
    targetMode = DEFAULT_JOURNAL_MODE;
  }

  const current = await prisma.$queryRawUnsafe<{ journal_mode: string }[]>('PRAGMA journal_mode;');
  const currentMode = current[0]?.journal_mode?.toUpperCase();
  if (currentMode === targetMode) {
    return;
  }

  console.log(`Converting database journal mode from ${currentMode} to ${targetMode}...`);
  // targetMode is validated against VALID_JOURNAL_MODES above, safe to interpolate
  const result = await prisma.$queryRawUnsafe<{ journal_mode: string }[]>(`PRAGMA journal_mode = ${targetMode};`);
  const resultMode = result[0]?.journal_mode?.toUpperCase();
  if (resultMode === targetMode) {
    console.log(`Database journal mode is now ${resultMode}.`);
  } else {
    // sqlite refuses the switch rather than corrupting anything (e.g. WAL on a
    // network filesystem), so just report what we're actually running with.
    console.warn(`Could not convert database journal mode to ${targetMode}, still using ${resultMode}.`);
  }
}

class CronWorker {
  interval: number;
  is_running: boolean;
  intervalId: NodeJS.Timeout;
  constructor() {
    this.interval = 1000; // Default interval of 1 second
    this.is_running = false;
    this.intervalId = setInterval(() => {
      this.run();
    }, this.interval);
  }
  async run() {
    if (this.is_running) {
      return;
    }
    this.is_running = true;
    try {
      // Loop logic here
      await this.loop();
    } catch (error) {
      console.error('Error in cron worker loop:', error);
    }
    this.is_running = false;
  }

  async loop() {
    await processQueue();
  }
}

// make sure the db journal mode is set before the loop starts hitting the db
ensureJournalMode()
  .catch(error => {
    console.warn('Could not check/convert database journal mode:', error);
  })
  .finally(() => {
    // it automatically starts the loop
    const cronWorker = new CronWorker();
    console.log('Cron worker started with interval:', cronWorker.interval, 'ms');
  });

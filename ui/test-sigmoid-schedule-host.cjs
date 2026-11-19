const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const ts = require('typescript');

const file = path.join(__dirname, 'src/utils/sigmoidSchedule.ts');
const { outputText } = ts.transpileModule(readFileSync(file, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const helpers = {};
new Function('exports', outputText)(helpers);
const { parseScheduleRows, trainingBucketsFromLog, sigmoidJobProcess, scheduleContext, jobScheduleStep } = helpers;

test('an empty bucket lookup retries on progress after resume', () => {
  const { bucketLookupStep } = helpers;
  assert.notEqual(bucketLookupStep([], 800), bucketLookupStep([], 801));
  const buckets = [{ width: 832, height: 1248 }];
  assert.equal(bucketLookupStep(buckets, 801), bucketLookupStep(buckets, 802));
});

test('the host accepts only ordered schedule rows and recognized fields', () => {
  const rows = [
    { start_step: 9, shift: -0.25, scale: 1.15 },
    { start_step: 71, shift: 1.3, scale: 0.45 },
  ];
  assert.deepEqual(parseScheduleRows(rows), [
    { start_step: 9, shift: -0.25, scale: 1.15 },
    { start_step: 71, shift: 1.3, scale: 0.45 },
  ]);
  assert.deepEqual(parseScheduleRows([]), []);
  for (const invalid of [
    null,
    {},
    [null],
    [rows[1], rows[0]],
    [rows[0], rows[0]],
    [{ ...rows[0], extra: true }],
    [{ ...rows[0], start_step: 3.5 }],
    [{ ...rows[0], start_step: -1 }],
    [{ ...rows[0], shift: '0.2' }],
    [{ ...rows[0], shift: Infinity }],
    [{ ...rows[0], scale: 0 }],
    [{ ...rows[0], scale: true }],
  ])
    assert.equal(parseScheduleRows(invalid), null);
});

test('runtime bucket dimensions and frame counts come from toolkit bucket blocks', () => {
  const log = [
    '64x64: 1 files',
    'Bucket sizes for /job/image:',
    '832x1248: 40 files',
    '1024x1024: 9 files (example=9)',
    '2 buckets made',
    '64x64: 1 files',
    'Bucket sizes for /job/video:',
    '896x512x33f: 10 files',
    '896x512x49f: 2 files',
    '2 buckets made',
    'Bucket sizes for /job/duplicate:',
    '832x1248: 1 files',
    '1 buckets made',
  ].join('\r\n');
  assert.deepEqual(trainingBucketsFromLog(log), [
    { width: 832, height: 1248 },
    { width: 1024, height: 1024 },
    { width: 896, height: 512, frames: 33 },
    { width: 896, height: 512, frames: 49 },
  ]);
});

test('the saved-job plugin is offered for FlowMatch sigmoid and sends only relevant context', () => {
  const process = {
    train: { noise_scheduler: 'flowmatch', timestep_type: 'sigmoid', steps: 500, timestep_shift: -0.2, lr: 0.123 },
    model: { arch: 'moss_image:base', name_or_path: '/private/ckpt', model_kwargs: { api_key: 'secret' } },
    datasets: [{ folder_path: '/private/dataset' }],
  };
  const job = value => JSON.stringify({ config: { process: [value] } });
  assert.deepEqual(sigmoidJobProcess(job(process)), process);
  assert.deepEqual(
    sigmoidJobProcess(job({ ...process, train: { noise_scheduler: 'flowmatch' } })).model,
    process.model,
  );
  assert.equal(
    sigmoidJobProcess(job({ ...process, train: { ...process.train, timestep_type: 'infonoise' } })),
    undefined,
  );
  assert.equal(sigmoidJobProcess(job({ ...process, train: { ...process.train, noise_scheduler: 'ddpm' } })), undefined);
  assert.equal(sigmoidJobProcess(job({ ...process, train: { ...process.train, linear_timesteps: true } })), undefined);
  assert.equal(sigmoidJobProcess(job({ ...process, model: { arch: 'minimax_h3_image' } })), undefined);
  assert.equal(sigmoidJobProcess('malformed'), undefined);
  const context = scheduleContext(process);
  assert.equal(context.train.steps, 500);
  assert.equal(context.train.timestep_shift, -0.2);
  assert.equal(context.model.arch, 'moss_image:base');
  assert.doesNotMatch(JSON.stringify(context), /private|secret|"lr"/);
  assert.equal(scheduleContext({ ...process, model: { is_flux: true } }).model.arch, 'flux');
  assert.equal(scheduleContext({ ...process, model: { arch: 'flex1' } }).model.arch, 'flux');
});

test('a completed job shows its last trained step rather than the next index', () => {
  assert.equal(jobScheduleStep({ step: 100, status: 'completed' }, 100), 99);
  assert.equal(jobScheduleStep({ step: 80, status: 'completed' }, 100), 80);
  assert.equal(jobScheduleStep({ step: 80, status: 'running' }, 100), 80);
});

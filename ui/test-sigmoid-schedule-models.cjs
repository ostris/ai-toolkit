const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const samplerPath = path.join(__dirname, 'public', 'timestep-schedule', 'sampler.js');
const source = fs.readFileSync(samplerPath, 'utf8');
const sampler = vm.runInNewContext(`${source}\nTimestepSampler;`, {}, { filename: samplerPath });

function close(actual, expected, tolerance = 0.00012) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `Expected ${actual} to be within ${tolerance} of ${expected}`);
}

test('MOSS variants share the static shift-5 training grid', () => {
  const { models, resolveBucket, shiftGrid } = sampler;
  for (const arch of ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit']) {
    const bucket = resolveBucket(models[arch], {});
    assert.equal(bucket.tokens, null);
    const grid = shiftGrid(models[arch], bucket.tokens);
    assert.equal(grid.length, 1000);
    assert.equal(grid[0], 1000);
    close(grid[500], 834.4370727539062);
    close(grid[999], 24.414066314697266);
  }
});

test('image buckets use the encoded dimensions and effective trainer patch size', () => {
  const { models, resolveBucket, shiftGrid } = sampler;
  const bucket = { width: 1024, height: 768 };
  const expected = [
    ['flux', 128, 96, 3072],
    ['flex2', 128, 96, 3072],
    ['flux_kontext', 128, 96, 12288],
    ['flux2', 64, 48, 3072],
    ['flux2_klein_4b', 64, 48, 3072],
    ['flux2_klein_9b', 64, 48, 3072],
    ['fibo2', 64, 48, 3072],
    ['chroma', 128, 96, 12288],
    ['chroma_radiance', 1024, 768, 786432],
    ['f-lite', 128, 96, 3072],
    ['krea2', 128, 96, 3072],
    ['ming_image', 128, 96, 3072],
    ['cogview4', 128, 96, 3072],
    ['pixart', 128, 96, 3072],
    ['qwen_image', 128, 96, 3072],
    ['qwen_image_edit', 128, 96, 3072],
    ['qwen_image_edit_plus', 128, 96, 3072],
    ['qwen_image_2', 64, 48, 3072],
  ];
  for (const [arch, latentWidth, latentHeight, tokens] of expected) {
    const result = resolveBucket(models[arch], bucket);
    assert.equal(result.latentWidth, latentWidth, arch);
    assert.equal(result.latentHeight, latentHeight, arch);
    assert.equal(result.tokens, tokens, arch);
  }
  for (const [arch, midpoint, terminal] of [
    ['flux', 726.4463500976562, 2.651200294494629],
    ['krea2', 689.5286254882812, 2.218201160430908],
    ['qwen_image_2', 661.4443359375, 20],
  ]) {
    const grid = shiftGrid(models[arch], resolveBucket(models[arch], bucket).tokens);
    close(grid[500], midpoint);
    close(grid[999], terminal);
  }
});

test('LTX Shift uses the compressed frame and height axes', () => {
  const { models, resolveBucket, shiftGrid } = sampler;
  for (const arch of ['ltx2', 'ltx2.3', 'ltx2.5']) {
    const model = models[arch];
    assert.equal(model.videoBucket, true, arch);
    const bucket = resolveBucket(model, { width: 832, height: 480, frames: 49 });
    assert.equal(bucket.latentWidth, 26, arch);
    assert.equal(bucket.latentHeight, 15, arch);
    assert.equal(bucket.latentFrames, 7, arch);
    assert.equal(bucket.tokens, 105, arch);
    assert.equal(resolveBucket(model, { width: 480, height: 832, frames: 49 }).tokens, 182, arch);
    assert.equal(resolveBucket(model, { width: 960, height: 480, frames: 49 }).tokens, 105, arch);
    assert.equal(resolveBucket(model, { width: 832, height: 480, frames: 1 }).tokens, 15, arch);
    const grid = shiftGrid(model, bucket.tokens);
    close(grid[500], 684.80126953125);
    close(grid[999], 100);
  }
});

test('invalid or unverified geometry fails with a specific reason', () => {
  const { models, resolveBucket } = sampler;
  assert.throws(() => resolveBucket(models.qwen_image_2, { width: 1008, height: 768 }), /multiples of 32/);
  assert.throws(() => resolveBucket(models.flux2, { width: 1024.5, height: 768 }), /positive whole numbers/);
  assert.throws(() => resolveBucket(models.ltx2, { width: 832, height: 480, frames: 48 }), /1, 9, 17/);
  assert.throws(() => resolveBucket(models.ltx2, { width: 832, height: 480 }), /frame count/);
  assert.throws(() => resolveBucket(models.chroma_radiance, { width: 4096, height: 2048 }), /training scheduler/);
  assert.throws(
    () => resolveBucket({ label: 'Example', dynamic: {} }, { width: 1024, height: 768 }),
    /geometry is unavailable/,
  );
});

test('Wan stage bounds match the trainer’s range selection', () => {
  const { models, shiftGrid } = sampler;
  const model = models.wan22_14b;
  const high = shiftGrid(model, null, 'high');
  const low = shiftGrid(model, null, 'low');
  assert.equal(high.length, 419);
  assert.equal(low.length, 582);
  close(high[418], 875.1493530273438);
  assert.equal(high[418], low[0]);
});

test('fixed MiniMax H3 Image ranks are uniform across the six training sigmas', () => {
  const { models, shiftGrid, gridCurve } = sampler;
  const model = models.minimax_h3_image;
  const grid = shiftGrid(model, null);
  assert.equal(grid.length, 6);
  const curve = gridCurve(grid, 'style', Boolean(model.fixed));
  close(curve.cdf(800), 0.5);
  close(curve.quantile(0.9), 1000);
});

test('sigmoid expectation follows shift direction and rank bias', () => {
  const { sigmoidCurve } = sampler;
  close(sigmoidCurve(0, 1, 'balanced').quantile(0.5), 500.31471252441406, 0.02);
  assert.ok(sigmoidCurve(-0.7, 0.9, 'balanced').quantile(0.5) > 650);
  assert.ok(sigmoidCurve(0, 1, 'content').cdf(500) < 0.25);
  assert.ok(sigmoidCurve(0, 1, 'style').cdf(500) > 0.75);
});

test('distribution uses 200 five-timestep intervals with total probability one', () => {
  const { distributionPoints, sigmoidCurve } = sampler;
  const points = distributionPoints(sigmoidCurve(-1, 1.2, 'balanced'));
  assert.equal(points.length, 200);
  assert.equal(points[0][0], 2.5);
  assert.equal(points.at(-1)[0], 997.5);
  close(
    points.reduce((sum, point) => sum + point[1], 0),
    1,
    1e-10,
  );
});

test('the schedule peak covers an interior probability above both endpoint peaks', () => {
  const { scheduleDistributionPeak, distributionPoints, interpolateSigmoid, sigmoidCurve } = sampler;
  const from = { start_step: 0, shift: 0, scale: 0.005 };
  const to = { start_step: 1000, shift: -Math.log(505 / 495), scale: 0.005 };
  const peak = ({ shift, scale }) =>
    Math.max(...distributionPoints(sigmoidCurve(shift, scale, 'balanced')).map(p => p[1]));
  const middle = peak(interpolateSigmoid(from, to, 0.5));
  const maximum = scheduleDistributionPeak({ shift: 0, scale: 1 }, [from, to]);
  assert.ok(middle > Math.max(peak(from), peak(to)) + 0.4);
  assert.ok(maximum >= middle);
  close(maximum, middle, 0.001);
});

test('the schedule peak includes an active scalar fallback', () => {
  const { scheduleDistributionPeak } = sampler;
  const narrow = { shift: 0, scale: 0.0001 };
  const point = { start_step: 10, shift: 0, scale: 1 };
  assert.ok(scheduleDistributionPeak(narrow, [point]) > 0.49);
  assert.ok(scheduleDistributionPeak(narrow, [{ ...point, start_step: 0 }]) < 0.01);
});

test('a narrow peak moving across the chart keeps a tight axis', () => {
  const { scheduleDistributionPeak, distributionPoints, interpolateSigmoid, sigmoidCurve } = sampler;
  const from = { start_step: 0, shift: 0.786, scale: 0.112 };
  const to = { start_step: 1000, shift: -1.728, scale: 0.051 };
  const maximum = scheduleDistributionPeak(from, [from, to]);
  for (let step = 0; step <= 1000; step += 10) {
    const { shift, scale } = interpolateSigmoid(from, to, step / 1000);
    const peak = Math.max(...distributionPoints(sigmoidCurve(shift, scale, 'balanced')).map(point => point[1]));
    assert.ok(maximum >= peak);
  }
  assert.ok(maximum < 0.35, `Expected a tight axis bound, got ${maximum}`);
});

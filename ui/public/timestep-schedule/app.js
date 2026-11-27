(() => {
  'use strict';

  const {
    biasName,
    models,
    sigmoidCurve,
    interpolateSigmoid,
    distributionWidth,
    distributionPoints,
    scheduleDistributionPeak,
    shiftGrid,
    gridCurve,
    resolveBucket,
  } = TimestepSampler;
  const get = id => document.getElementById(id);
  const stepInput = get('step');
  const scrub = get('scrub');
  const reference = get('reference');
  const stage = get('stage');
  const bias = get('bias');
  const bucketSelect = get('bucket');
  const widthInput = get('width');
  const heightInput = get('height');
  const framesInput = get('frames');
  const view = get('view');
  const canvas = get('chart');
  const context = canvas.getContext('2d');
  const tooltip = get('tooltip');
  const darkMode = window.matchMedia('(prefers-color-scheme: dark)');
  const pct = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 });
  const formatStep = new Intl.NumberFormat('en');
  const prefix = 'aitk-timestep-schedule:';

  const scheduleKeys = [
    ['start_step', 'Start step', 'numeric'],
    ['shift', 'Shift', 'decimal'],
    ['scale', 'Scale', 'decimal'],
  ];
  const parameterRanges = {
    shift: { min: -3, max: 3 },
    scale: { min: 0.01, max: 3 },
  };
  const chartTitles = {
    distribution: 'Training timestep distribution',
    survival: 'Share at or above a timestep',
    quantiles: 'Selected timestep quantiles',
  };

  let initialized = false;
  let readOnly = false;
  let totalSteps = null;
  let pointStepEnd = 0;
  let playhead = 0;
  let fallback = { shift: 0, scale: 1 };
  let draft = [];
  let validRows = [];
  let rowElements = [];
  let validationError = '';
  let model = null;
  let modelArch = '';
  let buckets = [];
  let refProfile = null;
  let profiles = [];
  let plotBounds;
  let distributionAxis = null;

  let exactPair = fallback;
  let playbackId = 0;

  function post(type, value = {}) {
    window.parent.postMessage({ type: prefix + type, ...value }, '*');
  }

  function isStep(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }

  function timelineEnd() {
    if (totalSteps !== null) return totalSteps - 1;
    return Math.max(1, playhead ?? 0, validRows.at(-1)?.start_step ?? 0);
  }

  function parseNumber(value) {
    const trimmed = value.trim();
    return /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) ? Number(trimmed) : NaN;
  }

  function parseStartStep(value) {
    const trimmed = value.trim();
    return /^[+-]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  }

  function rowToDraft(row) {
    return Object.fromEntries(scheduleKeys.map(([key]) => [key, String(row?.[key] ?? '')]));
  }

  function makeRows(focusIndex) {
    const body = get('rows');
    body.replaceChildren();
    rowElements = draft.map((row, index) => {
      const tr = body.insertRow();
      const inputs = Object.create(null);
      const sliders = Object.create(null);
      for (const [key, label, inputMode] of scheduleKeys) {
        const cell = tr.insertCell();
        if (readOnly) {
          cell.textContent = row[key];
        } else {
          const input = document.createElement('input');
          input.type = 'text';
          input.inputMode = inputMode;
          input.ariaLabel = label;
          input.value = row[key];
          input.addEventListener('input', () => {
            pause();
            row[key] = input.value;
            refreshSchedule();
          });
          const slider = document.createElement('input');
          slider.type = 'range';
          slider.step = key === 'start_step' ? '1' : '0.01';
          slider.ariaLabel = `Point ${index + 1} ${label}`;
          slider.addEventListener('input', () => {
            if (key === 'start_step') {
              const { min, max } = pointStepLimits(index);
              slider.value = String(Math.min(max, Math.max(min, slider.valueAsNumber)));
            }
            pause();
            row[key] = input.value = slider.value;
            refreshSchedule(slider, index);
          });
          slider.addEventListener('change', () => refreshSchedule());
          inputs[key] = input;
          sliders[key] = slider;
          cell.append(input, slider);
        }
      }
      if (!readOnly) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = 'Delete';
        button.ariaLabel = `Delete point ${index + 1}`;
        button.addEventListener('click', () => {
          pause();
          draft.splice(index, 1);
          makeRows(Math.min(index, draft.length - 1));
        });
        tr.insertCell().append(button);
      }
      return { tr, inputs, sliders };
    });
    get('empty').hidden = draft.length > 0;
    get('rows').closest('table').hidden = !draft.length;
    refreshSchedule();
    if (focusIndex !== undefined) {
      (rowElements[focusIndex]?.inputs.start_step ?? get('add')).focus();
    }
  }

  function pointStepLimits(index) {
    const previous = draft[index - 1];
    const next = draft[index + 1];
    return {
      min: previous ? parseStartStep(previous.start_step) + 1 : 0,
      max: next ? parseStartStep(next.start_step) - 1 : pointStepEnd,
    };
  }

  function refreshSliders(rows, activeSlider) {
    pointStepEnd = Math.max(pointStepEnd, timelineEnd(), validRows.at(-1)?.start_step ?? 0);
    for (const [index, row] of rows.entries()) {
      const { sliders } = rowElements[index];
      const start = sliders.start_step;
      const { min, max } = pointStepLimits(index);
      start.disabled = !isStep(min) || !isStep(max) || min >= max;
      start.min = '0';
      start.max = String(pointStepEnd);
      if (start !== activeSlider && isStep(row.start_step)) start.value = String(row.start_step);
      for (const [key, range] of Object.entries(parameterRanges)) {
        const slider = sliders[key];
        const value = row[key];
        if (slider === activeSlider || !Number.isFinite(value) || (key === 'scale' && value <= 0)) continue;
        slider.min = String(key === 'shift' ? Math.min(range.min, Math.floor(value)) : range.min);
        slider.max = String(Math.max(range.max, Math.ceil(value)));
        slider.value = String(value);
      }
    }
  }

  function refreshSchedule(activeSlider, activeRowIndex) {
    const rows = draft.map(draftRow => ({
      start_step: parseStartStep(draftRow.start_step),
      shift: parseNumber(draftRow.shift),
      scale: parseNumber(draftRow.scale),
    }));
    validationError = '';
    let previous = -1;
    for (const [index, row] of rows.entries()) {
      const errors = {
        start_step: !isStep(row.start_step)
          ? 'Start step must be a nonnegative whole number.'
          : row.start_step <= previous
            ? 'Start steps must be strictly increasing.'
            : '',
        shift: Number.isFinite(row.shift) ? '' : 'Shift must be finite.',
        scale: Number.isFinite(row.scale) && row.scale > 0 ? '' : 'Scale must be positive and finite.',
      };
      for (const [key, error] of Object.entries(errors)) {
        validationError ||= error ? `Point ${index + 1}. ${error}` : '';
        const input = rowElements[index].inputs[key];
        if (input) {
          input.setAttribute('aria-invalid', String(Boolean(error)));
          input.title = error;
        }
      }
      if (isStep(row.start_step)) previous = row.start_step;
    }
    const distributionChanged =
      !validationError &&
      (rows.length !== validRows.length ||
        (rows[0]?.start_step === 0) !== (validRows[0]?.start_step === 0) ||
        rows.some((row, index) => row.shift !== validRows[index].shift || row.scale !== validRows[index].scale));
    if (!validationError) validRows = rows;
    if (distributionChanged) distributionAxis = null;
    const errorStatus = get('schedule-error');
    const fixedGrid = Boolean(model?.fixed);
    const fixedGridError =
      fixedGrid && draft.length ? 'This model uses a fixed timestep grid. Remove all points to apply.' : '';
    errorStatus.textContent = validationError
      ? `${validationError} Chart shows the last valid schedule.`
      : fixedGridError;
    errorStatus.hidden = !errorStatus.textContent;
    get('apply').disabled = Boolean(validationError || fixedGridError);
    get('add').disabled = Boolean(validationError) || fixedGrid;
    get('add').title = fixedGrid
      ? 'This model uses a fixed timestep grid.'
      : 'Use the first available start step at or after the playhead.';

    if (activeSlider && !validationError) playhead = rows[activeRowIndex].start_step;
    refreshTimeline();
    if (!readOnly) refreshSliders(rows, activeSlider);
    refreshPair(distributionChanged);
  }

  function activeIndex(step = playhead ?? 0) {
    return validRows.findLastIndex(row => row.start_step <= step);
  }

  function scheduledPair(step, index) {
    const from = validRows[index];
    if (!from) return fallback;
    const to = validRows[index + 1];
    if (step === from.start_step || !to) return from;
    const fraction = (step - from.start_step) / (to.start_step - from.start_step);
    return interpolateSigmoid(from, to, fraction);
  }

  function refreshTimeline() {
    const lastStep = timelineEnd();
    const scrubEnd = Math.max(lastStep, playhead ?? 0);
    stepInput.max = readOnly || totalSteps === null ? '' : String(lastStep);
    scrub.max = String(scrubEnd);
    scrub.disabled = readOnly || !scrubEnd;
    get('play').disabled = !lastStep;
    const markers = get('markers');
    markers.replaceChildren();
    if (scrubEnd > 0) {
      for (const row of validRows) {
        if (row.start_step > scrubEnd) break;
        const marker = document.createElement('span');
        marker.style.left = `${(100 * row.start_step) / scrubEnd}%`;
        markers.append(marker);
      }
    }
    get('step-range').textContent =
      totalSteps === null ? 'Training length unknown' : `0–${formatStep.format(lastStep)}`;
  }

  function setPlayhead(step) {
    if (playhead === step) return;
    playhead = step;
    if (Math.max(timelineEnd(), step) !== Number(scrub.max)) {
      refreshTimeline();
      if (!readOnly && !validationError && totalSteps === null) refreshSliders(validRows);
    }
    refreshPair();
  }

  function canSplit(index) {
    return index >= 0 && index + 1 < validRows.length && playhead > validRows[index].start_step;
  }

  function refreshPair(redraw = false) {
    const step = playhead ?? 0;
    const index = activeIndex(step);
    const row = validRows[index];
    const between = row && step > row.start_step && index + 1 < validRows.length;
    const pair = scheduledPair(step, index);
    if (document.activeElement !== stepInput || readOnly || playbackId) stepInput.value = playhead ?? '';
    scrub.value = playhead ?? 0;
    get('active-source').textContent =
      index < 0
        ? 'Scalar fallback'
        : between
          ? `Between points ${index + 1} and ${index + 2}`
          : `Schedule point ${index + 1}`;
    get('active-shift').textContent = pair.shift;
    get('active-scale').textContent = pair.scale;
    get('following').textContent = playhead === null ? 'Awaiting training step' : 'Following training';
    get('split').disabled = Boolean(validationError || model?.fixed) || !canSplit(index);
    get('split').title = canSplit(index)
      ? 'Insert a point at the playhead with the current values.'
      : 'Move the playhead between two schedule points.';
    for (const [rowIndex, { tr }] of rowElements.entries()) {
      const used = index === rowIndex || (between && rowIndex === index + 1);
      tr.classList.toggle('current', used && !validationError && playhead !== null);
    }
    if (redraw || pair.shift !== exactPair.shift || pair.scale !== exactPair.scale || !profiles.length) {
      exactPair = pair;
      renderChart();
    }
  }

  function insertRow(split) {
    if (readOnly || validationError || model?.fixed) return;
    pause();
    const index = activeIndex();
    if (split && !canSplit(index)) return;
    let start = playhead;
    for (const row of validRows) {
      if (row.start_step > start) break;
      if (row.start_step === start) start++;
    }
    if (!isStep(start)) return;
    const next = validRows.findIndex(row => row.start_step > start);
    const insertAt = next < 0 ? draft.length : next;
    const { shift, scale } = scheduledPair(start, activeIndex(start));
    draft.splice(insertAt, 0, rowToDraft({ start_step: start, shift, scale }));
    makeRows(insertAt);
  }

  function pause() {
    cancelAnimationFrame(playbackId);
    playbackId = 0;
    get('play').textContent = 'Play';
  }

  function togglePlayback() {
    if (readOnly) return;
    if (playbackId) {
      pause();
      return;
    }
    const end = timelineEnd();
    if (!end) return;
    if (playhead >= end) setPlayhead(0);
    const startStep = playhead;
    const startTime = performance.now();
    get('play').textContent = 'Pause';
    function advance(now) {
      const elapsed = Math.max(0, now - startTime);
      const step = Math.min(end, startStep + Math.floor((end * elapsed) / 10000));
      setPlayhead(step);
      if (step < end) playbackId = requestAnimationFrame(advance);
      else pause();
    }
    playbackId = requestAnimationFrame(advance);
  }

  function applyBucket(bucket) {
    widthInput.value = bucket?.width ?? '';
    heightInput.value = bucket?.height ?? '';
    framesInput.value = bucket?.frames ?? '';
  }

  function bucketKey(bucket) {
    return `${bucket.width}:${bucket.height}:${model?.videoBucket ? (bucket.frames ?? '') : ''}`;
  }

  function updateBuckets(configured, preserveInput = false) {
    const incoming = (Array.isArray(configured) ? configured : []).filter(
      bucket =>
        bucket &&
        Number.isSafeInteger(bucket.width) &&
        bucket.width > 0 &&
        Number.isSafeInteger(bucket.height) &&
        bucket.height > 0,
    );
    const unique = [...new Map(incoming.map(bucket => [bucketKey(bucket), bucket])).values()];
    const previous = bucketSelect.value === '' ? null : buckets[Number(bucketSelect.value)];
    if (
      preserveInput &&
      unique.length === buckets.length &&
      unique.every((bucket, index) => bucketKey(bucket) === bucketKey(buckets[index]))
    )
      return false;
    buckets = unique;
    bucketSelect.replaceChildren(new Option('Enter dimensions', ''));
    for (const [index, bucket] of buckets.entries()) {
      const frames = model?.videoBucket && bucket.frames ? ` · ${bucket.frames} frames` : '';
      bucketSelect.add(new Option(`${bucket.width} × ${bucket.height}${frames}`, String(index)));
    }
    get('known-buckets').hidden = !buckets.length;
    const selectedIndex =
      preserveInput && previous ? buckets.findIndex(bucket => bucketKey(bucket) === bucketKey(previous)) : -1;
    bucketSelect.value = selectedIndex >= 0 ? String(selectedIndex) : '';
    const untouched = !widthInput.value && !heightInput.value && !framesInput.value;
    if (!preserveInput || untouched) {
      bucketSelect.value = buckets.length === 1 ? '0' : '';
      applyBucket(buckets.length === 1 ? buckets[0] : undefined);
    }
    return true;
  }

  function refreshReference() {
    const showModel = reference.value === 'model';
    get('bucket-controls').hidden = !showModel || !model?.dynamic;
    get('frames-control').hidden = !model?.videoBucket;
    get('stage-control').hidden = !showModel || !model?.boundary;
    get('bias-control').hidden = !showModel || !model || Boolean(model.fixed || model.unavailable);
    let profile = null;
    let info = '';
    let error = false;
    if (showModel) {
      try {
        if (!model)
          throw new Error(
            modelArch
              ? `Model Shift reference is unavailable for ${modelArch}.`
              : 'Select a model in the job to preview Model Shift.',
          );
        if (model.unavailable) throw new Error(`${model.label} has no diffusion timestep scheduler.`);
        let tokens;
        if (model.dynamic) {
          const bucket = { width: widthInput.valueAsNumber, height: heightInput.valueAsNumber };
          if (model.videoBucket) bucket.frames = framesInput.valueAsNumber;
          if (!widthInput.value || !heightInput.value || (model.videoBucket && !framesInput.value)) {
            throw new Error(
              model.videoBucket
                ? 'Enter the exact bucket width, height and frame count.'
                : 'Enter the exact bucket width and height.',
            );
          }
          ({ tokens } = resolveBucket(model, bucket));
          info = `Derived latent sequence length: ${formatStep.format(tokens)} tokens`;
        } else if (model.fixed) {
          info = 'This model uses a fixed timestep grid.';
        } else {
          info = `Static Shift coefficient: ${model.shift}`;
        }
        const grid = shiftGrid(model, tokens, stage.value);
        profile = {
          label: model.fixed ? `${model.label} · Fixed grid` : `${model.label} Shift · ${biasName[bias.value]}`,
          curve: gridCurve(grid, bias.value, Boolean(model.fixed)),
        };
      } catch (failure) {
        info = failure.message;
        error = true;
      }
    } else if (Object.hasOwn(biasName, reference.value)) {
      profile = {
        label: `Sigmoid (0, 1) · ${biasName[reference.value]}`,
        curve: sigmoidCurve(0, 1, reference.value),
      };
    }
    const infoElement = get('model-info');
    infoElement.textContent = info;
    infoElement.hidden = !info;
    infoElement.classList.toggle('error', error);
    refProfile = profile;
    distributionAxis = null;
    renderChart();
  }

  function seriesPoints(curve) {
    if (view.value === 'survival') {
      return Array.from({ length: 201 }, (_, i) => [5 * i, 1 - curve.cdf(5 * i)]);
    }
    if (view.value === 'quantiles') {
      return Array.from({ length: 101 }, (_, i) => [i, curve.quantile(1 - i / 100)]);
    }
    return distributionPoints(curve);
  }

  function scaleDistributionAxis(referencePoints) {
    const peak = Math.max(scheduleDistributionPeak(fallback, validRows), ...referencePoints.map(point => point[1]));
    const target = Math.min(1, 1.06 * peak);
    const unit = 10 ** Math.floor(Math.log10(target / 5));
    const tick = unit * [1, 2, 2.5, 5, 10].find(step => step * unit >= target / 5);
    const ticks = Math.ceil(target / tick);
    return { max: Math.min(1, ticks * tick), ticks };
  }

  function fillLegendAndStats() {
    const legend = get('legend');
    const stats = get('stats');
    legend.replaceChildren();
    stats.replaceChildren();
    for (const profile of profiles) {
      const item = document.createElement('span');
      const swatch = document.createElement('i');
      const row = stats.insertRow();
      swatch.style.setProperty('--series', `var(${profile.color})`);
      swatch.classList.toggle('dashed', profile.dashed);
      item.append(swatch, profile.label);
      legend.append(item);
      row.style.setProperty('--series', `var(${profile.color})`);
      const { curve } = profile;
      const values = [
        profile.label,
        Math.round(curve.quantile(0.5)),
        pct.format(curve.cdf(250)),
        pct.format(1 - curve.cdf(600)),
        pct.format(1 - curve.cdf(900)),
      ];
      for (const value of values) row.insertCell().textContent = value;
    }
  }

  function draw() {
    if (!profiles.length || !context) return;
    const { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const dpi = window.devicePixelRatio || 1;
    const pixelWidth = Math.round(width * dpi);
    const pixelHeight = Math.round(height * dpi);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    context.setTransform(dpi, 0, 0, dpi, 0, 0);
    context.clearRect(0, 0, width, height);
    const left = width < 400 ? 42 : 54;
    const top = 25;
    const right = width - 15;
    const bottom = height - 48;
    const compact = width < 490;
    const quantiles = view.value === 'quantiles';
    const distribution = view.value === 'distribution';
    const xMax = quantiles ? 100 : 1000;
    const yMax = distribution ? distributionAxis.max : quantiles ? 1000 : 1;
    const xPixel = value => left + (value * (right - left)) / xMax;
    const yPixel = value => bottom - (value * (bottom - top)) / yMax;
    plotBounds = { left, top, right, bottom, xMax };
    const styles = getComputedStyle(document.documentElement);
    context.font = '12px system-ui';
    context.fillStyle = styles.getPropertyValue('--muted').trim();
    context.textAlign = 'left';
    context.fillText(
      quantiles
        ? 'Timestep'
        : distribution
          ? `Probability per ${distributionWidth} timesteps`
          : 'Probability at or above threshold',
      left,
      13,
    );
    const yTicks = distribution ? distributionAxis.ticks : 5;
    context.strokeStyle = styles.getPropertyValue('--line').trim();
    context.lineWidth = 1;
    for (let i = 0; i <= yTicks; i++) {
      const v = (yMax * i) / yTicks;
      const y = yPixel(v);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
      context.textAlign = 'right';
      context.fillText(quantiles ? String(v) : pct.format(v), left - 7, y + 4);
    }
    const xTicks = compact ? 5 : 10;
    for (let i = 0; i <= xTicks; i++) {
      const v = (i * xMax) / xTicks;
      context.textAlign = i === xTicks ? 'right' : 'center';
      context.fillText(String(v), xPixel(v), bottom + 20);
    }
    context.textAlign = 'center';
    context.fillText(
      quantiles ? 'Percentile from highest noise (%)' : 'Timestep · 0 = clean · 1000 = noise',
      (right + left) / 2,
      height - 5,
    );
    for (const profile of [...profiles].reverse()) {
      context.beginPath();
      for (const [index, [x, y]] of profile.points.entries()) {
        if (index === 0) context.moveTo(xPixel(x), yPixel(y));
        else context.lineTo(xPixel(x), yPixel(y));
      }
      context.strokeStyle = styles.getPropertyValue(profile.color).trim();
      context.lineWidth = profile.dashed ? 2 : 3;
      context.setLineDash(profile.dashed ? [7, 4] : []);
      context.stroke();
    }
    context.setLineDash([]);
  }

  function renderChart() {
    const curve = sigmoidCurve(exactPair.shift, exactPair.scale, 'balanced');
    const custom = {
      label: 'Custom sigmoid · Balanced',
      curve,
      color: '--custom',
      dashed: false,
      points: seriesPoints(curve),
    };
    profiles = [custom];
    if (refProfile) {
      if (refProfile.view !== view.value) {
        refProfile.view = view.value;
        refProfile.points = seriesPoints(refProfile.curve);
      }
      profiles.push({ ...refProfile, color: '--reference', dashed: true });
    }
    if (view.value === 'distribution') {
      distributionAxis ??= scaleDistributionAxis(refProfile?.points ?? []);
      canvas.ariaLabel = `Training timestep distribution. Vertical scale 0 to ${pct.format(distributionAxis.max)}. Summary values follow in the table.`;
    } else {
      canvas.ariaLabel = `${chartTitles[view.value]}. Summary values follow in the table.`;
    }
    get('chart-heading').textContent = chartTitles[view.value];
    fillLegendAndStats();
    tooltip.hidden = true;
    draw();
  }

  function showTooltip(event) {
    if (!plotBounds || event.pointerType === 'touch') return;
    const { left, top, right, bottom, xMax } = plotBounds;
    if (event.offsetX < left || event.offsetX > right || event.offsetY < top || event.offsetY > bottom) {
      tooltip.hidden = true;
      return;
    }
    const x = Math.round((xMax * (event.offsetX - left)) / (right - left));
    const intervalCount = 1000 / distributionWidth;
    const interval = Math.min(intervalCount - 1, Math.floor(x / distributionWidth));
    const binLow = interval * distributionWidth;
    const binHigh = binLow + distributionWidth;
    const title = document.createElement('strong');
    title.textContent =
      view.value === 'quantiles'
        ? `${x}% from highest noise`
        : view.value === 'survival'
          ? `Timestep ≥ ${x}`
          : `${binLow}–${binHigh} timesteps`;
    tooltip.replaceChildren(title);
    for (const { label, curve, points } of profiles) {
      const line = document.createElement('div');
      let value;
      if (view.value === 'quantiles') value = Math.round(curve.quantile(1 - x / 100));
      else if (view.value === 'survival') value = pct.format(1 - curve.cdf(x));
      else value = pct.format(points[interval][1]);
      line.textContent = `${label} · ${value}`;
      tooltip.append(line);
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(4, Math.min(event.offsetX + 10, canvas.clientWidth - tooltip.offsetWidth - 4))}px`;
    tooltip.style.top = `${Math.max(0, Math.min(event.offsetY + 10, canvas.clientHeight - tooltip.offsetHeight))}px`;
  }

  function initialize(config) {
    pause();
    readOnly = config.readOnly === true;
    const train = config.train ?? {};
    totalSteps = isStep(train.steps) && train.steps > 0 ? train.steps : null;
    pointStepEnd = 0;
    const initialStep = isStep(config.step) ? config.step : null;
    playhead = readOnly ? initialStep : Math.min(initialStep ?? 0, totalSteps === null ? Infinity : totalSteps - 1);
    fallback = {
      shift: Number.isFinite(train.timestep_shift) ? train.timestep_shift : 0,
      scale: Number.isFinite(train.timestep_scale) && train.timestep_scale > 0 ? train.timestep_scale : 1,
    };
    draft = Array.isArray(train.timestep_sigmoid_schedule) ? train.timestep_sigmoid_schedule.map(rowToDraft) : [];
    validRows = [];
    distributionAxis = null;
    modelArch = typeof config.model?.arch === 'string' ? config.model.arch : '';
    const bareArch = modelArch.split(':')[0];
    const trainerArch = bareArch === 'flex1' ? 'flux' : bareArch;
    model = Object.hasOwn(models, modelArch)
      ? models[modelArch]
      : Object.hasOwn(models, trainerArch)
        ? models[trainerArch]
        : null;
    reference.value = '';
    refProfile = null;
    stage.value = 'high';
    bias.value = 'balanced';
    updateBuckets(config.buckets);
    const jobBias = train.content_or_style;
    const note = get('bias-note');
    note.hidden = !Object.hasOwn(biasName, jobBias) || jobBias === 'balanced';
    note.textContent = note.hidden
      ? ''
      : `This job uses the ${biasName[jobBias]} bias. The custom preview always uses Balanced.`;

    stepInput.readOnly = readOnly;
    get('play').hidden = readOnly;
    get('following').hidden = !readOnly;
    get('add').hidden = readOnly;
    get('split').hidden = readOnly;
    get('apply').closest('footer').hidden = readOnly;
    get('rows').closest('.schedule').toggleAttribute('data-readonly', readOnly);
    get('rows').closest('table').querySelector('th:last-child').hidden = readOnly;
    get('loading').hidden = true;
    get('editor').hidden = false;
    makeRows();
    refreshReference();
    initialized = true;
  }

  function commitStepInput() {
    if (readOnly) return;
    const step = stepInput.valueAsNumber;
    if (!isStep(step)) return;
    pause();
    setPlayhead(totalSteps === null ? step : Math.min(step, timelineEnd()));
  }

  stepInput.addEventListener('input', commitStepInput);
  stepInput.addEventListener('change', () => {
    commitStepInput();
    stepInput.value = playhead ?? '';
  });
  scrub.addEventListener('input', () => {
    if (readOnly) return;
    pause();
    setPlayhead(scrub.valueAsNumber);
  });
  get('play').addEventListener('click', togglePlayback);
  get('add').addEventListener('click', () => insertRow(false));
  get('split').addEventListener('click', () => insertRow(true));
  get('close').addEventListener('click', () => post('close'));
  get('apply').addEventListener('click', () => {
    if (initialized && !readOnly && !get('apply').disabled) post('apply', { schedule: validRows });
  });

  bucketSelect.addEventListener('change', () => {
    applyBucket(bucketSelect.value === '' ? undefined : buckets[Number(bucketSelect.value)]);
    refreshReference();
  });
  for (const input of [widthInput, heightInput, framesInput]) {
    input.addEventListener('input', () => {
      bucketSelect.value = '';
      refreshReference();
    });
  }
  for (const select of [reference, stage, bias]) select.addEventListener('change', refreshReference);
  view.addEventListener('change', renderChart);
  darkMode.addEventListener('change', draw);
  canvas.addEventListener('pointermove', showTooltip);
  canvas.addEventListener('pointerleave', () => {
    tooltip.hidden = true;
  });
  new ResizeObserver(draw).observe(canvas);

  window.addEventListener('keydown', event => {
    if (initialized && !readOnly && event.key === 'Escape') post('close');
  });
  window.addEventListener('message', event => {
    if (event.source !== window.parent) return;
    const message = event.data;
    if (message?.type === prefix + 'init' && message.context && typeof message.context === 'object') {
      initialize(message.context);
    } else if (message?.type === prefix + 'progress' && initialized) {
      if (Array.isArray(message.buckets) && updateBuckets(message.buckets, true)) refreshReference();
      if (readOnly && isStep(message.step)) setPlayhead(message.step);
    }
  });
  post('ready');
})();

const TimestepSampler = (() => {
  const scheduleSize = 1000;
  const distributionWidth = 5;
  const distributionCount = 1000 / distributionWidth;
  const biases = ['balanced', 'content', 'style'];
  const biasName = { balanced: 'Balanced', content: 'Content', style: 'Style' };

  function rankTails(bias, size = scheduleSize, fixed = false) {
    const width = size - 1;
    return Array.from({ length: size + 1 }, (_, k) => {
      if (k === size) return 0;
      if (fixed) return 1 - k / size;
      if (width === 0) return 1;
      if (bias === 'balanced') return 1 - k / width;
      const rank = (k * (scheduleSize - 1)) / (scheduleSize * width);
      return bias === 'content' ? 1 - Math.cbrt(rank) : Math.cbrt(1 - rank);
    });
  }

  const tails = Object.fromEntries(biases.map(bias => [bias, rankTails(bias)]));

  function normalCDF(x) {
    const a = Math.abs(x) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * a);
    const tail =
      0.5 *
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-a * a);
    return x < 0 ? tail : 1 - tail;
  }

  function sigmoidCDF(timestep, shift, scale, rankTail) {
    if (timestep <= 0) return 0;
    if (timestep >= 1000) return 1;
    // Count schedule entries above this timestep. Select a rank at or below
    // that threshold by averaging rankTail over the binomial count.
    const p = normalCDF((Math.log((1000 - timestep) / timestep) - shift) / scale);
    if (!rankTail) {
      // Balanced rank selection is uniform over all ranks except the last one.
      // For K ~ Binomial(n,p), E[max(n-1-K,0)] = n-1 - np + p^n.
      const q = 1 - p;
      return Math.max(0, (scheduleSize * q + Math.expm1(scheduleSize * Math.log1p(-q))) / (scheduleSize - 1));
    }
    if (p <= 0 || p >= 1) return rankTail[p <= 0 ? 0 : scheduleSize];
    const mode = Math.min(scheduleSize, Math.floor((scheduleSize + 1) * p));
    const odds = p / (1 - p);
    let mass = 1;
    let weighted = rankTail[mode];
    let weight = 1;
    for (let k = mode; k > 0; k--) {
      weight *= k / ((scheduleSize - k + 1) * odds);
      mass += weight;
      weighted += weight * rankTail[k - 1];
      if (weight < 1e-16) break;
    }
    weight = 1;
    for (let k = mode; k < scheduleSize; k++) {
      weight *= ((scheduleSize - k) * odds) / (k + 1);
      mass += weight;
      weighted += weight * rankTail[k + 1];
      if (weight < 1e-16) break;
    }
    return weighted / mass;
  }

  function sigmoidCurve(shift, scale, bias) {
    const cdf = timestep => sigmoidCDF(timestep, shift, scale, bias === 'balanced' ? null : tails[bias]);
    return {
      cdf,
      quantile(p) {
        if (p === 0 || p === 1) return 1000 * p;
        let lo = 0;
        let hi = 1000;
        for (let i = 0; i < 18; i++) {
          const mid = (lo + hi) / 2;
          if (cdf(mid) < p) lo = mid;
          else hi = mid;
        }
        return (lo + hi) / 2;
      },
    };
  }

  function interpolateSigmoid(from, to, fraction) {
    return {
      shift: (1 - fraction) * from.shift + fraction * to.shift,
      scale:
        from.scale <= to.scale
          ? from.scale + fraction * (to.scale - from.scale)
          : to.scale + (1 - fraction) * (from.scale - to.scale),
    };
  }

  function distributionCdfs(curve) {
    const edges = new Float64Array(distributionCount + 1);
    for (let i = 0; i < distributionCount; i++) edges[i] = curve.cdf(i * distributionWidth);
    edges[distributionCount] = 1; // Include draws exactly at timestep 1000 in the final interval.
    return edges;
  }

  function distributionPoints(curve) {
    const edges = distributionCdfs(curve);
    return Array.from({ length: distributionCount }, (_, i) => [
      (i + 0.5) * distributionWidth,
      Math.max(0, edges[i + 1] - edges[i]),
    ]);
  }

  function distributionUpperBound(left, right) {
    let peak = 0;
    for (let i = 0; i < distributionCount; i++) {
      peak = Math.max(peak, Math.max(left[i + 1], right[i + 1]) - Math.min(left[i], right[i]));
    }
    return peak;
  }

  function scheduleDistributionPeak(fallback, points) {
    const cdfs = ({ shift, scale }) => distributionCdfs(sigmoidCurve(shift, scale, 'balanced'));
    const ends = points.map(cdfs);
    let peak = 0;
    if (!points.length || points[0].start_step > 0) {
      const edges = cdfs(fallback);
      peak = distributionUpperBound(edges, edges);
    }
    for (const edges of ends) peak = Math.max(peak, distributionUpperBound(edges, edges));
    let ceiling = peak;
    for (let i = 1; i < points.length; i++) {
      const from = points[i - 1];
      const to = points[i];
      // At a fixed boundary, (logit - shift) / scale is monotone through
      // a segment. Endpoint CDFs therefore bound every interior interval mass.
      const interval = (low, high, left, right) => ({
        low,
        high,
        left,
        right,
        bound: distributionUpperBound(left, right),
      });
      const pending = [interval(0, 1, ends[i - 1], ends[i])];
      let splits = 0;
      while (pending.length) {
        // Refine the widest remaining peak bound first so the work cap
        // cannot leave a distant part of the segment with a loose bound.
        const { low, high, left, right, bound } = pending.sort((a, b) => a.bound - b.bound).pop();
        if (bound <= peak + 0.00025 || splits >= 256) {
          ceiling = Math.max(ceiling, bound);
          break;
        }
        if (high - low <= 2 ** -18) {
          ceiling = Math.max(ceiling, bound);
          continue;
        }
        const middle = (low + high) / 2;
        const mid = cdfs(interpolateSigmoid(from, to, middle));
        peak = Math.max(peak, distributionUpperBound(mid, mid));
        splits++;
        pending.push(interval(low, middle, left, mid), interval(middle, high, mid, right));
      }
    }
    return Math.max(ceiling, peak);
  }

  const flow = (shift, dynamic = null, terminal = null) => ({ shift, dynamic, terminal });
  const spatial = (maxLength = 4096, max = 1.15) => ({ baseLength: 256, maxLength, base: 0.5, max });
  const qwen = flow(1, spatial(8192, 0.9), 0.02);
  const flux = flow(3, spatial());
  const ltx = flow(1, { baseLength: 1024, maxLength: 4096, base: 0.95, max: 2.05 }, 0.1);
  const fixedH3 = { fixed: [1, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132] };
  const models = {};

  function imageBucket(spatialCompression, patchSize, alignment = spatialCompression * patchSize) {
    return { spatialCompression, patchSize, alignment };
  }

  function add(group, schedule, names, bucket) {
    for (const [arch, label] of names) {
      models[arch] = { arch, label, group, ...schedule, bucket, videoBucket: Boolean(bucket?.temporalCompression) };
    }
  }

  // Training schedules from AI Toolkit through 4a7f285. Use the VAE's actual
  // output shape and the patch size passed by BaseSDTrainProcess on Shift.
  add('Image', flow(1), [
    ['ideogram4', 'Ideogram 4'],
    ['nucleus_image', 'Nucleus Image'],
    ['omnigen2', 'OmniGen2'],
  ]);
  add('Image', flow(3), [
    ['anima', 'Anima'],
    ['boogu_image', 'Boogu Image'],
    ['boogu_image_edit', 'Boogu Image Edit'],
    ['ernie_image', 'ERNIE Image'],
    ['hidream', 'HiDream I1'],
    ['hidream_e1', 'HiDream E1'],
    ['hidream_o1', 'HiDream O1'],
    ['prx_pixel', 'PRX Pixel'],
    ['zeta_chroma', 'Zeta Chroma'],
    ['zimage', 'Z-Image'],
    ['zimage_l2p', 'Z-Image L2P'],
  ]);
  add('Image', flow(4), [['iris3b', 'Iris 3B']]);
  add('Image', flow(5), [
    ['moss_image:base', 'MOSS Image Base'],
    ['moss_image', 'MOSS Image SFT'],
    ['moss_image:base_edit', 'MOSS Image Base Edit'],
    ['moss_image:edit', 'MOSS Image SFT Edit'],
  ]);
  add('Image', flow(6), [
    ['mageflow', 'MageFlow'],
    ['mageflow_edit', 'MageFlow Edit'],
    ['lumina2', 'Lumina Image 2'],
  ]);
  add('Image', flow(6.93), [['flux3', 'FLUX.3']]);
  // Chroma and Chroma Radiance fall back to patch size 1. Radiance's
  // identity VAE passes the pixel grid through to the scheduler.
  add('Image', flux, [['chroma', 'Chroma']], imageBucket(8, 1, 32));
  add('Image', flux, [['chroma_radiance', 'Chroma Radiance']], imageBucket(1, 1, 32));
  add(
    'Image',
    flux,
    [
      ['f-lite', 'F-Lite'],
      ['flex2', 'Flex.2'],
    ],
    imageBucket(8, 2),
  );
  add('Image', flux, [['flux', 'FLUX.1 / Flex.1']], imageBucket(8, 2, 32));
  add('Image', flux, [['flux_kontext', 'FLUX.1 Kontext']], imageBucket(8, 1, 16));
  // The FLUX.2 family and Fibo 2 already pack 2×2 positions in the encoder.
  add(
    'Image',
    flux,
    [
      ['fibo2', 'Fibo 2'],
      ['flux2', 'FLUX.2 Dev'],
      ['flux2_klein_4b', 'FLUX.2 Klein 4B'],
      ['flux2_klein_9b', 'FLUX.2 Klein 9B'],
    ],
    imageBucket(16, 1),
  );
  add('Image', flow(1, spatial(6400)), [['krea2', 'Krea 2']], imageBucket(8, 2));
  add('Image', flow(6, { ...spatial(), large: 1.35 }), [['ming_image', 'Ming Image']], imageBucket(8, 2));
  add(
    'Image',
    flow(1, { baseLength: 256, maxLength: 4096, base: 0.25, max: 0.75, linear: true }),
    [['cogview4', 'CogView4']],
    imageBucket(8, 2),
  );
  add(
    'Image',
    qwen,
    [
      ['qwen_image', 'Qwen Image'],
      ['qwen_image_edit', 'Qwen Image Edit'],
      ['qwen_image_edit_plus', 'Qwen Image Edit Plus'],
    ],
    imageBucket(8, 2, 32),
  );
  add('Image', qwen, [['qwen_image_2', 'Qwen Image 2 / 2.1']], imageBucket(16, 1, 32));
  add('Image', fixedH3, [['minimax_h3_image', 'MiniMax H3 Image']]);
  add('Image', flow(12), [['p_h3_image', 'P-H3 Image']]);
  add('Video', flow(3), [
    ['wan21', 'Wan 2.1'],
    ['wan21_i2v', 'Wan 2.1 Image to Video'],
  ]);
  add('Video', flow(5), [['wan22_5b', 'Wan 2.2 5B']]);
  add('Video', { ...flow(5), boundary: 0.875 }, [
    ['wan22_14b', 'Wan 2.2 14B'],
    ['wan22_14b_i2v', 'Wan 2.2 14B Image to Video'],
  ]);
  add(
    'Video',
    ltx,
    [
      ['ltx2', 'LTX-2'],
      ['ltx2.3', 'LTX-2.3'],
      ['ltx2.5', 'LTX-2.5'],
    ],
    { spatialCompression: 32, temporalCompression: 8, patchSize: 1, alignment: 32 },
  );
  add('Video', flow(12), [
    ['minimax_h3', 'MiniMax H3'],
    ['minimax_h3_ref2va', 'MiniMax H3 Ref2VA'],
    ['minimax_h3_omni', 'MiniMax H3 Omni'],
  ]);
  add('Video', flow(10), [['minimax_h3_vsa', 'MiniMax H3 Fast / VSA']]);
  add('Audio', flow(3), [
    ['ace_step_15', 'ACE-Step 1.5'],
    ['ace_step_15_xl', 'ACE-Step 1.5 XL'],
  ]);
  add('Audio', flow(1), [['yue2', 'YuE 2']]);
  add('Legacy', { ...flow(3), legacy: true }, [
    ['sd1', 'Stable Diffusion 1'],
    ['sd2', 'Stable Diffusion 2'],
    ['sd3', 'Stable Diffusion 3 / 3.5'],
    ['sdxl', 'SDXL'],
    ['auraflow', 'AuraFlow'],
    ['vega', 'Vega'],
    ['ssd', 'SSD'],
    ['pixart_sigma', 'PixArt Sigma'],
  ]);
  add('Legacy', { ...flux, legacy: true }, [['pixart', 'PixArt']], imageBucket(8, 2));
  add('Language', { unavailable: true }, [
    ['qwen25_omni', 'Qwen 2.5 Omni'],
    ['qwen38', 'Qwen 3.8'],
  ]);

  function dynamicCoefficient(config, tokens) {
    const { baseLength, maxLength, base, max, linear, large } = config;
    const mu =
      large && tokens >= maxLength ? large : base + ((tokens - baseLength) * (max - base)) / (maxLength - baseLength);
    return linear ? mu : Math.exp(mu);
  }

  function resolveBucket(model, { width, height, frames } = {}) {
    if (!model.dynamic) return { tokens: null };
    if (!model.bucket) throw new Error(`Training bucket geometry is unavailable for ${model.label}.`);

    const { spatialCompression, temporalCompression, patchSize, alignment } = model.bucket;
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
      throw new Error('Enter bucket width and height as positive whole numbers.');
    }
    if (width % alignment !== 0 || height % alignment !== 0) {
      throw new Error(`${model.label} bucket dimensions must be multiples of ${alignment}.`);
    }

    const latentWidth = width / spatialCompression;
    const latentHeight = height / spatialCompression;
    let latentFrames;
    if (temporalCompression) {
      if (!Number.isSafeInteger(frames) || frames < 1 || (frames - 1) % temporalCompression !== 0) {
        throw new Error(
          `Enter a valid ${model.label} frame count (1, ${1 + temporalCompression}, ${1 + 2 * temporalCompression}, …).`,
        );
      }
      latentFrames = (frames - 1) / temporalCompression + 1;
    }

    // CustomFlowMatchEulerDiscreteScheduler multiplies latent axes 2 and 3.
    // For LTX's [B, C, F, H, W] those axes are frames and height.
    const tokens = Math.floor(((latentFrames ?? latentWidth) * latentHeight) / patchSize ** 2);
    if (!Number.isSafeInteger(tokens) || !Number.isFinite(dynamicCoefficient(model.dynamic, tokens))) {
      throw new Error('This bucket is too large for the training scheduler.');
    }
    return latentFrames === undefined
      ? { tokens, latentWidth, latentHeight }
      : { tokens, latentWidth, latentHeight, latentFrames };
  }

  function shiftGrid(model, tokens, stage = 'high') {
    const f = Math.fround;
    if (model.fixed) return model.fixed.map(sigma => f(f(sigma) * scheduleSize));
    const endpoint = sigma =>
      model.dynamic ? sigma : f(f(f(model.shift) * sigma) / f(1 + f(f(model.shift - 1) * sigma)));
    const min = endpoint(f(1 / scheduleSize));
    const max = endpoint(1);
    const coefficient = model.dynamic ? dynamicCoefficient(model.dynamic, tokens) : model.shift;
    const sigmas = Array.from({ length: scheduleSize }, (_, i) => {
      const u =
        (max * scheduleSize + ((min * scheduleSize - max * scheduleSize) / (scheduleSize - 1)) * i) / scheduleSize;
      return model.dynamic
        ? coefficient / (coefficient + (1 / u - 1))
        : (coefficient * u) / (1 + (coefficient - 1) * u);
    });
    const terminalScale = model.terminal ? (1 - sigmas.at(-1)) / (1 - model.terminal) : null;
    const grid = sigmas.map(sigma => f(f(terminalScale ? 1 - (1 - sigma) / terminalScale : sigma) * scheduleSize));
    if (!model.boundary) return grid;
    const edge = model.boundary * scheduleSize;
    // Match BaseSDTrainProcess multistage lookup, including its left/right boundary rules.
    const highCount = grid.filter(t => t > edge).length;
    const lowStart = grid.filter(t => t >= edge).length - 1;
    return stage === 'high' ? grid.slice(0, highCount) : grid.slice(lowStart);
  }

  function gridCurve(grid, bias, fixed = false) {
    const rankTail = rankTails(bias, grid.length, fixed);
    return {
      cdf(t) {
        let lo = 0;
        let hi = grid.length;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (grid[mid] >= t) lo = mid + 1;
          else hi = mid;
        }
        return rankTail[lo];
      },
      quantile(p) {
        let k = grid.length - 1;
        while (rankTail[k] < p || rankTail[k] === 0) k--;
        return grid[k];
      },
    };
  }

  return {
    biases,
    biasName,
    models,
    sigmoidCurve,
    interpolateSigmoid,
    distributionWidth,
    distributionPoints,
    scheduleDistributionPeak,
    shiftGrid,
    gridCurve,
    dynamicCoefficient,
    resolveBucket,
  };
})();

// Client side of the inference engine stream protocol
// (extensions_built_in/inference_engine/protocol.py):
//   u32 header_len | json header | u64 payload_len | payload
// plus the latent -> RGB preview projection driven entirely by the `preview`
// block the engine sends in its `start` frame (no per-arch code here).

export interface EngineFrame {
  header: any;
  payload: Uint8Array;
}

export interface PreviewInfo {
  format: string;
  channels: number;
  dims: number;
  spatial: number;
  temporal: number;
  reshape: string | null;
  factors: number[][];
  bias: number[] | null;
}

/** Read frames from a fetch Response body, calling onFrame for each one. */
export async function readEngineFrames(
  response: Response,
  onFrame: (frame: EngineFrame) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!response.body) throw new Error('response has no body');
  const reader = response.body.getReader();
  let buf = new Uint8Array(0);

  const append = (chunk: Uint8Array) => {
    const next = new Uint8Array(buf.length + chunk.length);
    next.set(buf, 0);
    next.set(chunk, buf.length);
    buf = next;
  };

  const parse = () => {
    let offset = 0;
    while (true) {
      if (buf.length - offset < 4) break;
      const view = new DataView(buf.buffer, buf.byteOffset + offset);
      const hlen = view.getUint32(0, true);
      if (buf.length - offset < 4 + hlen + 8) break;
      const headerBytes = buf.subarray(offset + 4, offset + 4 + hlen);
      const lenView = new DataView(buf.buffer, buf.byteOffset + offset + 4 + hlen);
      // payloads are far below 2^53, so the low/high split is safe
      const plen = lenView.getUint32(0, true) + lenView.getUint32(4, true) * 2 ** 32;
      const start = offset + 4 + hlen + 8;
      if (buf.length < start + plen) break;
      const header = JSON.parse(new TextDecoder().decode(headerBytes));
      const payload = buf.slice(start, start + plen);
      offset = start + plen;
      onFrame({ header, payload });
    }
    if (offset > 0) buf = buf.slice(offset);
  };

  while (true) {
    if (signal?.aborted) {
      await reader.cancel();
      return;
    }
    const { value, done } = await reader.read();
    if (done) break;
    if (value && value.length) {
      append(value);
      parse();
    }
  }
}

/** Decode a `latent` frame payload to Float32Array (engine sends float16 or float32). */
export function payloadToFloat32(header: any, payload: Uint8Array): Float32Array {
  const dtype: string = header.dtype;
  const n = header.shape.reduce((a: number, b: number) => a * b, 1);
  if (dtype === 'float32') {
    return new Float32Array(payload.buffer.slice(payload.byteOffset, payload.byteOffset + n * 4));
  }
  if (dtype === 'float16') {
    const out = new Float32Array(n);
    const view = new DataView(payload.buffer, payload.byteOffset, n * 2);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(view.getUint16(i * 2, true));
    return out;
  }
  throw new Error(`unsupported latent dtype ${dtype}`);
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export interface LatentImage {
  width: number;
  height: number;
  frames: number;
  /** RGBA pixels per frame */
  frameData: Uint8ClampedArray[];
}

/**
 * Project a latent tensor to RGB frames. Layout BCHW -> 1 frame, BCFHW -> F
 * frames (batch item 0). With a preview table the projection is the linear
 * per-pixel (channels x 3) map; without one, three channels are stretched to
 * a grayscale-ish false color so the user still sees denoising progress.
 */
export function latentToImage(header: any, data: Float32Array, preview: PreviewInfo | null): LatentImage | null {
  const shape: number[] = header.shape;
  const layout: string = header.layout;
  let C: number, F: number, H: number, W: number;
  if (layout === 'BCHW' && shape.length === 4) {
    [, C, H, W] = shape;
    F = 1;
  } else if (layout === 'BCFHW' && shape.length === 5) {
    [, C, F, H, W] = shape;
  } else if (layout === 'BLC' && shape.length === 3) {
    // sequence latents (audio): show as a channel x time heatmap
    const [, L, Cc] = shape;
    const px = new Uint8ClampedArray(L * Cc * 4);
    let max = 1e-6;
    for (let i = 0; i < L * Cc; i++) max = Math.max(max, Math.abs(data[i]));
    for (let t = 0; t < L; t++) {
      for (let c = 0; c < Cc; c++) {
        const v = data[t * Cc + c] / max; // -1..1
        const o = (c * L + t) * 4;
        px[o] = 128 + 127 * Math.max(0, v);
        px[o + 1] = 128 - 127 * Math.abs(v) * 0.5;
        px[o + 2] = 128 + 127 * Math.max(0, -v);
        px[o + 3] = 255;
      }
    }
    return { width: L, height: Cc, frames: 1, frameData: [px] };
  } else {
    return null;
  }

  let factors: number[][] | null = null;
  let bias: number[] = [0, 0, 0];
  let reshape: string | null = null;
  if (preview && preview.factors && preview.channels === C) {
    factors = preview.factors;
    bias = preview.bias || [0, 0, 0];
    reshape = preview.reshape;
  } else if (preview && preview.reshape === 'flux2_2x2' && C === 128 && preview.factors) {
    factors = preview.factors;
    bias = preview.bias || [0, 0, 0];
    reshape = 'flux2_2x2';
  }

  const frameSize = H * W;
  const chanStride = F * frameSize; // BCFHW: c, f, h, w
  const frameData: Uint8ClampedArray[] = [];
  let outW = W;
  let outH = H;
  if (reshape === 'flux2_2x2') {
    outW = W * 2;
    outH = H * 2;
  }

  for (let f = 0; f < F; f++) {
    const px = new Uint8ClampedArray(outW * outH * 4);
    if (factors) {
      if (reshape === 'flux2_2x2') {
        // 128 channels = 32 x (2x2 pixel shuffle): channel index c*4 + dy*2 + dx
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const base = y * W + x;
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                let r = bias[0], g = bias[1], b = bias[2];
                for (let c = 0; c < 32; c++) {
                  const ch = c * 4 + dy * 2 + dx;
                  const v = data[ch * chanStride + f * frameSize + base];
                  r += v * factors[c][0];
                  g += v * factors[c][1];
                  b += v * factors[c][2];
                }
                const o = ((y * 2 + dy) * outW + (x * 2 + dx)) * 4;
                px[o] = (r + 1) * 127.5;
                px[o + 1] = (g + 1) * 127.5;
                px[o + 2] = (b + 1) * 127.5;
                px[o + 3] = 255;
              }
            }
          }
        }
      } else {
        for (let i = 0; i < frameSize; i++) {
          let r = bias[0], g = bias[1], b = bias[2];
          for (let c = 0; c < C; c++) {
            const v = data[c * chanStride + f * frameSize + i];
            r += v * factors[c][0];
            g += v * factors[c][1];
            b += v * factors[c][2];
          }
          const o = i * 4;
          px[o] = (r + 1) * 127.5;
          px[o + 1] = (g + 1) * 127.5;
          px[o + 2] = (b + 1) * 127.5;
          px[o + 3] = 255;
        }
      }
    } else {
      // unknown family: normalize the first three channels
      const chans = [0, Math.min(1, C - 1), Math.min(2, C - 1)];
      const stats = chans.map(c => {
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < frameSize; i++) {
          const v = data[c * chanStride + f * frameSize + i];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        return { mn, range: Math.max(1e-6, mx - mn) };
      });
      for (let i = 0; i < frameSize; i++) {
        const o = i * 4;
        for (let k = 0; k < 3; k++) {
          const v = data[chans[k] * chanStride + f * frameSize + i];
          px[o + k] = ((v - stats[k].mn) / stats[k].range) * 255;
        }
        px[o + 3] = 255;
      }
    }
    frameData.push(px);
  }
  return { width: outW, height: outH, frames: F, frameData };
}

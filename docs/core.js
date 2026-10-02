// How Missoni Is This? Core logic shared by the web page (docs/app.js) and the
// reference builder (tools/reference_builder.html), so visitors' photos are
// scored by exactly the same code and runtime that built the reference.
//
// Mirrors src/how_missoni.py: segment the garment, place it on a neutral
// background, embed with CLIP, compare with the Missoni reference.

export const SEG_MODEL = 'Xenova/segformer_b2_clothes';
export const CLIP_MODEL = 'Xenova/clip-vit-base-patch32';
export const DTYPE = 'q8'; // quantized models: ~29 MB + ~89 MB, small enough for a browser
export const GARMENT_LABELS = new Set(['Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', 'Scarf']);
export const NEUTRAL = [123, 117, 104]; // CLIP's mean colour, so the background reads as "nothing"
export const TOP_K = 5;
export const DUPLICATE_SIMILARITY = 0.98;
export const MIN_GARMENT_SHARE = 0.01;
export const CLIP_SIZE = 224;
export const MAX_SIDE = 1024; // phone photos are huge; this keeps the browser fast

/** Load both models. `T` is the transformers.js module (CDN in the browser, npm in Node). */
export async function loadModels(T, progress_callback) {
  // WebAssembly everywhere: quantized models give different numbers on other
  // backends (WebGPU, Node), and the reference was built with this one.
  const opts = { dtype: DTYPE, device: 'wasm', progress_callback };
  const [segmenter, processor, vision] = await Promise.all([
    T.pipeline('image-segmentation', SEG_MODEL, opts),
    T.AutoProcessor.from_pretrained(CLIP_MODEL, { progress_callback }),
    T.CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, opts),
  ]);
  return { segmenter, processor, vision };
}

/**
 * Cut out the garment. Returns { found, share, image, pixels }:
 * image = garment on a neutral square (RawImage) for CLIP,
 * pixels = sampled garment RGB values for the palette.
 */
export async function isolateGarment(T, models, image) {
  image = image.rgb();
  const longest = Math.max(image.width, image.height);
  if (longest > MAX_SIDE) {
    const f = MAX_SIDE / longest;
    image = resize(T, image, Math.round(image.width * f), Math.round(image.height * f));
  }
  const { width: w, height: h, data } = image;
  const segments = await models.segmenter(image);
  const mask = new Uint8Array(w * h);
  for (const s of segments) {
    if (!GARMENT_LABELS.has(s.label)) continue;
    const m = s.mask.data;
    for (let i = 0; i < m.length; i++) if (m[i]) mask[i] = 1;
  }
  let n = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    n++;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const share = n / (w * h);
  if (share < MIN_GARMENT_SHARE) return { found: false, share, image, pixels: samplePixels(data, null, 4000) };

  const cw = x1 - x0 + 1, ch = y1 - y0 + 1, side = Math.max(cw, ch);
  const ox = Math.floor((side - cw) / 2), oy = Math.floor((side - ch) / 2);
  const out = new Uint8ClampedArray(side * side * 3);
  for (let i = 0; i < side * side; i++) out.set(NEUTRAL, i * 3);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const src = (y + y0) * w + (x + x0);
    if (!mask[src]) continue;
    const dst = ((y + oy) * side + (x + ox)) * 3;
    out[dst] = data[src * 3]; out[dst + 1] = data[src * 3 + 1]; out[dst + 2] = data[src * 3 + 2];
  }
  return { found: true, share, image: new T.RawImage(out, side, side, 3), pixels: samplePixels(data, mask, 4000) };
}

function samplePixels(data, mask, max) {
  const idx = [];
  const total = data.length / 3;
  for (let i = 0; i < total; i++) if (!mask || mask[i]) idx.push(i);
  const step = Math.max(1, Math.floor(idx.length / max));
  const px = [];
  for (let j = 0; j < idx.length; j += step) {
    const i = idx[j] * 3;
    px.push([data[i], data[i + 1], data[i + 2]]);
  }
  return px;
}

/**
 * Resize an RGB RawImage in plain JavaScript: area averaging when shrinking,
 * bilinear when enlarging. Node and browsers resize images differently, which
 * changes fine knit patterns; doing it here keeps both environments identical.
 */
export function resize(T, image, w, h) {
  const { width: W, height: H, data } = image;
  const out = new Uint8ClampedArray(w * h * 3);
  const sx = W / w, sy = H / h;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 3;
    if (sx >= 1 && sy >= 1) {
      const xa = Math.floor(x * sx), xb = Math.max(xa + 1, Math.floor((x + 1) * sx));
      const ya = Math.floor(y * sy), yb = Math.max(ya + 1, Math.floor((y + 1) * sy));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = ya; yy < yb; yy++) for (let xx = xa; xx < xb; xx++) {
        const i = (yy * W + xx) * 3; r += data[i]; g += data[i + 1]; b += data[i + 2]; n++;
      }
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n;
    } else {
      const fx = Math.min(W - 1, Math.max(0, (x + 0.5) * sx - 0.5)), fy = Math.min(H - 1, Math.max(0, (y + 0.5) * sy - 0.5));
      const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
      const ax = fx - x0, ay = fy - y0;
      for (let c = 0; c < 3; c++) {
        const p = (yy, xx) => data[(yy * W + xx) * 3 + c];
        out[o + c] = (p(y0, x0) * (1 - ax) + p(y0, x1) * ax) * (1 - ay) + (p(y1, x0) * (1 - ax) + p(y1, x1) * ax) * ay;
      }
    }
  }
  return new T.RawImage(out, w, h, 3);
}

/** CLIP image embedding, unit length. The garment square is brought to CLIP's 224 px here, not by the processor. */
export async function embed(T, models, rawImage) {
  if (rawImage.width !== CLIP_SIZE || rawImage.height !== CLIP_SIZE) rawImage = resize(T, rawImage, CLIP_SIZE, CLIP_SIZE);
  const inputs = await models.processor(rawImage);
  const { image_embeds } = await models.vision(inputs);
  const e = Float32Array.from(image_embeds.data);
  const norm = Math.hypot(...e);
  return e.map((v) => v / norm);
}

/**
 * Score an embedding against the reference ({ looks, embeddings, calibration }).
 * 50 = as Missoni as a typical Missoni runway look (calibration: each archive
 * look scored with its own collection hidden).
 */
export function score(ref, e) {
  const sims = ref.embeddings.map((r, i) => {
    let s = 0;
    for (let d = 0; d < r.length; d++) s += r[d] * e[d];
    return { i, s };
  }).filter((x) => x.s < DUPLICATE_SIMILARITY);
  sims.sort((a, b) => b.s - a.s);
  const top = sims.slice(0, TOP_K);
  const similarity = top.reduce((a, x) => a + x.s, 0) / top.length;
  const scoreValue = (100 * ref.calibration.filter((c) => c <= similarity).length) / ref.calibration.length;
  const votes = {};
  for (const { i, s } of top) votes[ref.looks[i].collection] = (votes[ref.looks[i].collection] || 0) + s;
  const closest = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
  return {
    score: scoreValue, similarity, closest,
    neighbours: top.map(({ i, s }) => ({ ...ref.looks[i], similarity: s })),
  };
}

export function verdict(score, found = true) {
  if (!found) return 'No garment detected';
  if (score >= 60) return 'Very Missoni';
  if (score >= 30) return 'Missoni-adjacent';
  if (score >= 10) return 'A hint of Missoni';
  return 'Not very Missoni';
}

/** k-means palette of garment pixels -> [{ hex, share }], largest first. Deterministic. */
export function palette(pixels, k = 5, iterations = 12) {
  if (!pixels.length) return [];
  k = Math.min(k, pixels.length);
  const centres = [pixels[0]];
  while (centres.length < k) {
    // Farthest-point initialisation: stable and spreads the swatches.
    let best = pixels[0], bestD = -1;
    for (const p of pixels) {
      const d = Math.min(...centres.map((c) => dist2(p, c)));
      if (d > bestD) { bestD = d; best = p; }
    }
    centres.push(best);
  }
  let labels = new Array(pixels.length).fill(0);
  for (let it = 0; it < iterations; it++) {
    labels = pixels.map((p) => argmin(centres.map((c) => dist2(p, c))));
    for (let c = 0; c < k; c++) {
      const members = pixels.filter((_, i) => labels[i] === c);
      if (members.length) centres[c] = [0, 1, 2].map((j) => members.reduce((a, p) => a + p[j], 0) / members.length);
    }
  }
  return centres
    .map((c, ci) => ({ hex: toHex(c), share: labels.filter((l) => l === ci).length / labels.length }))
    .filter((s) => s.share > 0)
    .sort((a, b) => b.share - a.share);
}

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
const argmin = (xs) => xs.reduce((bi, x, i) => (x < xs[bi] ? i : bi), 0);
const toHex = (c) => '#' + c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('').toUpperCase();

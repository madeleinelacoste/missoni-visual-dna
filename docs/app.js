// Page logic for How Missoni Is This? The scoring itself lives in core.js,
// shared with the Node scripts that built reference.json.
import * as T from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
import { loadModels, isolateGarment, embed, score, palette, verdict } from './core.js';

const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };

// Download progress across all model files.
const files = {};
function onProgress(p) {
  if (p.status === 'progress' && p.total) files[p.file] = [p.loaded, p.total];
  const [loaded, total] = Object.values(files).reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
  if (total) {
    $('bar').style.width = `${(100 * loaded) / total}%`;
    status(`Loading the models… ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB (once, then cached)`);
  }
}

const ready = Promise.all([
  loadModels(T, onProgress),
  fetch('reference.json').then((r) => r.json()),
]).then(([models, ref]) => {
  $('bar').style.width = '100%';
  status('Ready. Choose a photo.');
  return { models, ref };
}).catch((err) => {
  status(`Couldn't load the models: ${err.message}. Try reloading the page.`);
  throw err;
});

let busy = false;
async function analyse(blob) {
  if (busy) return;
  busy = true;
  const url = URL.createObjectURL(blob);
  $('preview').src = url;
  $('preview').hidden = false;
  $('dropText').hidden = true;
  try {
    const { models, ref } = await ready;
    status('Cutting out the garment…');
    const image = await T.RawImage.fromBlob(blob);
    const g = await isolateGarment(T, models, image);
    status('Comparing with the Missoni archive…');
    const r = score(ref, await embed(T, models, g.image));
    show(r, g);
    status('Done. Try another photo.');
  } catch (err) {
    console.error(err);
    status(`Something went wrong: ${err.message}`);
  } finally {
    busy = false;
  }
}

function show(r, g) {
  $('empty').hidden = true;
  $('result').hidden = false;
  const s = g.found ? Math.round(r.score) : 0;
  $('score').textContent = g.found ? s : '–';
  $('verdict').textContent = verdict(r.score, g.found);
  $('explain').textContent = g.found
    ? `As Missoni as ${s}% of Missoni's own runway looks. Closest collection: ${r.closest}.`
    : 'Try a photo where one outfit is clearly visible.';
  $('meter').style.width = `${g.found ? s : 0}%`;
  $('meterLabel').setAttribute('aria-label', `Score ${s} out of 100; 50 is a typical Missoni look`);

  $('palette').replaceChildren(...palette(g.pixels).map(({ hex, share }) => {
    const d = document.createElement('div');
    d.style.flex = String(share);
    d.style.background = hex;
    d.title = `${hex} · ${Math.round(share * 100)}%`;
    return d;
  }));

  const canvas = $('garment');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(g.image.toCanvas(), 0, 0, canvas.width, canvas.height);

  $('matches').replaceChildren(...r.neighbours.map((n) => {
    const tr = document.createElement('tr');
    const page = (n.source_url.match(/#page=(\d+)/) || [])[1];
    tr.innerHTML = `<td>${n.collection}</td><td><a href="${n.source_url}" target="_blank" rel="noopener">lookbook${page ? `, p. ${page}` : ''}</a></td><td class="num">${n.similarity.toFixed(3)}</td>`;
    return tr;
  }));
}

// Inputs: file picker, drag and drop, example buttons.
$('file').addEventListener('change', (e) => e.target.files[0] && analyse(e.target.files[0]));
const drop = $('drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => e.dataTransfer.files[0] && analyse(e.dataTransfer.files[0]));
document.querySelectorAll('.examples button').forEach((b) =>
  b.addEventListener('click', async () => analyse(await (await fetch(b.dataset.src)).blob())));

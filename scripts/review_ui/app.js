"use strict";
// Review tool front end. Talks to scripts/review.py on localhost; nothing leaves this machine.

const $ = (s) => document.querySelector(s);
const S = {
  items: [], i: 0, item: null, slot: null, idx: 0, original: false, imgToken: 0, goToken: 0,
  heat: true, counter: false, alpha: 0.65, regions: {}, fine: {},
  bright: 1, contrast: 1, zoom: 1, pan: { x: 0, y: 0 },
};
const SHORT = { sag_fs: "Sagittal", cor_fs: "Coronal", ax_fs: "Axial", t1: "T1" };
const SUB = { sag_fs: "fluid-sensitive", cor_fs: "fluid-sensitive", ax_fs: "fluid-sensitive", t1: "not fluid-sensitive" };
const RED = [230, 103, 103], BLUE = [57, 135, 229];
const pct = (p) => `${Math.round(p * 100)}%`;
const stateOf = (p) => (p >= 0.5 ? "present" : "absent");
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await r.json().catch(() => ({}));
  if (!r.ok && r.status !== 202) throw new Error(data.error || `The server answered ${r.status}.`);
  return { status: r.status, data };
}

// ---------------------------------------------------------------- session and navigation
function applySession(s) {
  S.items = s.items;
  const sel = $("#q-label");
  if (sel.options.length === 1) s.labels.forEach((l) => sel.add(new Option(l, l)));
  $("#q-mode").value = s.mode;
  sel.value = s.label || "";
  renderTally(s.tally);
}

function renderTally(t) {
  $("#tally").innerHTML = t.graded
    ? `<strong>${t.graded}</strong> graded, <strong>${t.label_wrong}</strong> training labels wrong, <strong>${t.model_wrong}</strong> model mistakes`
    : "Nothing graded in this queue yet";
}

async function go(i) {
  if (i >= S.items.length) return showDone();
  if (i < 0) return;
  const token = ++S.goToken;
  $("#done").hidden = true;
  $("#main").hidden = false;
  const { data } = await api(`/api/item/${i}`);
  if (token !== S.goToken) return; // a newer navigation won
  S.i = i;
  S.item = data;
  S.regions = {};
  S.fine = {};
  S.slot = data.start.slot;
  S.idx = data.start.idx;
  resetView(false);
  renderItem();
}

function nextOpen(after) {
  const k = S.items.findIndex((it, j) => j > after && !it.decision);
  return k === -1 ? S.items.length : k;
}

// ---------------------------------------------------------------- item panel
function card(name, p) {
  const st = stateOf(p);
  return `<div class="card ${st}"><div class="k">${name}</div><div class="v">${pct(p)}</div><div class="s">${st}</div></div>`;
}

function renderItem() {
  const it = S.item;
  $("#position").textContent = `Item ${it.i + 1} of ${it.n}`;
  $("#track-fill").style.width = `${(100 * S.items.filter((x) => x.decision).length) / Math.max(1, it.n)}%`;
  $("#finding").textContent = it.label;
  $("#question").textContent = it.question;
  const gold = it.gold === null
    ? `<div class="card none"><div class="k">Gold label</div><div class="v">Not in gold set</div></div>`
    : `<div class="card ${it.gold ? "present" : "absent"}"><div class="k">Gold label</div><div class="v">${it.gold ? "Yes" : "No"}</div><div class="s">${it.gold ? "present" : "absent"}</div></div>`;
  $("#cards").innerHTML = card("Model", it.model) + card("Training label", it.report_label) + gold;
  const agree = stateOf(it.model) === stateOf(it.report_label);
  $("#agree").innerHTML = `The model and the training label <strong>${agree ? "agree" : "disagree"}</strong>.`;
  $("#summary").textContent = it.summary;

  renderReport();
  for (const kind of ["accept", "deny"]) {
    const c = it.choices[kind], b = $(`#${kind}`);
    b.querySelector(".t").textContent = c.title;
    b.querySelector(".d").textContent = c.detail;
    b.classList.toggle("says-present", c.truth === 1);
    b.classList.toggle("says-absent", c.truth === 0);
    b.classList.toggle("picked", it.decision === kind);
  }
  updateView();
}

function renderReport() {
  const r = S.item.report;
  const n = r.hits;
  const hits = $("#hits");
  hits.classList.toggle("none", n === 0);
  hits.textContent = n
    ? `${n} sentence${n > 1 ? "s are" : " is"} about this finding (highlighted).`
    : "No sentence mentions this finding. Read the full report before deciding.";
  if (!r.translated) hits.textContent += " The English translation isn't downloaded yet, so this is the original.";
  const toggle = $("#lang-toggle");
  toggle.hidden = !r.translated || r.english_original;
  toggle.querySelector("span").textContent = S.original ? "Show English" : "Show original";
  $("#sentences").innerHTML = r.sentences
    .map((s) => {
      const main = S.original || !s.en ? s.src : s.en;
      const sub = S.original && s.en && s.en !== s.src ? `<span class="orig">${esc(s.en)}</span>` : "";
      return `<p class="${s.hit ? "hit" : ""}">${esc(main)}${sub}</p>`;
    })
    .join("");
}

// ---------------------------------------------------------------- viewer
const series = () => S.item.series.find((s) => s.slot === S.slot);
const sliceWeight = () => Math.min(1, Math.abs(series().contrib[S.idx]) / S.item.scale);

function strength(c) {
  const m = Math.abs(c) / S.item.scale;
  return m > 0.66 ? "strong" : m > 0.33 ? "moderate" : m > 0.1 ? "weak" : "almost no";
}

function updateView() {
  const s = series();
  $("#series").innerHTML = S.item.series
    .map((x, k) => `<button type="button" role="tab" aria-selected="${x.slot === S.slot}" data-slot="${x.slot}" title="Key ${k + 1}">` +
      `${SHORT[x.slot]}<small>${SUB[x.slot]}</small><span class="n">${x.n}</span></button>`)
    .join("");
  const range = $("#slice-range");
  range.max = s.n;
  range.value = S.idx + 1;
  $("#slice-out").textContent = `${S.idx + 1} / ${s.n}`;
  const c = s.contrib[S.idx];
  const best = S.slot === S.item.start.slot && S.idx === S.item.start.idx;
  const push = strength(c) === "almost no" ? "almost no evidence" : `${strength(c)} push toward ${c > 0 ? "present" : "absent"}`;
  $("#hud-slice").textContent = `${SHORT[S.slot]} ${S.idx + 1}/${s.n} · ${push}${best ? " · strongest slice" : ""}`;
  $("#heat-toggle").setAttribute("aria-pressed", String(S.heat));
  $("#counter-toggle").setAttribute("aria-pressed", String(S.counter));
  drawStrip();
  loadImage();
}

async function loadImage() {
  const token = ++S.imgToken;
  const r = await fetch(`/api/image/${S.i}/${S.slot}/${S.idx}.png`);
  if (token !== S.imgToken) return;
  const pending = $("#pending");
  if (r.status === 202) {
    const d = await r.json();
    pending.hidden = false;
    pending.textContent = `Downloading this queue's images from Kaggle. ${d.ready} of ${d.studies} studies are ready; the first batch takes about a minute.`;
    $("#slice").removeAttribute("src");
    clearHeat();
    setTimeout(() => token === S.imgToken && loadImage(), 2500);
    return;
  }
  if (!r.ok) {
    pending.hidden = false;
    pending.textContent = "This slice couldn't be loaded. Check the terminal running review.py.";
    return;
  }
  const blob = await r.blob();
  if (token !== S.imgToken) return;
  const img = $("#slice");
  const old = img.src;
  img.src = URL.createObjectURL(blob);
  if (old.startsWith("blob:")) URL.revokeObjectURL(old);
  pending.hidden = true;
  drawHeat();
}

// ---------------------------------------------------------------- highlight
function sizeCanvas() {
  const c = $("#heat"), box = $("#lightbox");
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(box.clientWidth * dpr);
  c.height = Math.round(box.clientHeight * dpr);
  return c;
}

function clearHeat() {
  const c = sizeCanvas();
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function setLegend(kind) {
  const noun = esc(S.item.noun);
  const parts = [];
  if (kind) {
    parts.push(`<span class="sw present"></span>evidence for ${noun}`);
    if (kind === "outline") parts.push(`<span class="sw ring"></span>strongest area`);
    if (S.counter) parts.push(`<span class="sw absent"></span>evidence against`);
  }
  $("#hud-legend").innerHTML = parts.join(" &nbsp; ");
}

async function drawHeat() {
  clearHeat();
  setLegend(null);
  $("#hud-status").textContent = "";
  const locate = $("#locate");
  if (!S.item || !$("#slice").src) return;
  if (!S.heat) { locate.textContent = "Highlight is off (press H)."; return; }
  if (S.item.negligible) {
    locate.textContent = "The model's call barely depends on any slice, so there is no area to highlight.";
    return;
  }
  const weight = sliceWeight();
  if (weight < 0.1) {
    locate.innerHTML = "This slice barely affects the model's call. Press <kbd>G</kbd> for the strongest slice.";
    return;
  }
  const key = `${S.i}/${S.slot}/${S.idx}`;
  if (S.fine[key]) return renderGrid(S.fine[key].grid, S.fine[key].n, weight, true);

  // quick coarse map (4x4) while the fine map (8x8, re-encoded) is computed
  if (!S.regions[key]) S.regions[key] = (await api(`/api/region/${key}`)).data.grid;
  if (key !== `${S.i}/${S.slot}/${S.idx}`) return; // moved on while loading
  renderGrid(S.regions[key], 4, weight, false);
  $("#hud-status").textContent = "Refining highlight…";
  fetchFine(key, weight);
}

async function fetchFine(key, weight) {
  for (let tries = 0; tries < 60; tries++) {
    const { status, data } = await api(`/api/fine/${key}`);
    if (status === 200) {
      S.fine[key] = data;
      if (key === `${S.i}/${S.slot}/${S.idx}` && S.heat) {
        $("#hud-status").textContent = "";
        renderGrid(data.grid, data.n, weight, true);
      }
      return;
    }
    if (key !== `${S.i}/${S.slot}/${S.idx}`) return;
    await new Promise((r) => setTimeout(r, 2000)); // images still downloading
  }
}

// bilinear upsampling of an n*n cell-centred grid to m*m, with a border of zeros so contours close
function upsample(cells, n, m) {
  const out = new Float32Array((m + 2) * (m + 2));
  for (let y = 0; y < m; y++) {
    for (let x = 0; x < m; x++) {
      const u = Math.min(n - 1, Math.max(0, ((x + 0.5) / m) * n - 0.5));
      const v = Math.min(n - 1, Math.max(0, ((y + 0.5) / m) * n - 0.5));
      const x0 = Math.floor(u), y0 = Math.floor(v), x1 = Math.min(n - 1, x0 + 1), y1 = Math.min(n - 1, y0 + 1);
      const fx = u - x0, fy = v - y0;
      const val = cells[y0 * n + x0] * (1 - fx) * (1 - fy) + cells[y0 * n + x1] * fx * (1 - fy)
        + cells[y1 * n + x0] * (1 - fx) * fy + cells[y1 * n + x1] * fx * fy;
      out[(y + 1) * (m + 2) + (x + 1)] = val;
    }
  }
  return out;
}

// marching squares: line segments where the field crosses thr, in padded-grid coordinates
function contour(field, w, thr) {
  const segs = [];
  const at = (x, y) => field[y * w + x];
  for (let y = 0; y < w - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const a = at(x, y), b = at(x + 1, y), c = at(x + 1, y + 1), d = at(x, y + 1);
      const k = (a > thr) * 8 + (b > thr) * 4 + (c > thr) * 2 + (d > thr);
      if (k === 0 || k === 15) continue;
      const t = (p, q) => (thr - p) / (q - p);
      const T = [x + t(a, b), y], R = [x + 1, y + t(b, c)], B = [x + t(d, c), y + 1], L = [x, y + t(a, d)];
      const table = { 1: [[L, B]], 2: [[B, R]], 3: [[L, R]], 4: [[T, R]], 5: [[L, T], [B, R]], 6: [[T, B]], 7: [[L, T]],
        8: [[L, T]], 9: [[T, B]], 10: [[T, R], [L, B]], 11: [[T, R]], 12: [[L, R]], 13: [[B, R]], 14: [[L, B]] };
      for (const [p, q] of table[k]) segs.push([p[0], p[1], q[0], q[1]]);
    }
  }
  return segs;
}

function place(cx, cy) {
  const h = cx < 0.36 ? "left" : cx > 0.64 ? "right" : "centre";
  const v = cy < 0.36 ? "upper" : cy > 0.64 ? "lower" : "middle";
  return v === "middle" && h === "centre" ? "centre" : v === "middle" ? `middle ${h}` : h === "centre" ? `${v} centre` : `${v} ${h}`;
}

function renderGrid(cells, n, weight, fine) {
  const c = sizeCanvas(), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const peak = Math.max(...cells.map(Math.abs)) || 1;
  const posPeak = Math.max(0, ...cells);
  const negPeak = Math.max(0, ...cells.map((v) => -v));

  // 1. soft heat layer: red toward present, blue (optional) toward absent
  const off = document.createElement("canvas");
  off.width = off.height = n;
  const px = off.getContext("2d").createImageData(n, n);
  cells.forEach((v, k) => {
    const a = Math.abs(v) / peak;
    if (v < 0 && !S.counter) return;
    const [r, gg, b] = v > 0 ? RED : BLUE;
    px.data.set([r, gg, b, Math.round(255 * S.alpha * Math.max(0.35, weight) * (a < 0.2 ? 0 : Math.pow(a, 1.2)))], k * 4);
  });
  off.getContext("2d").putImageData(px, 0, 0);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(off, 0, 0, n, n, 0, 0, W, H);

  // 2. outline around the strongest area (fine map only: the coarse 4x4 cells are too blunt to outline)
  const M = 64, w = M + 2, dpr = window.devicePixelRatio || 1;
  const toPx = (x, y) => [((x - 0.5) / M) * W, ((y - 0.5) / M) * H];
  const drawOutline = (field, thr, rgb, dashed) => {
    const segs = contour(field, w, thr);
    if (!segs.length) return null;
    for (const [lw, col] of [[5 * dpr, "rgba(0,0,0,0.75)"], [2.4 * dpr, `rgb(${rgb.join(",")})`]]) {
      g.beginPath();
      g.setLineDash(dashed ? [6 * dpr, 5 * dpr] : []);
      for (const [x1, y1, x2, y2] of segs) {
        const [ax, ay] = toPx(x1, y1), [bx, by] = toPx(x2, y2);
        g.moveTo(ax, ay);
        g.lineTo(bx, by);
      }
      g.lineWidth = lw;
      g.lineCap = "round";
      g.strokeStyle = col;
      g.stroke();
    }
    g.setLineDash([]);
    return segs;
  };

  const locate = $("#locate");
  const noun = esc(S.item.noun);
  if (!fine || posPeak < 0.2 * peak || posPeak < 0.01) {
    setLegend("heat");
    if (!fine) locate.textContent = "Showing the quick map; the sharper highlight follows in a moment.";
    else locate.innerHTML = `On this slice the image mostly argues <strong>against</strong> ${noun}${S.counter ? " (blue)" : " (press C to see where)"}.`;
    return;
  }
  const pos = upsample(cells.map((v) => Math.max(0, v)), n, M);
  const thr = 0.5 * posPeak;
  const segs = drawOutline(pos, thr, RED, false);
  if (S.counter && negPeak >= 0.2 * peak) drawOutline(upsample(cells.map((v) => Math.max(0, -v)), n, M), 0.5 * negPeak, BLUE, true);

  // where the outlined area sits, and a tag above it
  let sx = 0, sy = 0, sw = 0, minx = Infinity, miny = Infinity;
  for (let y = 1; y <= M; y++) {
    for (let x = 1; x <= M; x++) {
      const v = pos[y * w + x];
      if (v > thr) { sx += x * v; sy += y * v; sw += v; minx = Math.min(minx, x); miny = Math.min(miny, y); }
    }
  }
  if (segs && sw > 0) {
    const where = place((sx / sw - 0.5) / M, (sy / sw - 0.5) / M);
    locate.innerHTML = `Red outline: where the model's evidence for <strong>${noun}</strong> sits on this slice (${where} of the image). It explains the model, it is not a diagnosis.`;
    const [tx, ty] = toPx(minx, miny);
    const label = `evidence for ${S.item.noun}`;
    g.font = `700 ${12 * dpr}px "Atkinson Hyperlegible", system-ui, sans-serif`;
    const tw = g.measureText(label).width + 10 * dpr, th = 18 * dpr;
    const bx = Math.min(W - tw - 4, Math.max(4, tx)), by = Math.max(4, ty - th - 6 * dpr);
    g.fillStyle = "rgba(0,0,0,0.72)";
    g.fillRect(bx, by, tw, th);
    g.fillStyle = `rgb(${RED.join(",")})`;
    g.fillText(label, bx + 5 * dpr, by + 13 * dpr);
  }
  setLegend("outline");
}

// ---------------------------------------------------------------- strip
function drawStrip() {
  const svg = $("#strip");
  const W = svg.clientWidth, H = 60, top = 14, mid = top + (H - top) / 2, half = (H - top) / 2 - 3;
  const all = S.item.series, gap = 14;
  const total = all.reduce((a, s) => a + s.n, 0);
  const bw = (W - gap * (all.length - 1)) / total;
  const w = Math.max(1, bw - (bw > 4 ? 1.5 : 0.5));
  const out = [`<line class="zero" x1="0" x2="${W}" y1="${mid}" y2="${mid}"/>`];
  let x = 0;
  all.forEach((s, si) => {
    if (si) out.push(`<line class="sep" x1="${x - gap / 2}" x2="${x - gap / 2}" y1="${top}" y2="${H}"/>`);
    out.push(`<text class="series-label" x="${x}" y="11">${SHORT[s.slot]}</text>`);
    s.contrib.forEach((c, k) => {
      const h = Math.max(1, (Math.abs(c) / S.item.scale) * half);
      const bx = x + k * bw;
      out.push(`<rect x="${bx.toFixed(2)}" y="${(c > 0 ? mid - h : mid).toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" rx="${Math.min(2, w / 2).toFixed(2)}" fill="var(--${c > 0 ? "present" : "absent"})"/>`);
      out.push(`<rect class="bar-hit" data-slot="${s.slot}" data-k="${k}" x="${bx.toFixed(2)}" y="${top}" width="${bw.toFixed(2)}" height="${H - top}"/>`);
    });
    if (s.slot === S.slot) out.push(`<rect class="cursor" x="${(x + S.idx * bw - 1.5).toFixed(2)}" y="${top}" width="${(bw + 1.5).toFixed(2)}" height="${H - top}" rx="2"/>`);
    x += s.n * bw + gap;
  });
  svg.innerHTML = out.join("");
}

function setSlice(slot, idx) {
  const s = S.item.series.find((x) => x.slot === slot);
  if (idx === undefined) { // entering a series: open its strongest-evidence slice
    idx = s.contrib.reduce((best, c, k) => (Math.abs(c) > Math.abs(s.contrib[best]) ? k : best), 0);
  }
  S.slot = slot;
  S.idx = Math.max(0, Math.min(s.n - 1, idx));
  updateView();
}

function stepSeries(d) {
  const slots = S.item.series.map((s) => s.slot);
  setSlice(slots[(slots.indexOf(S.slot) + d + slots.length) % slots.length]);
}

const strongest = () => setSlice(S.item.start.slot, S.item.start.idx);

// ---------------------------------------------------------------- zoom, pan, brightness
function applyView() {
  const lb = $("#lightbox");
  const maxPan = ((S.zoom - 1) * lb.clientWidth) / 2;
  S.pan.x = Math.max(-maxPan, Math.min(maxPan, S.pan.x));
  S.pan.y = Math.max(-maxPan, Math.min(maxPan, S.pan.y));
  $("#zoomer").style.transform = `translate(${S.pan.x}px, ${S.pan.y}px) scale(${S.zoom})`;
  $("#slice").style.filter = `brightness(${S.bright}) contrast(${S.contrast})`;
  lb.classList.toggle("zoomed", S.zoom > 1.001);
  $("#hud-zoom").textContent = S.zoom > 1.001 ? `${Math.round(S.zoom * 100)}%` : "";
}

function zoomBy(f, cx = 0, cy = 0) { // cx, cy: pointer offset from the image centre, in screen pixels
  const z = Math.max(1, Math.min(6, S.zoom * f));
  S.pan.x = cx - ((cx - S.pan.x) * z) / S.zoom;
  S.pan.y = cy - ((cy - S.pan.y) * z) / S.zoom;
  S.zoom = z;
  if (z === 1) S.pan = { x: 0, y: 0 };
  applyView();
}

function resetView(redraw = true) {
  S.zoom = 1;
  S.pan = { x: 0, y: 0 };
  S.bright = 1;
  S.contrast = 1;
  $("#bright").value = 1;
  $("#contrast").value = 1;
  applyView();
  if (redraw && S.item) drawHeat();
}

// ---------------------------------------------------------------- decisions
async function decide(kind) {
  if (!S.item) return;
  const i = S.i;
  const { data } = await api("/api/decision", { i, decision: kind });
  S.items[i].decision = kind;
  renderTally(data.tally);
  toast(kind === "unsure" ? "Marked unsure" : `Saved. ${data.saved}`);
  go(nextOpen(i));
}

function showDone() {
  S.item = null;
  $("#main").hidden = true;
  $("#done").hidden = false;
  const t = $("#tally").textContent;
  $("#done-text").textContent = `${t}. Every decision is saved in meta/reviews.csv.`;
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
}

function toggleHelp(show) {
  const h = $("#help");
  h.hidden = show === undefined ? !h.hidden : !show;
  $("#help-toggle").setAttribute("aria-expanded", String(!h.hidden));
}

// ---------------------------------------------------------------- input
function bind() {
  $("#queue-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const { data } = await api("/api/queue", { mode: $("#q-mode").value, label: $("#q-label").value });
    applySession(data);
    if (!data.items.length) {
      showDone();
      $("#done-text").textContent = "Nothing left to review for this selection.";
      return;
    }
    go(data.first_open);
  });
  for (const k of ["accept", "deny", "unsure"]) $(`#${k}`).addEventListener("click", () => decide(k));
  $("#skip").addEventListener("click", () => go(S.i + 1));
  $("#back").addEventListener("click", () => go(S.i - 1));
  $("#strongest").addEventListener("click", strongest);
  $("#heat-toggle").addEventListener("click", () => { S.heat = !S.heat; updateView(); });
  $("#counter-toggle").addEventListener("click", () => { S.counter = !S.counter; updateView(); });
  $("#alpha").addEventListener("input", (e) => { S.alpha = Number(e.target.value); drawHeat(); });
  $("#bright").addEventListener("input", (e) => { S.bright = Number(e.target.value); applyView(); });
  $("#contrast").addEventListener("input", (e) => { S.contrast = Number(e.target.value); applyView(); });
  $("#zoom-in").addEventListener("click", () => zoomBy(1.25));
  $("#zoom-out").addEventListener("click", () => zoomBy(0.8));
  $("#view-reset").addEventListener("click", () => resetView());
  $("#slice-range").addEventListener("input", (e) => setSlice(S.slot, Number(e.target.value) - 1));
  $("#lang-toggle").addEventListener("click", () => { S.original = !S.original; renderReport(); });
  $("#help-toggle").addEventListener("click", () => toggleHelp());
  $("#help-close").addEventListener("click", () => toggleHelp(false));
  $("#help").addEventListener("click", (e) => { if (e.target.id === "help") toggleHelp(false); });
  $("#series").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-slot]");
    if (b) setSlice(b.dataset.slot);
  });

  const svg = $("#strip"), tip = $("#tip");
  svg.addEventListener("click", (e) => {
    const r = e.target.closest(".bar-hit");
    if (r) setSlice(r.dataset.slot, Number(r.dataset.k));
  });
  svg.addEventListener("mousemove", (e) => {
    const r = e.target.closest(".bar-hit");
    if (!r) { tip.hidden = true; return; }
    const s = S.item.series.find((x) => x.slot === r.dataset.slot), k = Number(r.dataset.k), c = s.contrib[k];
    const what = strength(c) === "almost no" ? "almost no evidence" : `${strength(c)} push toward ${c > 0 ? "present" : "absent"}`;
    tip.textContent = `${s.title}, slice ${k + 1}: ${what}`;
    const box = svg.getBoundingClientRect();
    tip.style.left = `${Math.min(box.width - 140, Math.max(140, e.clientX - box.left))}px`;
    tip.hidden = false;
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });

  // scroll moves through slices like a PACS viewer; pinch or ctrl+scroll zooms around the pointer
  const lb = $("#lightbox");
  let wheel = 0;
  $("#stage").addEventListener("wheel", (e) => {
    if (!S.item) return;
    e.preventDefault();
    if (e.ctrlKey) {
      const b = lb.getBoundingClientRect();
      zoomBy(Math.exp(-e.deltaY * 0.01), e.clientX - b.left - b.width / 2, e.clientY - b.top - b.height / 2);
      return;
    }
    wheel += e.deltaY;
    if (Math.abs(wheel) >= 40) {
      setSlice(S.slot, S.idx + Math.sign(wheel));
      wheel = 0;
    }
  }, { passive: false });
  let drag = null;
  lb.addEventListener("pointerdown", (e) => {
    if (S.zoom <= 1.001) return;
    drag = { x: e.clientX, y: e.clientY, px: S.pan.x, py: S.pan.y };
    lb.setPointerCapture(e.pointerId);
    lb.classList.add("panning");
  });
  lb.addEventListener("pointermove", (e) => {
    if (!drag) return;
    S.pan.x = drag.px + e.clientX - drag.x;
    S.pan.y = drag.py + e.clientY - drag.y;
    applyView();
  });
  const endDrag = () => { drag = null; lb.classList.remove("panning"); };
  lb.addEventListener("pointerup", endDrag);
  lb.addEventListener("pointercancel", endDrag);
  lb.addEventListener("dblclick", () => resetView());

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") return toggleHelp(false);
    if (e.key === "?") { e.preventDefault(); return toggleHelp(); }
    if (!S.item) return;
    const k = e.key.toLowerCase();
    const actions = {
      a: () => decide("accept"), d: () => decide("deny"), u: () => decide("unsure"),
      s: () => go(S.i + 1), b: () => go(S.i - 1), g: strongest,
      h: () => { S.heat = !S.heat; updateView(); },
      c: () => { S.counter = !S.counter; updateView(); },
      o: () => { if (!$("#lang-toggle").hidden) { S.original = !S.original; renderReport(); } },
      arrowleft: () => setSlice(S.slot, S.idx - 1), arrowright: () => setSlice(S.slot, S.idx + 1),
      arrowup: () => stepSeries(-1), arrowdown: () => stepSeries(1),
      "+": () => zoomBy(1.25), "=": () => zoomBy(1.25), "-": () => zoomBy(0.8), "0": () => resetView(),
    };
    if (/^[1-4]$/.test(k) && S.item.series[Number(k) - 1]) actions[k] = () => setSlice(S.item.series[Number(k) - 1].slot);
    if (actions[k]) { e.preventDefault(); actions[k](); }
  });
  window.addEventListener("resize", () => { if (S.item) { drawStrip(); applyView(); drawHeat(); } });
}

(async function init() {
  try {
    const { data } = await api("/api/session");
    applySession(data);
    bind();
    if (!data.items.length) { showDone(); $("#done-text").textContent = "Nothing left to review for this selection."; return; }
    await go(data.first_open);
  } catch (err) {
    document.body.innerHTML = `<p style="padding:24px">The review server isn't answering: ${esc(err.message)}. Restart it with <code>python scripts/review.py</code>.</p>`;
  }
})();

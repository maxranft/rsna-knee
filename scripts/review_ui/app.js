"use strict";
// Review tool front end. Talks to scripts/review.py on localhost; nothing leaves this machine.

const $ = (s) => document.querySelector(s);
const S = { items: [], i: 0, item: null, slot: null, idx: 0, heat: true, original: false, imgToken: 0, goToken: 0, regions: {} };
const SHORT = { sag_fs: "Sagittal", cor_fs: "Coronal", ax_fs: "Axial", t1: "T1" };
const SUB = { sag_fs: "fluid-sensitive", cor_fs: "fluid-sensitive", ax_fs: "fluid-sensitive", t1: "not fluid-sensitive" };
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
  S.slot = data.start.slot;
  S.idx = data.start.idx;
  renderItem();
}

function nextOpen(after) {
  const k = S.items.findIndex((it, j) => j > after && !it.decision);
  return k === -1 ? S.items.length : k;
}

// ---------------------------------------------------------------- item
function renderItem() {
  const it = S.item;
  $("#position").textContent = `Item ${it.i + 1} of ${it.n}`;
  $("#track-fill").style.width = `${(100 * S.items.filter((x) => x.decision).length) / Math.max(1, it.n)}%`;
  $("#question").textContent = it.question;

  const opinion = (name, p) =>
    `<dt>${name}</dt><dd><div class="opinion"><div class="meter"><i style="width:${(p * 100).toFixed(1)}%"></i></div>` +
    `<span class="value">${pct(p)}<span class="state">${stateOf(p)}</span></span></div></dd>`;
  const gold = it.gold === null ? "Not in the gold set" : `Radiologist-checked: ${it.gold ? "present" : "absent"}`;
  const agree = stateOf(it.model) === stateOf(it.report_label);
  $("#opinions").innerHTML =
    opinion("Model", it.model) + opinion("Training label", it.report_label) +
    `<dt>Gold label</dt><dd>${gold}</dd>` +
    `<p class="verdict-line">The model and the training label <strong>${agree ? "agree" : "disagree"}</strong>.</p>`;
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

function strength(c) {
  const m = Math.abs(c) / S.item.scale;
  return m > 0.66 ? "strong" : m > 0.33 ? "moderate" : m > 0.1 ? "weak" : "almost no";
}

function updateView() {
  const s = series();
  $("#series").innerHTML = S.item.series
    .map((x) => `<button type="button" role="tab" aria-selected="${x.slot === S.slot}" data-slot="${x.slot}">` +
      `<span>${SHORT[x.slot]}<small>${SUB[x.slot]}</small></span><span class="n">${x.n}</span></button>`)
    .join("");
  const c = s.contrib[S.idx];
  const where = strength(c) === "almost no" ? "Almost no evidence on this slice."
    : `${strength(c)[0].toUpperCase() + strength(c).slice(1)} push toward ${c > 0 ? "present" : "absent"}.`;
  const best = S.slot === S.item.start.slot && S.idx === S.item.start.idx ? " This is the model's strongest-evidence slice." : "";
  $("#slice-caption").innerHTML = `<strong>Slice ${S.idx + 1} of ${s.n}</strong><br>${where}${best}`;
  $("#heat-toggle span").textContent = S.heat ? "Hide heat map" : "Show heat map";
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

function clearHeat() {
  const c = $("#heat");
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

async function drawHeat() {
  const c = $("#heat");
  const box = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(box.width * dpr);
  c.height = Math.round(box.height * dpr);
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (!S.heat || !S.item || !$("#slice").src || S.item.negligible) return;
  const key = `${S.i}/${S.slot}/${S.idx}`;
  if (!S.regions[key]) S.regions[key] = (await api(`/api/region/${key}`)).data.grid;
  if (key !== `${S.i}/${S.slot}/${S.idx}` || !S.heat) return; // moved on while loading
  // 4x4 region evidence drawn at 4x4 and scaled up smoothly: red = toward present, blue = toward absent.
  // Shape comes from this slice; overall strength follows how much the slice matters to the study.
  const cells = S.regions[key];
  const peak = Math.max(...cells.map(Math.abs)) || 1;
  const weight = Math.min(1, Math.abs(series().contrib[S.idx]) / S.item.scale);
  if (weight < 0.1) return; // this slice doesn't matter to the prediction: nothing honest to show
  const off = document.createElement("canvas");
  off.width = off.height = 4;
  const px = off.getContext("2d").createImageData(4, 4);
  cells.forEach((v, k) => {
    const a = Math.abs(v) / peak;
    const [r, gg, b] = v > 0 ? [230, 103, 103] : [57, 135, 229];
    px.data.set([r, gg, b, Math.round(255 * 0.55 * weight * (a < 0.15 ? 0 : a))], k * 4);
  });
  off.getContext("2d").putImageData(px, 0, 0);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(off, 0, 0, 4, 4, 0, 0, c.width, c.height);
}

function drawStrip() {
  const svg = $("#strip");
  const W = svg.clientWidth, H = 80, top = 16, mid = top + (H - top) / 2, half = (H - top) / 2 - 3;
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
  $("#heat-toggle").addEventListener("click", () => { S.heat = !S.heat; updateView(); });
  $("#lang-toggle").addEventListener("click", () => { S.original = !S.original; renderReport(); });
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

  let wheel = 0; // scroll on the image moves through slices, like a PACS viewer
  $("#stage").addEventListener("wheel", (e) => {
    if (!S.item) return;
    e.preventDefault();
    wheel += e.deltaY;
    if (Math.abs(wheel) >= 40) {
      setSlice(S.slot, S.idx + Math.sign(wheel));
      wheel = 0;
    }
  }, { passive: false });

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey || !S.item) return;
    const k = e.key.toLowerCase();
    const actions = {
      a: () => decide("accept"), d: () => decide("deny"), u: () => decide("unsure"),
      s: () => go(S.i + 1), b: () => go(S.i - 1),
      h: () => { S.heat = !S.heat; updateView(); },
      o: () => { if (!$("#lang-toggle").hidden) { S.original = !S.original; renderReport(); } },
      arrowleft: () => setSlice(S.slot, S.idx - 1), arrowright: () => setSlice(S.slot, S.idx + 1),
      arrowup: () => stepSeries(-1), arrowdown: () => stepSeries(1),
    };
    if (/^[1-4]$/.test(k) && S.item.series[Number(k) - 1]) actions[k] = () => setSlice(S.item.series[Number(k) - 1].slot);
    if (actions[k]) { e.preventDefault(); actions[k](); }
  });
  window.addEventListener("resize", () => { if (S.item) { drawStrip(); drawHeat(); } });
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

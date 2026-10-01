"use strict";
// Knee MRI Review front end (PACS-style). Talks to scripts/review.py on localhost; nothing leaves this machine.

const $ = (s, r = document) => r.querySelector(s);
const S = {
  items: [], i: 0, item: null, original: false, goToken: 0,
  layout: 1, active: 0, vps: [], tool: "wl",
  heat: true, counter: false, alpha: 0.65,
  wl: { w: 255, l: 127.5, invert: false },
  regions: {}, fine: {},
};
const CODE = { sag_fs: "SAG FS", cor_fs: "COR FS", ax_fs: "AX FS", t1: "T1" };
const NAME = { sag_fs: "Sagittal fat-sat", cor_fs: "Coronal fat-sat", ax_fs: "Axial fat-sat", t1: "T1 (no fat-sat)" };
const RED = [230, 103, 103], BLUE = [57, 135, 229];
const pct = (p) => `${Math.round(p * 100)}%`;
const stateOf = (p) => (p >= 0.5 ? "present" : "absent");
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const dpr = () => window.devicePixelRatio || 1;

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
    ? `<strong>${t.graded}</strong> graded · <strong>${t.label_wrong}</strong> labels wrong · <strong>${t.model_wrong}</strong> model errors`
    : "Nothing graded in this worklist yet";
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
  renderItem();
}

function nextOpen(after) {
  const k = S.items.findIndex((it, j) => j > after && !it.decision);
  return k === -1 ? S.items.length : k;
}

// ---------------------------------------------------------------- truth and verdict
function truth(it) {
  if (it.gold !== null) return { state: it.gold ? "present" : "absent", p: null, src: "Radiologist gold label", verified: true };
  return { state: stateOf(it.report_label), p: it.report_label, src: "Report label, not radiologist-checked", verified: false };
}

function renderVerdict() {
  const it = S.item, t = truth(it), m = stateOf(it.model), agree = t.state === m;
  $("#verdict").innerHTML =
    `<span class="what">${esc(cap(it.noun))}</span>` +
    `<span class="slot"><span class="k">Actual</span><span class="chip ${t.state}">${t.state.toUpperCase()}</span><span class="src">${t.src}</span></span>` +
    `<span class="arrow" aria-hidden="true">→</span>` +
    `<span class="slot"><span class="k">Model prediction</span><span class="chip ${m}">${m.toUpperCase()} <span class="p">${pct(it.model)}</span></span></span>` +
    `<span class="match ${agree ? "" : "no"}">${agree ? "Model matches" : "Model disagrees"}</span>`;
}

// ---------------------------------------------------------------- item panel
function meterRow(name, p) {
  const st = stateOf(p);
  return `<tr><td class="n">${name}</td><td class="b"><div class="meter"><i style="width:${(p * 100).toFixed(1)}%;background:var(--${st})"></i></div></td>` +
    `<td class="v">${pct(p)}</td><td class="s"><span class="state ${st}">${cap(st)}</span></td></tr>`;
}

function renderItem() {
  const it = S.item;
  $("#position").textContent = `Item ${it.i + 1} of ${it.n}`;
  $("#track-fill").style.width = `${(100 * S.items.filter((x) => x.decision).length) / Math.max(1, it.n)}%`;
  $("#finding").textContent = it.label;
  $("#question").textContent = it.question;
  const gold = it.gold === null
    ? `<tr><td class="n">Gold label</td><td class="b"></td><td class="v">–</td><td class="s"><span class="state none">Not in set</span></td></tr>`
    : `<tr><td class="n">Gold label</td><td class="b"></td><td class="v">–</td><td class="s"><span class="state ${it.gold ? "present" : "absent"}">${it.gold ? "Present" : "Absent"}</span></td></tr>`;
  $("#assess").innerHTML = meterRow("Model", it.model) + meterRow("Training label", it.report_label) + gold;
  const agree = stateOf(it.model) === stateOf(it.report_label);
  $("#agree").innerHTML = `Model and training label <strong>${agree ? "agree" : "disagree"}</strong>.`;
  $("#summary").textContent = it.summary;
  $("#study-head").innerHTML = `<strong>Study</strong>…${esc(it.study_short)}<br>${it.series.length} series`;
  renderVerdict();
  renderReport();
  for (const kind of ["accept", "deny"]) {
    const c = it.choices[kind], b = $(`#${kind}`);
    b.classList.toggle("says-present", c.truth === 1);
    b.classList.toggle("says-absent", c.truth === 0);
    b.classList.toggle("picked", it.decision === kind);
    b.title = `${c.title}. ${c.detail}`;
  }
  showConsequence(null);
  buildViewports();
  renderThumbs();
}

function showConsequence(kind) {
  const ch = S.item.choices, el = $("#consequence");
  if (kind && ch[kind]) { el.innerHTML = `<b>${esc(ch[kind].title)}</b>. ${esc(ch[kind].detail)}`; return; }
  const a = ch.accept.title.replace(/^Accept:\s*/, ""), d = ch.deny.title.replace(/^Deny:\s*/, "");
  el.innerHTML = `<b>Accept</b> records ${esc(a)}. <b>Deny</b> records ${esc(d)}.`;
}

function renderReport() {
  const r = S.item.report;
  const n = r.hits;
  const hits = $("#hits");
  hits.classList.toggle("none", n === 0);
  hits.textContent = n
    ? `${n} sentence${n > 1 ? "s" : ""} about this finding (highlighted).`
    : "No sentence mentions this finding. Read the full report.";
  if (!r.translated) hits.textContent += " Translation not downloaded yet; showing the original.";
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

// ---------------------------------------------------------------- series browser
const seriesOf = (slot) => S.item.series.find((s) => s.slot === slot);
const strongestIdx = (slot) => {
  const s = seriesOf(slot);
  return s.contrib.reduce((best, c, k) => (Math.abs(c) > Math.abs(s.contrib[best]) ? k : best), 0);
};

function renderThumbs() {
  const it = S.item;
  $("#thumbs").innerHTML = it.series.map((s, k) => {
    const peak = s.contrib.reduce((a, c) => (Math.abs(c) > Math.abs(a) ? c : a), 0);
    const w = Math.min(100, (100 * Math.abs(peak)) / it.scale);
    return `<button type="button" class="thumb" data-slot="${s.slot}" title="${NAME[s.slot]}: ${s.n} images. Bar: strongest evidence in this series">` +
      `<div class="im"><img alt="" data-src="/api/image/${it.i}/${s.slot}/${strongestIdx(s.slot)}.png"><span class="se">Se ${k + 1}</span><span class="n">${s.n} im</span></div>` +
      `<span class="cap"><b>${CODE[s.slot]}</b>${NAME[s.slot]}</span>` +
      `<span class="ev"><i style="width:${w.toFixed(0)}%;background:var(--${peak >= 0 ? "present" : "absent"})"></i></span></button>`;
  }).join("");
  for (const img of document.querySelectorAll("#thumbs img")) loadThumb(img, 0);
  markActiveThumb();
}

function loadThumb(img, tries) {
  const item = S.i;
  img.onerror = () => { if (tries < 40 && item === S.i) setTimeout(() => loadThumb(img, tries + 1), 3000); };
  img.src = `${img.dataset.src}?t=${tries}`;
}

function markActiveThumb() {
  const vp = S.vps[S.active];
  for (const b of document.querySelectorAll("#thumbs .thumb")) b.classList.toggle("active", !!vp && b.dataset.slot === vp.slot);
}

// ---------------------------------------------------------------- viewports
function buildViewports() {
  const it = S.item, grid = $("#grid");
  const prevSlot = S.vps[S.active] ? S.vps[S.active].slot : it.start.slot;
  grid.className = `grid ${S.layout === 1 ? "g1" : "g4"}`;
  grid.innerHTML = "";
  let plan;
  if (S.layout === 1) {
    const slot = it.series.some((s) => s.slot === prevSlot) && S.vps.length ? prevSlot : it.start.slot;
    plan = [{ slot, idx: slot === it.start.slot ? it.start.idx : strongestIdx(slot) }];
  } else {
    plan = it.series.slice(0, 4).map((s) => ({ slot: s.slot, idx: s.slot === it.start.slot ? it.start.idx : strongestIdx(s.slot) }));
  }
  S.vps = plan.map((p, k) => makeViewport(k, p.slot, p.idx));
  S.active = Math.max(0, S.vps.findIndex((v) => v.slot === prevSlot));
  layoutSquares();
  S.vps.forEach(renderVp);
  setActive(S.active);
}

function makeViewport(k, slot, idx) {
  const el = document.createElement("div");
  el.className = `vp tool-${S.tool}`;
  el.innerHTML = `<div class="sq"><div class="zoomer"><img alt="" draggable="false"><canvas></canvas></div></div>` +
    `<div class="ov ov-tl"></div><div class="ov ov-tr"></div><div class="ov ov-bl"></div><div class="ov ov-br"></div>` +
    `<svg class="bar" aria-hidden="true"></svg><div class="pending" hidden></div>`;
  $("#grid").appendChild(el);
  const vp = {
    k, el, slot, idx, zoom: 1, pan: { x: 0, y: 0 }, imgToken: 0, outline: false,
    sq: $(".sq", el), zoomer: $(".zoomer", el), img: $("img", el), canvas: $("canvas", el), bar: $(".bar", el), pending: $(".pending", el),
    ov: { tl: $(".ov-tl", el), tr: $(".ov-tr", el), bl: $(".ov-bl", el), br: $(".ov-br", el) },
  };
  bindViewport(vp);
  return vp;
}

function layoutSquares() {
  for (const vp of S.vps) {
    const side = Math.max(50, Math.min(vp.el.clientWidth - 26, vp.el.clientHeight - 4));
    vp.sq.style.width = vp.sq.style.height = `${side}px`;
    applyTransform(vp);
  }
}

function renderVp(vp) {
  renderCorners(vp);
  drawBar(vp);
  loadImage(vp);
}

function strength(c) {
  const m = Math.abs(c) / S.item.scale;
  return m > 0.66 ? "strong" : m > 0.33 ? "moderate" : m > 0.1 ? "weak" : "no";
}

function renderCorners(vp) {
  const s = seriesOf(vp.slot), it = S.item;
  const se = it.series.findIndex((x) => x.slot === vp.slot) + 1;
  const c = s.contrib[vp.idx], st = strength(c);
  const best = vp.slot === it.start.slot && vp.idx === it.start.idx;
  vp.ov.tl.innerHTML = `${CODE[vp.slot]}<br>Se ${se}/${it.series.length}${S.layout === 1 ? `<br>${NAME[vp.slot]}` : ""}`;
  vp.ov.tr.innerHTML = `Im ${vp.idx + 1}/${s.n}` +
    (st === "no" ? "<br>no evidence" : `<br><span class="${c > 0 ? "pr" : "ab"}">${st} ${c > 0 ? "▲ present" : "▼ absent"}</span>`) +
    (best ? `<br><span class="hl">◆ strongest image</span>` : "");
  renderWL(vp);
}

function renderWL(vp) {
  vp.ov.bl.innerHTML = `W ${Math.round(S.wl.w)}  L ${Math.round(S.wl.l)}${S.wl.invert ? "  INV" : ""}<br>Zoom ${vp.zoom.toFixed(2)}×`;
}

function drawBar(vp) {
  const s = seriesOf(vp.slot), svg = vp.bar;
  const H = Math.max(40, vp.el.clientHeight - 16), n = s.n, h = H / n;
  const out = [`<rect class="track" x="0" y="0" width="14" height="${H}" rx="2"/>`];
  s.contrib.forEach((c, k) => {
    const a = Math.min(1, Math.abs(c) / S.item.scale);
    if (a < 0.06) return;
    out.push(`<rect x="2" y="${(k * h).toFixed(2)}" width="10" height="${Math.max(1, h - 0.6).toFixed(2)}" fill="rgb(${(c > 0 ? RED : BLUE).join(",")})" fill-opacity="${(0.25 + 0.75 * a).toFixed(2)}"/>`);
  });
  out.push(`<rect class="cur" x="0" y="${(vp.idx * h + h / 2 - 1).toFixed(2)}" width="14" height="2"/>`);
  svg.setAttribute("viewBox", `0 0 14 ${H}`);
  svg.style.height = `${H}px`;
  svg.innerHTML = out.join("");
}

async function loadImage(vp) {
  const token = ++vp.imgToken;
  const r = await fetch(`/api/image/${S.i}/${vp.slot}/${vp.idx}.png`);
  if (token !== vp.imgToken) return;
  if (r.status === 202) {
    const d = await r.json();
    vp.pending.hidden = false;
    vp.pending.textContent = `Retrieving images from Kaggle (${d.ready} of ${d.studies} studies ready).`;
    vp.img.removeAttribute("src");
    clearCanvas(vp);
    setTimeout(() => token === vp.imgToken && loadImage(vp), 2500);
    return;
  }
  if (!r.ok) {
    vp.pending.hidden = false;
    vp.pending.textContent = "This image couldn't be loaded. Check the terminal running review.py.";
    return;
  }
  const blob = await r.blob();
  if (token !== vp.imgToken) return;
  const old = vp.img.src;
  vp.img.src = URL.createObjectURL(blob);
  if (old.startsWith("blob:")) URL.revokeObjectURL(old);
  vp.pending.hidden = true;
  drawHeat(vp);
}

function setSlice(vp, slot, idx) {
  const s = seriesOf(slot);
  if (idx === undefined) idx = strongestIdx(slot); // entering a series: open its strongest-evidence image
  vp.slot = slot;
  vp.idx = Math.max(0, Math.min(s.n - 1, idx));
  renderVp(vp);
  if (vp.k === S.active) { markActiveThumb(); renderStatus(); }
}

function setActive(k) {
  S.active = k;
  S.vps.forEach((v) => v.el.classList.toggle("active", v.k === k && S.vps.length > 1));
  markActiveThumb();
  const vp = S.vps[k];
  if (vp) updateLocate(vp);
  renderStatus();
}

function stepSeries(d) {
  const vp = S.vps[S.active], slots = S.item.series.map((s) => s.slot);
  setSlice(vp, slots[(slots.indexOf(vp.slot) + d + slots.length) % slots.length]);
}

function strongest() {
  const it = S.item;
  let vp = S.vps.find((v) => v.slot === it.start.slot);
  if (!vp) vp = S.vps[S.active];
  setActive(vp.k);
  setSlice(vp, it.start.slot, it.start.idx);
}

function setLayout(n) {
  S.layout = n;
  document.querySelectorAll("[data-layout]").forEach((b) => b.classList.toggle("on", Number(b.dataset.layout) === n));
  if (S.item) buildViewports();
}

function setTool(t) {
  S.tool = t;
  document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
  S.vps.forEach((v) => { v.el.className = `vp tool-${t}${v.k === S.active && S.vps.length > 1 ? " active" : ""}`; });
  renderStatus();
}

// ---------------------------------------------------------------- window/level, zoom, pan
function applyWL() {
  const { w, l, invert } = S.wl;
  let slope = 255 / w, icpt = -(l - w / 2) / w;
  if (invert) { slope = -slope; icpt = 1 - icpt; }
  for (const f of document.querySelectorAll("#wl feFuncR, #wl feFuncG, #wl feFuncB")) {
    f.setAttribute("slope", slope.toFixed(5));
    f.setAttribute("intercept", icpt.toFixed(5));
  }
  $("#invert").setAttribute("aria-pressed", String(invert));
  S.vps.forEach(renderWL);
  renderStatus();
}

function applyTransform(vp) {
  const side = vp.sq.clientWidth;
  const maxPan = ((vp.zoom - 1) * side) / 2;
  vp.pan.x = Math.max(-maxPan, Math.min(maxPan, vp.pan.x));
  vp.pan.y = Math.max(-maxPan, Math.min(maxPan, vp.pan.y));
  vp.zoomer.style.transform = `translate(${vp.pan.x}px, ${vp.pan.y}px) scale(${vp.zoom})`;
  renderWL(vp);
}

function zoomAt(vp, f, cx = 0, cy = 0) {
  const z = Math.max(1, Math.min(8, vp.zoom * f));
  vp.pan.x = cx - ((cx - vp.pan.x) * z) / vp.zoom;
  vp.pan.y = cy - ((cy - vp.pan.y) * z) / vp.zoom;
  vp.zoom = z;
  if (z === 1) vp.pan = { x: 0, y: 0 };
  applyTransform(vp);
}

function resetAll() {
  S.wl = { w: 255, l: 127.5, invert: false };
  applyWL();
  for (const vp of S.vps) { vp.zoom = 1; vp.pan = { x: 0, y: 0 }; applyTransform(vp); }
}

function bindViewport(vp) {
  let drag = null, wheel = 0;
  vp.el.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".bar")) return;
    setActive(vp.k);
    drag = { x: e.clientX, y: e.clientY, w: S.wl.w, l: S.wl.l, zoom: vp.zoom, px: vp.pan.x, py: vp.pan.y };
    vp.el.setPointerCapture(e.pointerId);
  });
  vp.el.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (S.tool === "wl") {
      S.wl.w = Math.max(2, Math.min(510, drag.w + dx));
      S.wl.l = Math.max(-64, Math.min(320, drag.l - dy));
      applyWL();
    } else if (S.tool === "pan") {
      vp.pan.x = drag.px + dx;
      vp.pan.y = drag.py + dy;
      applyTransform(vp);
    } else {
      vp.zoom = Math.max(1, Math.min(8, drag.zoom * Math.exp(-dy * 0.01)));
      if (vp.zoom === 1) vp.pan = { x: 0, y: 0 };
      applyTransform(vp);
    }
  });
  const end = () => { drag = null; };
  vp.el.addEventListener("pointerup", end);
  vp.el.addEventListener("pointercancel", end);
  vp.el.addEventListener("dblclick", () => { vp.zoom = 1; vp.pan = { x: 0, y: 0 }; applyTransform(vp); });
  vp.el.addEventListener("wheel", (e) => {
    if (!S.item) return;
    e.preventDefault();
    setActive(vp.k);
    if (e.ctrlKey) {
      const b = vp.sq.getBoundingClientRect();
      zoomAt(vp, Math.exp(-e.deltaY * 0.01), e.clientX - b.left - b.width / 2, e.clientY - b.top - b.height / 2);
      return;
    }
    wheel += e.deltaY;
    if (Math.abs(wheel) >= 40) {
      setSlice(vp, vp.slot, vp.idx + Math.sign(wheel));
      wheel = 0;
    }
  }, { passive: false });
  vp.bar.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    setActive(vp.k);
    const b = vp.bar.getBoundingClientRect(), n = seriesOf(vp.slot).n;
    setSlice(vp, vp.slot, Math.floor(((e.clientY - b.top) / b.height) * n));
  });
}

// ---------------------------------------------------------------- highlight
function sizeCanvas(vp) {
  const side = vp.sq.clientWidth;
  vp.canvas.width = Math.round(side * dpr());
  vp.canvas.height = Math.round(side * dpr());
  return vp.canvas;
}

function clearCanvas(vp) {
  const c = sizeCanvas(vp);
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
  vp.outline = false;
  vp.where = null;
  vp.ov.br.innerHTML = "";
}

function sliceWeight(vp) {
  return Math.min(1, Math.abs(seriesOf(vp.slot).contrib[vp.idx]) / S.item.scale);
}

async function drawHeat(vp) {
  clearCanvas(vp);
  vp.note = "";
  if (!S.item || !vp.img.src) return;
  if (!S.heat) { vp.note = "Highlight is off (H)."; return updateLocate(vp); }
  if (S.item.negligible) { vp.note = "The model's call barely depends on any image, so there is no area to highlight."; return updateLocate(vp); }
  const weight = sliceWeight(vp);
  if (weight < 0.1) { vp.note = "This image barely affects the model's call. Press G for the strongest image."; return updateLocate(vp); }
  const key = `${S.i}/${vp.slot}/${vp.idx}`;
  if (S.fine[key]) return renderGrid(vp, S.fine[key].grid, S.fine[key].n, weight, true);
  if (!S.regions[key]) S.regions[key] = (await api(`/api/region/${key}`)).data.grid;
  if (key !== `${S.i}/${vp.slot}/${vp.idx}`) return; // moved on while loading
  renderGrid(vp, S.regions[key], 4, weight, false);
  fetchFine(vp, key, weight);
}

async function fetchFine(vp, key, weight) {
  vp.refining = true;
  renderStatus();
  for (let tries = 0; tries < 60; tries++) {
    const { status, data } = await api(`/api/fine/${key}`);
    if (status === 200) {
      S.fine[key] = data;
      vp.refining = false;
      if (key === `${S.i}/${vp.slot}/${vp.idx}` && S.heat) renderGrid(vp, data.grid, data.n, weight, true);
      renderStatus();
      return;
    }
    if (key !== `${S.i}/${vp.slot}/${vp.idx}`) break;
    await new Promise((r) => setTimeout(r, 2000)); // images still downloading
  }
  vp.refining = false;
  renderStatus();
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
      out[(y + 1) * (m + 2) + (x + 1)] = cells[y0 * n + x0] * (1 - fx) * (1 - fy) + cells[y0 * n + x1] * fx * (1 - fy)
        + cells[y1 * n + x0] * (1 - fx) * fy + cells[y1 * n + x1] * fx * fy;
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

function renderGrid(vp, cells, n, weight, fine) {
  const c = sizeCanvas(vp), g = c.getContext("2d"), W = c.width, H = c.height, k = dpr();
  g.clearRect(0, 0, W, H);
  vp.outline = false;
  vp.where = null;
  vp.note = "";
  const peak = Math.max(...cells.map(Math.abs)) || 1;
  const posPeak = Math.max(0, ...cells), negPeak = Math.max(0, ...cells.map((v) => -v));

  // soft heat layer: red toward present, blue (optional) toward absent
  const off = document.createElement("canvas");
  off.width = off.height = n;
  const px = off.getContext("2d").createImageData(n, n);
  cells.forEach((v, j) => {
    if (v < 0 && !S.counter) return;
    const a = Math.abs(v) / peak;
    const [r, gg, b] = v > 0 ? RED : BLUE;
    px.data.set([r, gg, b, Math.round(255 * S.alpha * 0.8 * Math.max(0.35, weight) * (a < 0.2 ? 0 : Math.pow(a, 1.2)))], j * 4);
  });
  off.getContext("2d").putImageData(px, 0, 0);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(off, 0, 0, n, n, 0, 0, W, H);

  const M = 64, w = M + 2;
  const toPx = (x, y) => [((x - 0.5) / M) * W, ((y - 0.5) / M) * H];
  const outline = (field, thr, rgb, dashed) => {
    const segs = contour(field, w, thr);
    if (!segs.length) return null;
    for (const [lw, col] of [[4.5 * k, "rgba(0,0,0,0.8)"], [2 * k, `rgb(${rgb.join(",")})`]]) {
      g.beginPath();
      g.setLineDash(dashed ? [5 * k, 4 * k] : []);
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

  if (!fine) { vp.note = "Quick map shown; the precise outline follows in a moment."; return updateLocate(vp); }
  if (posPeak < 0.2 * peak || posPeak < 0.01) {
    if (S.counter && negPeak >= 0.2 * peak) outline(upsample(cells.map((v) => Math.max(0, -v)), n, M), 0.5 * negPeak, BLUE, true);
    vp.note = `On this image the model's evidence points mostly <b class="ab-t">against</b> ${esc(S.item.noun)}${S.counter ? " (dashed blue)" : " (press C to show it)"}.`;
    vp.ov.br.innerHTML = S.counter ? `<span class="ab">┅ against</span>` : "";
    return updateLocate(vp);
  }
  const pos = upsample(cells.map((v) => Math.max(0, v)), n, M), thr = 0.5 * posPeak;
  const segs = outline(pos, thr, RED, false);
  if (S.counter && negPeak >= 0.2 * peak) outline(upsample(cells.map((v) => Math.max(0, -v)), n, M), 0.5 * negPeak, BLUE, true);
  let sx = 0, sy = 0, sw = 0, minx = Infinity, miny = Infinity;
  for (let y = 1; y <= M; y++) {
    for (let x = 1; x <= M; x++) {
      const v = pos[y * w + x];
      if (v > thr) { sx += x * v; sy += y * v; sw += v; minx = Math.min(minx, x); miny = Math.min(miny, y); }
    }
  }
  if (segs && sw > 0) {
    vp.outline = true;
    vp.where = place((sx / sw - 0.5) / M, (sy / sw - 0.5) / M);
    const [tx, ty] = toPx(minx, miny);
    const label = `${S.item.noun}`;
    g.font = `600 ${11 * k}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    const tw = g.measureText(label).width + 10 * k, th = 17 * k;
    const bx = Math.min(W - tw - 4, Math.max(4, tx)), by = Math.max(4, ty - th - 5 * k);
    g.fillStyle = "rgba(0,0,0,0.75)";
    g.fillRect(bx, by, tw, th);
    g.fillStyle = `rgb(${RED.join(",")})`;
    g.fillRect(bx, by, 2 * k, th);
    g.fillStyle = "#f2f2f2";
    g.fillText(label, bx + 6 * k, by + 12.5 * k);
    vp.ov.br.innerHTML = `<span class="pr">━ model evidence</span>${S.counter ? `<br><span class="ab">┅ against</span>` : ""}`;
  }
  updateLocate(vp);
}

function updateLocate(vp) {
  if (vp.k !== S.active) return;
  const el = $("#locate");
  if (vp.outline && vp.where) {
    el.innerHTML = `Outlined on ${CODE[vp.slot]} image ${vp.idx + 1}: the model's evidence for ${esc(S.item.noun)} is in the <b>${vp.where}</b> of the image.`;
  } else {
    el.innerHTML = vp.note || "";
  }
}

// ---------------------------------------------------------------- status bar, decisions
function renderStatus() {
  if (!S.item) return;
  const vp = S.vps[S.active];
  const tool = { wl: "Window/level", pan: "Pan", zoom: "Zoom" }[S.tool];
  $("#st-left").innerHTML = `Study <b>…${esc(S.item.study_short)}</b> · ${esc(S.item.label)}${vp ? ` · ${CODE[vp.slot]} Im <b>${vp.idx + 1}</b>/${seriesOf(vp.slot).n}` : ""}`;
  $("#st-mid").innerHTML = `Tool <b>${tool}</b> · W <b>${Math.round(S.wl.w)}</b> L <b>${Math.round(S.wl.l)}</b>${S.wl.invert ? " · inverted" : ""}`;
  const refining = S.vps.some((v) => v.refining);
  $("#st-right").innerHTML = `Highlight <b>${S.heat ? (refining ? "refining…" : "8×8 occlusion") : "off"}</b> · ${S.layout === 1 ? "1×1" : "2×2"}`;
}

async function decide(kind) {
  if (!S.item) return;
  const i = S.i;
  const { data } = await api("/api/decision", { i, decision: kind });
  S.items[i].decision = kind;
  renderTally(data.tally);
  toast(kind === "unsure" ? "Marked unsure" : `Saved: ${data.saved}`);
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

function redrawAll() { S.vps.forEach((vp) => { renderCorners(vp); drawHeat(vp); }); renderStatus(); }

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
  for (const k of ["accept", "deny", "unsure"]) {
    const b = $(`#${k}`);
    b.addEventListener("click", () => decide(k));
    if (k !== "unsure") {
      for (const ev of ["mouseenter", "focus"]) b.addEventListener(ev, () => S.item && showConsequence(k));
      for (const ev of ["mouseleave", "blur"]) b.addEventListener(ev, () => S.item && showConsequence(null));
    }
  }
  $("#skip").addEventListener("click", () => go(S.i + 1));
  $("#back").addEventListener("click", () => go(S.i - 1));
  document.querySelectorAll("[data-layout]").forEach((b) => b.addEventListener("click", () => setLayout(Number(b.dataset.layout))));
  document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
  $("#invert").addEventListener("click", () => { S.wl.invert = !S.wl.invert; applyWL(); });
  $("#reset").addEventListener("click", resetAll);
  $("#strongest").addEventListener("click", () => S.item && strongest());
  $("#heat-toggle").addEventListener("click", () => { S.heat = !S.heat; $("#heat-toggle").setAttribute("aria-pressed", String(S.heat)); redrawAll(); });
  $("#counter-toggle").addEventListener("click", () => { S.counter = !S.counter; $("#counter-toggle").setAttribute("aria-pressed", String(S.counter)); redrawAll(); });
  $("#alpha").addEventListener("input", (e) => { S.alpha = Number(e.target.value); S.vps.forEach(drawHeat); });
  $("#lang-toggle").addEventListener("click", () => { S.original = !S.original; renderReport(); });
  $("#help-toggle").addEventListener("click", () => toggleHelp());
  $("#help-close").addEventListener("click", () => toggleHelp(false));
  $("#help").addEventListener("click", (e) => { if (e.target.id === "help") toggleHelp(false); });
  $("#thumbs").addEventListener("click", (e) => {
    const b = e.target.closest(".thumb");
    if (!b || !S.item) return;
    const shown = S.vps.find((v) => v.slot === b.dataset.slot);
    if (S.layout === 4 && shown) { setActive(shown.k); return; }
    setSlice(S.vps[S.active], b.dataset.slot);
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") return toggleHelp(false);
    if (e.key === "?") { e.preventDefault(); return toggleHelp(); }
    if (!S.item) return;
    const k = e.key.toLowerCase(), vp = S.vps[S.active];
    const actions = {
      a: () => decide("accept"), d: () => decide("deny"), u: () => decide("unsure"),
      s: () => go(S.i + 1), b: () => go(S.i - 1), g: strongest,
      h: () => $("#heat-toggle").click(), c: () => $("#counter-toggle").click(),
      o: () => { if (!$("#lang-toggle").hidden) { S.original = !S.original; renderReport(); } },
      l: () => setLayout(S.layout === 1 ? 4 : 1),
      w: () => setTool("wl"), p: () => setTool("pan"), z: () => setTool("zoom"),
      i: () => { S.wl.invert = !S.wl.invert; applyWL(); }, r: resetAll, "0": resetAll,
      arrowleft: () => setSlice(vp, vp.slot, vp.idx - 1), arrowright: () => setSlice(vp, vp.slot, vp.idx + 1),
      arrowup: () => stepSeries(-1), arrowdown: () => stepSeries(1),
      tab: () => setActive((S.active + (e.shiftKey ? -1 : 1) + S.vps.length) % S.vps.length),
    };
    if (/^[1-4]$/.test(k) && S.item.series[Number(k) - 1]) actions[k] = () => setSlice(vp, S.item.series[Number(k) - 1].slot);
    if (actions[k]) { e.preventDefault(); actions[k](); }
  });
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (S.item) { layoutSquares(); S.vps.forEach((vp) => { drawBar(vp); drawHeat(vp); }); } }, 120);
  });
}

(async function init() {
  try {
    const { data } = await api("/api/session");
    applySession(data);
    bind();
    setLayout(location.hash === "#4" ? 4 : 1); // #4 in the URL opens the four-viewport layout
    setTool("wl");
    applyWL();
    if (!data.items.length) { showDone(); $("#done-text").textContent = "Nothing left to review for this selection."; return; }
    await go(data.first_open);
  } catch (err) {
    document.body.innerHTML = `<p style="padding:24px;color:#ddd;font:14px system-ui">The review server isn't answering: ${esc(err.message)}. Restart it with <code>python scripts/review.py</code>.</p>`;
  }
})();

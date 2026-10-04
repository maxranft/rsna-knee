"""Test the model on a real-world knee MRI from outside the competition, such as a study downloaded from a hospital
viewer. A test only: nothing here trains anything.

    python scripts/external_study.py "external scan"                      # the packaged submission model
    python scripts/external_study.py "external scan" --heads runs/<id>    # one run's 5 fold heads, or runs/ens-<id>
    python scripts/external_study.py "external scan" --slot sag_fs=5      # force a slot to series number 5 (or =none)
    python scripts/external_study.py "external scan" --fov-mm 150         # picture exports: the field of view, if known

Input: the viewer's download anywhere under <folder>/dicom/. Zips there are unpacked first. DICOM files need no .dcm
suffix (viewer exports often have none). A picture export (PNG/JPEG, one folder per series, e.g. 04_Sag_PD_FS/001.png)
also works, approximately: see "Picture exports" below.

Output in <folder>/results/: graphs/*.png (scripts/external_graphs.py, opened in Preview), report.html (written, not
opened), predictions.csv, series.csv, and study.npz + study.json for the review tool (python scripts/review.py
--external <folder>: the slices with the evidence highlight).

The study goes through the same code as the Kaggle test set (preprocess.choose_series / load_volume / to_uint8, then
features.encode, then Head), with one difference: the competition CSVs give each series' plane and fluid sensitivity
(the same bit as fat suppression in this dataset), and a hospital export does not, so both are read from the headers.
Plane comes from ImageOrientationPatient (agrees with the competition labels on all 2,214 surveyed series). Fat
suppression comes from SeriesDescription (99.5% agreement where there is one, 85% from RepetitionTime alone). Check the
slot thumbnails in the report and use --slot when a pick looks wrong.

Picture exports carry no headers, so three things are assumed: the plane and fat suppression come from the folder name,
the slices are in file-name order, and the pixel size comes from the field of view across the image's long side:
--fov-mm, else the number in <folder>/fov_mm.txt, else 160 mm (the training median). The scores are sensitive to it
(11% off moved them by up to 0.14 in a test), so read the real value off the viewer when possible. The contrast is the viewer's display window, not the scanner's values. The report says so; treat
its scores as approximate and prefer a DICOM download when the viewer offers one.

Evidence per finding is the review tool's slice occlusion, averaged over every head: the log-odds with every slice
minus without that slice (positive = the slice pushed toward present). It explains the model, not the knee.

Optional <folder>/truth.csv (finding,present), created blank on the first run: fill in 1/0 from the radiologist's
report to see the model next to the truth. One study gives no AUC, so read it as a sanity check.

Not part of the submission: a personal check of the model on a real scan. One study would not move a model trained
on 4,407, and heads trained with it could not go to Kaggle (external data must be free and public). Patient data
stays on this machine: the terminal output has no patient identifiers or predictions, and the folder is git-ignored. Never paste the report page, the images or truth.csv into a chat.
"""
import argparse
import base64
import html
import io
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee.features import encode, load_encoder  # noqa: E402
from knee.head import SLOT_NAMES, Head  # noqa: E402
from knee.preprocess import SLOTS, _to_grid, choose_series, load_volume, to_uint8  # noqa: E402
from review import NOTICEABLE, QUESTIONS, SLOT_TITLES  # noqa: E402

PACKAGED = ROOT / "build/rsna-knee-heads"
ARCH = ("tokens", "dim", "d", "layers", "labels", "feats")
NOT_DICOM = {".zip", ".exe", ".dll", ".htm", ".html", ".js", ".css", ".txt", ".pdf", ".jpg", ".jpeg", ".png", ".bmp",
             ".xml", ".ini", ".inf", ".json", ".csv", ".md", ".npz", ".py"}
PLANES = ["Sagittal", "Coronal", "Axial"]  # the patient axis the slice normal is closest to: x, y, z
PICTURES = {".png", ".jpg", ".jpeg"}
PLANE_WORDS = {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax|tra"}  # picture exports: plane from the folder name
LOCALIZER = re.compile(r"(?<![a-z])(loc|locali[sz]er|scout|survey|pilot|tri[ -]?pilot|3[ -]?pl(ane)?|cal|calibration)"
                       r"(?![a-z])", re.I)
# fat suppression from SeriesDescription; rules and order as validated on data/survey/headers.csv
NO_FS = re.compile(r"(?<![a-z])((no|non|w/o|without)[ _-]?(fs|fat[ _-]?sat)|nfs)(?![a-z])", re.I)
T1 = re.compile(r"(?<![a-z0-9])t1", re.I)
FS = re.compile(r"(?<![a-z])(fs|f/s|fse?[ _-]?ir|f[ _-]?sat|fat[ _-]?(sat|sup|supp|suppressed|suppression)|spair|spir"
                r"|stir|tirm|chess|water|we|dixon[ _-]?w|ideal[ _-]?w|sfs|mfs|pdfs|t2fs|fsfse|pdwfs|t2wfs|fspd|fst2|ir)"
                r"(?![a-z])", re.I)


# ---------------------------------------------------------------- reading the export
def unpack(dicom_dir):
    """Extract every zip under dicom/ next to itself, once."""
    for z in sorted(dicom_dir.rglob("*.zip")):
        dest = z.with_suffix("")
        if not dest.exists():
            print(f"unpacking {z.name}", flush=True)
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)


def orientation(ds):
    """ImageOrientationPatient, also from the functional groups of an enhanced (multi-frame) MR file."""
    if "ImageOrientationPatient" in ds:
        return ds.ImageOrientationPatient
    for seq in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        try:
            return getattr(ds, seq)[0].PlaneOrientationSequence[0].ImageOrientationPatient
        except (AttributeError, IndexError):
            pass
    return None


def plane_of(iop):
    v = np.array([float(a) for a in iop])
    return PLANES[int(np.argmax(np.abs(np.cross(v[:3], v[3:]))))]


def fat_suppressed(ds):
    """-> (1 if fat-suppressed / fluid-sensitive else 0, the rule that decided)."""
    d = str(ds.get("SeriesDescription", "") or "").lower()
    tr = float(ds.get("RepetitionTime", 0) or 0)
    if NO_FS.search(d):
        return 0, "description: no fat-sat"
    if T1.search(d):
        return 0, "description: T1"
    if FS.search(d):
        return 1, "description: fat-sat or STIR"
    if "IR" in re.findall(r"[A-Z]+", str(ds.get("ScanningSequence", ""))):
        return 1, "inversion recovery (STIR)"
    if 0 < tr < 1000:
        return 0, "short TR (T1-weighted)"
    if re.search(r"[a-z]", d):
        return 0, "description: no fat-sat named"
    return int(tr >= 1500), "no description: guessed from TR"


def scan(dicom_dir):
    """-> MR image series under dicom_dir: dicts with the competition's series columns plus what the report shows."""
    groups = defaultdict(list)
    for f in sorted(p for p in dicom_dir.rglob("*") if p.is_file()):
        if f.suffix.lower() in NOT_DICOM or f.name.upper() == "DICOMDIR" or f.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:  # not a DICOM file
            continue
        if ds.get("Modality") != "MR" or "Rows" not in ds or "SeriesInstanceUID" not in ds:
            continue
        iop = orientation(ds)
        groups[(str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID))].append((f, ds, plane_of(iop) if iop else None))
    series = []
    for (study, uid), items in groups.items():
        ds = items[0][1]
        planes = {p for _, _, p in items}
        images = sum(int(d.get("NumberOfFrames", 1) or 1) for _, d, _ in items)
        fs, rule = fat_suppressed(ds)
        desc = str(ds.get("SeriesDescription", "") or "")
        skip = ("localizer" if "LOCALIZER" in [str(t).upper() for t in ds.get("ImageType", [])] or LOCALIZER.search(desc)
                else "mixed orientations (localizer)" if len(planes) > 1
                else "no orientation in the headers" if None in planes
                else f"only {images} images" if images < 6 else "")
        series.append({"StudyInstanceUID": study, "SeriesInstanceUID": uid,
                       "SeriesNumber": int(ds.get("SeriesNumber", 0) or 0), "SeriesDescription": desc,
                       "Anatomical_Plane": planes.pop() if len(planes) == 1 else "mixed", "Fluid_Sensitive": fs,
                       "fs_rule": rule, "images": images, "skip": skip, "files": [f for f, _, _ in items]})
    return sorted(series, key=lambda s: s["SeriesNumber"])


def link_series(series, work):
    """Lay each series out as <work>/<SeriesInstanceUID>/<n>.dcm (symlinks): the layout preprocess reads."""
    shutil.rmtree(work, ignore_errors=True)
    for s in series:
        d = work / s["SeriesInstanceUID"]
        d.mkdir(parents=True)
        for i, f in enumerate(s["files"]):
            (d / f"{i:05d}.dcm").symlink_to(f.resolve())


def scan_pictures(root):
    """A picture export (one folder of PNG/JPEG slices per series) -> series dicts like scan() returns. The folder name
    ("04_Sag_PD_FS") gives the series number, the plane and fat suppression."""
    groups = defaultdict(list)
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in PICTURES and not f.name.startswith("."):
            groups[f.parent].append(f)
    series = []
    for d, files in groups.items():
        m = re.match(r"(\d+)[ _-]*(.*)", d.name)
        num, desc = (int(m[1]), m[2]) if m else (0, d.name)
        plane = next((p for p, rx in PLANE_WORDS.items() if re.search(rf"(?<![a-z])({rx})", desc.lower())), None)
        fs, rule = fat_suppressed({"SeriesDescription": desc})
        skip = ("localizer" if LOCALIZER.search(desc) else "no plane in the folder name" if plane is None
                else f"only {len(files)} images" if len(files) < 6 else "")
        files = sorted(files, key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f.name)])
        series.append({"StudyInstanceUID": "pictures", "SeriesInstanceUID": str(d), "SeriesNumber": num,
                       "SeriesDescription": desc, "Anatomical_Plane": plane or "unknown", "Fluid_Sensitive": fs,
                       "fs_rule": rule, "images": len(files), "skip": skip, "files": files})
    return sorted(series, key=lambda s: s["SeriesNumber"])


def load_pictures(files, fov_mm):
    """Picture slices -> float32 [D, H, W] on the training grid, with square pixels of fov_mm / long side."""
    slices = [np.asarray(Image.open(f).convert("L"), np.float32) for f in files]
    ps = fov_mm / max(slices[0].shape)
    return np.stack([_to_grid(a, (ps, ps)) for a in slices])


def choose_pictures(series):
    """choose_series' rule without headers: prefer 15-60 slices, then more slices, then the lower series number."""
    chosen = {}
    for name, prefs in SLOTS:
        for plane, fs in prefs:
            cands = [s for s in series if s["Anatomical_Plane"] == plane and s["Fluid_Sensitive"] == fs]
            if cands:
                chosen[name] = min(cands, key=lambda s: (not 15 <= s["images"] <= 60, -s["images"], s["SeriesNumber"]))
                break
    return chosen


def pick_slots(series, work, overrides):
    """{slot: series dict}: choose_series over the usable series (choose_pictures when work is None), then the --slot
    overrides (by SeriesNumber)."""
    usable = [s for s in series if not s["skip"]]
    if work is None:
        chosen = choose_pictures(usable)
    else:
        by_uid = {s["SeriesInstanceUID"]: s for s in series}
        chosen = {k: by_uid[u] for k, u in choose_series(usable, work).items()}
    for slot, num in overrides.items():
        if num == "none":
            chosen.pop(slot, None)
            continue
        hits = [s for s in series if str(s["SeriesNumber"]) == num]
        if not hits:
            sys.exit(f"--slot {slot}={num}: no series number {num} (see series.csv)")
        chosen[slot] = hits[0]
    return {k: chosen[k] for k in SLOT_NAMES if k in chosen}


# ---------------------------------------------------------------- the model
def member_runs(path):
    if path.name.startswith("ens-"):
        return [ROOT / "runs" / m for m in json.loads((path / "args.json").read_text())["members"]]
    if (path / "SOURCE_RUN.txt").exists():  # the packaged dataset (scripts/package_heads.py)
        return [ROOT / "runs" / m for m in (path / "SOURCE_RUN.txt").read_text().split()]
    return [path]


def load_heads(path):
    """-> (heads, config, PCA (mean, components) or None, training runs the heads came from).
    path: the packaged build/rsna-knee-heads, a run folder, or runs/ens-<id>."""
    runs = member_runs(path)
    dirs = runs if path.name.startswith("ens-") else [path]
    cfgs = [json.loads((d / "config.json").read_text()) for d in dirs]
    cfg = cfgs[0]
    assert all(c.get(k) == cfg.get(k) for c in cfgs for k in ARCH), "ensemble members differ in architecture"
    if cfg.get("canon"):
        sys.exit("these heads mirror left knees (--canon); laterality is not implemented here")
    heads = []
    for f in (f for d in dirs for f in sorted(d.glob("head_fold*.pt"))):
        h = Head(cfg["tokens"], cfg["dim"], cfg["d"], len(cfg["labels"]), cfg["layers"])
        h.load_state_dict(torch.load(f, map_location="cpu"))
        heads.append(h.eval())
    assert heads, f"no head_fold*.pt in {path}"
    pca = None
    if cfg["dim"] != 384:  # heads trained on a PCA-reduced cache (scripts/pca_cache.py)
        f = next(p for p in [path / "pca.npz", ROOT / cfg["feats"] / "pca.npz"] if p.exists())
        z = np.load(f)
        pca = (torch.as_tensor(z["mean"]).float(), torch.as_tensor(z["components"]).float())
    return heads, cfg, pca, runs


@torch.no_grad()
def predict(heads, x, slot, depth, chunk=64):
    """-> (probability per label, mean over heads as in the submission kernel; evidence [n_slices, n_labels], the
    log-odds with every slice minus without that slice, averaged over heads)."""
    n = x.shape[1]
    probs, ev = [], 0
    for h in heads:
        full = h(x, slot, depth, torch.zeros(1, n, dtype=torch.bool))[0]
        probs.append(torch.sigmoid(full))
        drops = []
        for s in range(0, n, chunk):
            idx = torch.arange(s, min(n, s + chunk))
            pad = torch.zeros(len(idx), n, dtype=torch.bool)
            pad[torch.arange(len(idx)), idx] = True
            drops.append(full - h(x.expand(len(idx), -1, -1, -1), slot.expand(len(idx), -1),
                                  depth.expand(len(idx), -1), pad))
        ev = ev + torch.cat(drops)
    return torch.stack(probs).mean(0).numpy(), (ev / len(heads)).numpy()


def reference_scores(runs, labels):
    """Out-of-fold probabilities of the training knees, averaged over the member runs: the yardstick for percentiles."""
    frames = [pd.read_csv(r / "oof.csv", index_col="StudyInstanceUID")[labels] for r in runs if (r / "oof.csv").exists()]
    return pd.concat(frames).groupby(level=0).mean() if frames else None


# ---------------------------------------------------------------- the report page
def png(arr, size=None):
    im = Image.fromarray(arr)
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def strip_svg(ev, bounds, scale, mark):
    """Evidence across every slice of every series: up (red) = toward present, down (blue) = toward absent."""
    w, h, n = 300, 44, len(ev)
    bw = w / n
    bars = []
    for i, v in enumerate(ev):
        bh = min(abs(v) / scale, 1) * (h / 2 - 2)
        y = h / 2 - bh if v > 0 else h / 2
        bars.append(f'<rect x="{i * bw:.1f}" y="{y:.1f}" width="{max(bw - 0.6, 0.6):.1f}" height="{bh:.1f}" '
                    f'class="{"up" if v > 0 else "down"}"/>')
    seps = "".join(f'<line x1="{b * bw:.1f}" x2="{b * bw:.1f}" y1="0" y2="{h}" class="sep"/>' for b in bounds[1:-1])
    tick = (f'<path d="M{(mark + 0.5) * bw:.1f} 0 l-3 -5 h6 z" class="tick"/>' if mark is not None else "")
    return (f'<svg viewBox="0 -6 {w} {h + 6}" class="strip" role="img" aria-label="evidence per slice">'
            f'<line x1="0" x2="{w}" y1="{h / 2}" y2="{h / 2}" class="axis"/>{seps}{"".join(bars)}{tick}</svg>')


def ordinal(k):
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def write_report(path, folder, preds, slots, arrays, series, model_note, truth_note):
    findings = []
    for r in preds:
        bar = f'<span class="bar"><i style="width:{r["probability"] * 100:.1f}%"></i></span>'
        pct = "–" if r["percentile"] is None else ordinal(int(r["percentile"]))
        truth = {1: '<b class="p">present</b>', 0: '<b class="a">absent</b>'}.get(r["truth"], '<span class="dim">–</span>')
        if r["top_slot"]:
            img = (f'<figure><img src="{png(arrays[r["top_slot"]][r["top_slice"]], 160)}" alt="">'
                   f'<figcaption>{SLOT_TITLES[r["top_slot"]]}, slice {r["top_slice"] + 1} of '
                   f'{len(arrays[r["top_slot"]])}</figcaption></figure>')
        else:
            img = '<p class="dim small">No single slice moves the score much.</p>'
        findings.append(f'<tr><th>{html.escape(r["finding"])}<small>{html.escape(QUESTIONS[r["finding"]][1])}</small></th>'
                        f'<td class="num">{r["probability"]:.0%} {bar}</td><td class="num">{pct}</td><td>{truth}</td>'
                        f'<td>{r["strip"]}</td><td>{img}</td></tr>')
    cards = []
    for name in SLOT_NAMES:
        if name not in slots:
            cards.append(f'<div class="card"><h3>{SLOT_TITLES[name]}</h3><p class="dim">No series in this slot. The '
                         f'model runs without it (it was trained with missing slots).</p></div>')
            continue
        s, a = slots[name], arrays[name]
        thumbs = "".join(f'<img src="{png(a[i], 128)}" alt="">' for i in np.linspace(0, len(a) - 1, 5)[1:4].round().astype(int))
        cards.append(f'<div class="card"><h3>{SLOT_TITLES[name]}</h3><p>Series {s["SeriesNumber"]} · '
                     f'{html.escape(s["SeriesDescription"]) or "<i>no description</i>"} · {len(a)} slices</p>'
                     f'<div class="thumbs">{thumbs}</div></div>')
    used = {s["SeriesInstanceUID"]: k for k, s in slots.items()}
    rows = "".join(f'<tr><td class="num">{s["SeriesNumber"]}</td><td>{html.escape(s["SeriesDescription"])}</td>'
                   f'<td>{s["Anatomical_Plane"]}</td><td>{"yes" if s["Fluid_Sensitive"] else "no"} '
                   f'<span class="dim">({s["fs_rule"]})</span></td><td class="num">{s["images"]}</td>'
                   f'<td>{SLOT_TITLES[used[s["SeriesInstanceUID"]]] if s["SeriesInstanceUID"] in used else ""}'
                   f'<span class="dim">{s["skip"] and "skipped: " + s["skip"]}</span></td></tr>' for s in series)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Real-world test</title><style>
:root {{ color-scheme: dark; --app: #0c0e10; --chrome: #14171a; --line: #262b30; --text: #dadde0; --text-2: #9ca3aa;
  --text-3: #6c737a; --accent: #d4a72c; --present: #e66767; --absent: #3987e5;
  --ui: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
body {{ margin: 0; background: var(--app); color: var(--text); font: 14px/1.5 var(--ui); font-variant-numeric: tabular-nums; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 15px; margin: 32px 0 8px; }}
h3 {{ font-size: 13px; margin: 0 0 4px; color: var(--text-2); font-weight: 600; }}
.dim {{ color: var(--text-3); }} .small {{ font-size: 12px; }} .note {{ color: var(--text-2); max-width: 75ch; }}
.callout {{ border-left: 3px solid var(--accent); padding: 8px 12px; background: var(--chrome); max-width: 75ch; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
thead th {{ color: var(--text-3); font-size: 12px; font-weight: 500; }}
tbody th {{ font-weight: 600; white-space: nowrap; }} th small {{ display: block; color: var(--text-3); font-weight: 400; }}
.num {{ text-align: right; white-space: nowrap; }}
.bar {{ display: inline-block; width: 64px; height: 6px; background: var(--line); border-radius: 3px; margin-left: 6px;
  vertical-align: middle; }} .bar i {{ display: block; height: 100%; background: var(--text-2); border-radius: 3px; }}
.p {{ color: var(--present); font-weight: 600; }} .a {{ color: var(--absent); font-weight: 600; }}
.strip {{ width: 300px; height: 50px; display: block; }} .strip .axis {{ stroke: var(--line); }}
.strip .sep {{ stroke: var(--text-3); stroke-dasharray: 2 2; }}
.strip .up {{ fill: var(--present); }} .strip .down {{ fill: var(--absent); }} .strip .tick {{ fill: var(--accent); }}
figure {{ margin: 0; }} figure img {{ width: 120px; height: 120px; display: block; border-radius: 2px; }}
figcaption {{ font-size: 11px; color: var(--text-3); max-width: 140px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.card {{ background: var(--chrome); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }}
.card p {{ margin: 0 0 8px; }} .thumbs {{ display: flex; gap: 4px; }} .thumbs img {{ width: 33%; border-radius: 2px; }}
code {{ background: var(--chrome); padding: 1px 5px; border-radius: 3px; }}
</style></head><body><main>
<h1>Real-world test: {html.escape(folder.name)}</h1>
<p class="dim">{model_note} · {datetime.now():%Y-%m-%d %H:%M}. Not a diagnosis: this shows what a competition model
does with one scan from outside the competition.</p>
{truth_note}
<h2>Findings</h2>
<p class="note">The <b>score</b> is the model's probability. The <b>percentile</b> ranks it among the training knees'
out-of-fold scores, which reads better than the raw score because base rates differ a lot between findings. The
<b>strip</b> runs over every slice of every series (dashed lines between series): red bars pushed the score toward
present, blue toward absent. The image is the slice that pushed hardest toward present (amber tick).</p>
<div class="wrap"><table><thead><tr><th>Finding</th><th class="num">Score</th><th class="num">Percentile</th>
<th>Radiologist</th><th>Evidence by slice</th><th>Strongest slice</th></tr></thead><tbody>{"".join(findings)}</tbody></table></div>
<h2>Series the model read</h2>
<p class="note">Check these. Each slot should show the plane and contrast in its title. If one is wrong, rerun with
<code>--slot sag_fs=&lt;series number&gt;</code> (slots: sag_fs, cor_fs, ax_fs, t1; <code>=none</code> drops one).</p>
<div class="cards">{"".join(cards)}</div>
<h2>Every series in the download</h2>
<p class="note">Plane comes from the image orientation. Fat suppression comes from the series name: the model's
"fluid-sensitive" slots are fat-suppressed series (PD/T2 fat-sat, STIR).</p>
<div class="wrap"><table><thead><tr><th class="num">#</th><th>Description</th><th>Plane</th><th>Fat-suppressed</th>
<th class="num">Images</th><th>Used as</th></tr></thead><tbody>{rows}</tbody></table></div>
</main></body></html>"""
    path.write_text(page)


def truth_summary(preds):
    """Within this one knee: do the findings the radiologist reported rank above the ones they ruled out?"""
    pos = [r["percentile"] for r in preds if r["truth"] == 1 and r["percentile"] is not None]
    neg = [r["percentile"] for r in preds if r["truth"] == 0 and r["percentile"] is not None]
    if not pos and not neg:
        return ('<p class="callout">No truth yet. Fill in <code>truth.csv</code> (1 = present, 0 = absent, from the '
                "radiologist's report) and run again to see the model next to it.</p>")
    if not pos or not neg:
        return f'<p class="callout">Truth filled in for {len(pos) + len(neg)} findings, all one way, so there is nothing to rank.</p>'
    pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return (f'<p class="callout">Radiologist: {len(pos)} present, {len(neg)} absent. Ranked by percentile, the model puts '
            f'a present finding above an absent one in {pairs:g} of {len(pos) * len(neg)} pairs '
            f'({pairs / (len(pos) * len(neg)):.0%}; 50% is chance). One knee, so this is a sanity check, not a score.</p>')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path, help='e.g. "external scan"; the DICOM download goes in <folder>/dicom/')
    ap.add_argument("--heads", type=Path, default=PACKAGED, help="default: the packaged submission model")
    ap.add_argument("--slot", action="append", default=[], metavar="SLOT=SERIES_NUMBER")
    ap.add_argument("--fov-mm", type=float, help="picture exports: field of view across the long side (default: "
                    "<folder>/fov_mm.txt, else 160)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    folder = args.folder if args.folder.is_absolute() else Path.cwd() / args.folder
    dicom_dir, out = folder / "dicom", folder / "results"
    if args.fov_mm is None:
        f = folder / "fov_mm.txt"
        args.fov_mm = float(f.read_text().split()[0]) if f.exists() else 160.0
    overrides = dict(s.split("=", 1) for s in args.slot)
    if bad := set(overrides) - set(SLOT_NAMES):
        sys.exit(f"unknown slot {bad}; slots are {SLOT_NAMES}")
    dicom_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)

    unpack(dicom_dir)
    series = scan(dicom_dir)
    pictures = not series
    if pictures:
        series = scan_pictures(dicom_dir)
    if not series:
        sys.exit(f"no MR DICOM files or picture folders under {dicom_dir}. Download the study from the viewer (DICOM "
                 "if offered) and put the zip or folder there.")
    studies = pd.Series([s["StudyInstanceUID"] for s in series if not s["skip"]]).value_counts()
    if len(studies) > 1:
        print(f"{len(studies)} MR studies in the download; using the one with the most series ({studies.iloc[0]})")
    series = [s for s in series if s["StudyInstanceUID"] == (studies.index[0] if len(studies) else series[0]["StudyInstanceUID"])]
    skipped = [s for s in series if s["skip"]]
    if pictures:
        print(f"no DICOM: using the picture export (approximate: {args.fov_mm:g} mm field of view assumed, display "
              "contrast)", flush=True)
    print(f"{sum(len(s['files']) for s in series)} {'picture' if pictures else 'DICOM'} files, {len(series)} MR series"
          + (f" ({len(skipped)} skipped: {', '.join(sorted({s['skip'] for s in skipped}))})" if skipped else ""), flush=True)

    work = None if pictures else out / "_series"
    if work:
        link_series(series, work)
    slots = pick_slots(series, work, overrides)
    if not slots:
        sys.exit("no usable series for any slot; see the series list with --slot to force one")
    arrays = {}
    for name, s in slots.items():
        vol = load_pictures(s["files"], args.fov_mm) if pictures else load_volume(work / s["SeriesInstanceUID"])[0]
        arrays[name] = to_uint8(vol)
        print(f"  {name:6s} = series {s['SeriesNumber']} '{s['SeriesDescription']}' ({len(arrays[name])} slices)", flush=True)
    if work:
        shutil.rmtree(work)

    heads, cfg, pca, runs = load_heads(args.heads)
    labels = cfg["labels"]
    enc = load_encoder(args.device, half=False)
    names = list(arrays)
    tok = encode(enc, np.concatenate([arrays[s] for s in names]), args.device, batch=32, half=False).cpu()
    if pca is not None:  # same projection as the training cache: (x - mean) @ components.T
        tok = (tok - pca[0]) @ pca[1].T
    x = tok.half().unsqueeze(0)  # the training cache is float16
    slot = torch.as_tensor(np.concatenate([np.full(len(arrays[s]), SLOT_NAMES.index(s)) for s in names])).long().unsqueeze(0)
    depth = torch.as_tensor(np.concatenate([np.linspace(0, 1, len(arrays[s])) for s in names]), dtype=torch.float32).unsqueeze(0)
    t1 = time.time()
    prob, ev = predict(heads, x, slot, depth)
    print(f"encoded {x.shape[1]} slices, ran {len(heads)} heads with slice occlusion ({time.time() - t1:.0f} s)", flush=True)

    ref = reference_scores(runs, labels)
    truth_csv = folder / "truth.csv"
    if not truth_csv.exists():
        truth_csv.write_text("finding,present\n" + "".join(f"{l},\n" for l in labels))
    truth = {r.finding: int(r.present) for r in pd.read_csv(truth_csv, dtype=str).itertuples()
             if str(r.present).strip() in {"0", "1"}}
    bounds = np.cumsum([0] + [len(arrays[s]) for s in names])
    scale = max(float(np.abs(ev).max()), NOTICEABLE)
    preds = []
    for k, label in enumerate(labels):
        e = ev[:, k]
        top = int(e.argmax()) if e.max() >= NOTICEABLE else None
        s_i = None if top is None else int(np.searchsorted(bounds, top, side="right") - 1)
        preds.append({"finding": label, "probability": float(prob[k]),
                      "percentile": None if ref is None else float((ref[label] < prob[k]).mean() * 100),
                      "truth": truth.get(label), "top_slot": None if top is None else names[s_i],
                      "top_slice": None if top is None else top - int(bounds[s_i]),
                      "top_evidence": None if top is None else float(e[top]),
                      "strip": strip_svg(e, bounds, scale, top)})
    pd.DataFrame([{k: v for k, v in r.items() if k != "strip"} for r in preds]).to_csv(out / "predictions.csv", index=False)
    # everything the review tool's --external mode shows, so it opens without re-reading the DICOM
    source = args.heads.resolve()
    model = (f"{len(heads)} heads from {len(runs)} training run{'s' * (len(runs) > 1)} "
             f"({source.relative_to(ROOT) if source.is_relative_to(ROOT) else source})")
    np.savez(out / "study.npz", tokens=x[0].numpy(), slot=slot[0].numpy(), depth=depth[0].numpy(), prob=prob, evidence=ev,
             **({} if ref is None else {"reference": ref.values.astype(np.float32)}),
             **{f"stack_{s}": arrays[s] for s in names})
    (out / "study.json").write_text(json.dumps({
        "heads": str(source), "model": model, "labels": labels, "slots": names,
        "series": {k: {"number": s["SeriesNumber"], "description": s["SeriesDescription"]} for k, s in slots.items()},
        "source": f"picture export, {args.fov_mm:g} mm field of view assumed" if pictures else "dicom",
        "created": datetime.now().isoformat(timespec="seconds")}, indent=1))
    used = {s["SeriesInstanceUID"]: k for k, s in slots.items()}
    pd.DataFrame([{**{k: v for k, v in s.items() if k not in ("files", "StudyInstanceUID", "SeriesInstanceUID")},
                   "slot": used.get(s["SeriesInstanceUID"], "")} for s in series]).to_csv(out / "series.csv", index=False)
    model_note = html.escape(model) + ("" if ref is None else f", percentiles against {len(ref):,} training knees")
    report = out / "report.html"
    caveat = ("" if not pictures else
              f'<p class="callout"><b>Approximate:</b> this is a picture export (PNG/JPEG), not DICOM. The pixel size is '
              f'assumed from a {args.fov_mm:g} mm field of view (<code>--fov-mm</code>), plane and fat suppression come '
              "from the folder names, and the contrast is the viewer's display window. Scores can shift from what the "
              "DICOM would give; download the DICOM if the viewer offers it.</p>")
    write_report(report, folder, preds, slots, arrays, series, model_note, caveat + truth_summary(preds))
    from external_graphs import write_graphs  # imported here: plotting is only needed at the end
    graphs = write_graphs(folder)
    print(f"graphs: {graphs[0].parent.relative_to(Path.cwd()) if graphs[0].is_relative_to(Path.cwd()) else graphs[0].parent} "
          f"(report page, not opened: {report.name}; {time.time() - t0:.0f} s total)")
    if not args.no_open:
        subprocess.run(["open", *map(str, graphs)])


if __name__ == "__main__":
    main()

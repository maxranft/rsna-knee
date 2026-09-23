"""Week-1 DICOM survey (CPU, runs on Kaggle where the data lives).

A. slices per series for every training series (directory listing only)
B. header survey on a random sample of studies: surviving tags, transfer syntaxes, geometry, scanner,
   laterality candidates
C. decode benchmark per transfer syntax x decoder plugin (I/O timed separately from decode)
D. parallel throughput with the best available decoder -> estimates test-time decode cost

Outputs land in /kaggle/working: series_counts.csv, headers.csv, tags.csv, decode_bench.csv, summary.json
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-openjpeg",
                "pylibjpeg-libjpeg", "python-gdcm"], check=False)

import io
import json
import os
import random
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import pixel_array

SEED = 0
N_HEADER_STUDIES = 400
N_BENCH_PER_SYNTAX = 60
N_PARALLEL_FILES = 1200
PLUGINS = ["", "gdcm", "pylibjpeg", "pillow"]  # "" = pydicom's automatic choice
OUT = Path("/kaggle/working")


def find_root():
    if os.environ.get("SURVEY_ROOT"):  # local smoke test
        return Path(os.environ["SURVEY_ROOT"])
    for p in [Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")]:
        if (p / "train_series.csv").exists():
            return p
    for p in Path("/kaggle/input").glob("*/*"):  # two levels only; recursive glob over 800k files is slow
        if (p / "train_series.csv").exists():
            return p
    for p in Path("/kaggle/input").glob("*"):
        if (p / "train_series.csv").exists():
            return p
    raise FileNotFoundError("competition data not mounted")


ROOT = find_root()
SPLIT = os.environ.get("SURVEY_SPLIT", "train")
OUT = Path(os.environ.get("SURVEY_OUT", OUT))
print("data root:", ROOT, SPLIT, flush=True)
series = pd.read_csv(ROOT / f"{SPLIT}_series.csv")
rng = random.Random(SEED)
summary = {"root": str(ROOT), "cpu_count": os.cpu_count(), "pydicom": pydicom.__version__}

# ---- A. slices per series ------------------------------------------------------------------
t0 = time.time()
def series_dir(r):
    return ROOT / f"{SPLIT}_series" / r.StudyInstanceUID / r.SeriesInstanceUID
series["n_slices"] = [sum(1 for e in os.scandir(series_dir(r)) if e.name.endswith(".dcm"))
                      if series_dir(r).exists() else -1 for r in series.itertuples()]
series.to_csv(OUT / "series_counts.csv", index=False)
summary["listing_seconds"] = round(time.time() - t0, 1)
summary["missing_series_dirs"] = int((series.n_slices < 0).sum())
summary["slices_per_series"] = series.n_slices[series.n_slices >= 0].describe().round(1).to_dict()
summary["total_slices"] = int(series.n_slices.clip(lower=0).sum())
print("A done", summary["listing_seconds"], "s", flush=True)

# ---- B. header survey ----------------------------------------------------------------------
FIELDS = ["Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
          "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "Laterality",
          "ImageLaterality", "BodyPartExamined", "ImageOrientationPatient", "ImagePositionPatient",
          "InstanceNumber", "BitsAllocated", "BitsStored", "PhotometricInterpretation",
          "SeriesDescription", "RepetitionTime", "EchoTime", "ScanningSequence", "PatientPosition"]
studies = sorted(series.StudyInstanceUID.unique())
sample = set(rng.sample(studies, min(N_HEADER_STUDIES, len(studies))))
tag_counter, rows, files_by_syntax = Counter(), [], {}
n_headers = 0
for r in series[series.StudyInstanceUID.isin(sample) & (series.n_slices > 0)].itertuples():
    files = sorted(series_dir(r).glob("*.dcm"))
    for f in {files[0], files[len(files) // 2]}:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        n_headers += 1
        tag_counter.update(e.keyword or str(e.tag) for e in ds)
        ts = str(ds.file_meta.TransferSyntaxUID.name)
        row = {"StudyInstanceUID": r.StudyInstanceUID, "SeriesInstanceUID": r.SeriesInstanceUID,
               "plane": r.Anatomical_Plane, "fluid": r.Fluid_Sensitive, "transfer_syntax": ts,
               "file_bytes": f.stat().st_size}
        for k in FIELDS:
            v = ds.get(k)
            row[k] = None if v is None else (list(v) if isinstance(v, pydicom.multival.MultiValue) else str(v))
        rows.append(row)
    for f in files:
        files_by_syntax.setdefault(ts, []).append(f)
hdr = pd.DataFrame(rows)
hdr.to_csv(OUT / "headers.csv", index=False)
tags = pd.DataFrame(tag_counter.most_common(), columns=["tag", "count"]).assign(share=lambda d: d["count"] / n_headers)
tags.to_csv(OUT / "tags.csv", index=False)
summary["headers_read"] = n_headers
summary["transfer_syntax_share"] = hdr.transfer_syntax.value_counts(normalize=True).round(3).to_dict()
summary["laterality_tags_present"] = {k: int(hdr[k].notna().sum()) for k in ["Laterality", "ImageLaterality", "BodyPartExamined"]}
summary["manufacturer"] = hdr.Manufacturer.value_counts().head(10).to_dict()
summary["field_strength"] = hdr.MagneticFieldStrength.value_counts().head(10).to_dict()
summary["rows_x_cols"] = (hdr.Rows.astype(str) + "x" + hdr.Columns.astype(str)).value_counts().head(10).to_dict()
print("B done", n_headers, "headers", flush=True)

# ---- C. decode benchmark -------------------------------------------------------------------
bench, bench_files = [], set()
for ts, files in files_by_syntax.items():
    for f in rng.sample(files, min(N_BENCH_PER_SYNTAX, len(files))):
        bench_files.add(f)
        t = time.perf_counter(); raw = f.read_bytes(); io_s = time.perf_counter() - t
        for p in PLUGINS:
            t = time.perf_counter()
            try:
                arr = pixel_array(io.BytesIO(raw), decoding_plugin=p)
                ok, shape, err = True, arr.shape, ""
            except Exception as e:  # plugin missing or unsupported for this syntax
                ok, shape, err = False, None, type(e).__name__
            bench.append({"transfer_syntax": ts, "plugin": p or "auto", "io_ms": io_s * 1e3,
                          "decode_ms": (time.perf_counter() - t) * 1e3, "ok": ok,
                          "shape": str(shape), "err": err, "bytes": len(raw)})
bench = pd.DataFrame(bench)
bench.to_csv(OUT / "decode_bench.csv", index=False)
agg = bench[bench.ok].groupby(["transfer_syntax", "plugin"]).decode_ms.median().round(2)
summary["decode_median_ms"] = {f"{a} | {b}": v for (a, b), v in agg.items()}
summary["decode_failures"] = bench[~bench.ok].groupby(["transfer_syntax", "plugin"]).err.first().to_dict()
summary["decode_failures"] = {f"{a} | {b}": v for (a, b), v in summary["decode_failures"].items()}
summary["io_median_ms"] = round(float(bench.drop_duplicates(["transfer_syntax", "bytes", "io_ms"]).io_ms.median()), 2)
print("C done", flush=True)


# ---- D. parallel throughput (fresh files, cold I/O + auto decode) -------------------------
def load(f):
    return pixel_array(f).shape


# skip files read in C so the page cache doesn't flatter the number
pool_files = [f for fs in files_by_syntax.values() for f in fs if f not in bench_files]
rng.shuffle(pool_files)
pool_files = pool_files[:N_PARALLEL_FILES]
if pool_files:
    t = time.time()
    with get_context("fork").Pool(os.cpu_count()) as pool:  # fork: script has no __main__ guard
        shapes = pool.map(load, pool_files, chunksize=8)
    rate = len(pool_files) / (time.time() - t)
    summary["parallel_files_per_s"] = round(rate, 1)
    # 4 slots x median slices, ~1300 test studies
    med = float(series.n_slices[series.n_slices > 0].median())
    summary["est_test_decode_minutes_4slots_all_slices"] = round(1300 * 4 * med / rate / 60, 1)
print("D done", flush=True)

(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))

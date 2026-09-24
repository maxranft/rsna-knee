"""DICOM study -> 4 slot volumes (uint8, fixed mm grid). Shared by the preprocessing kernel and the
submission notebook, so train and test go through identical code.

Slots (docs/labels.md): 0 sagittal FS, 1 coronal FS, 2 axial FS, 3 non-FS T1 (coronal, else sagittal).
Per series: sort slices along the slice normal, drop duplicate positions (multi-echo), apply the modality LUT,
resample in-plane to MM_PER_PX and centre crop/pad to SIZE, cap depth at MAX_SLICES by even subsampling,
percentile-normalise to uint8. No left/right flips anywhere.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut

SIZE = 256
MM_PER_PX = 0.7  # 256 px -> 179 mm; field of view median is 160 mm
MAX_SLICES = 40
PCT = (0.5, 99.5)
IO_THREADS = 8

SLOTS = [  # (name, [(plane, fluid_sensitive), ...] in order of preference)
    ("sag_fs", [("Sagittal", 1)]),
    ("cor_fs", [("Coronal", 1)]),
    ("ax_fs", [("Axial", 1)]),
    ("t1", [("Coronal", 0), ("Sagittal", 0)]),
]


def _read(f):
    return pydicom.dcmread(f)


def _first_header(series_dir):
    f = min(Path(series_dir).glob("*.dcm"))
    return pydicom.dcmread(f, stop_before_pixels=True), sum(1 for _ in Path(series_dir).glob("*.dcm"))


def choose_series(study_series, series_root):
    """study_series: rows of {SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive} for one study.
    Returns {slot_name: SeriesInstanceUID}. Among candidates prefer 2D stacks (15-60 slices),
    then the finest in-plane pixel spacing, then more slices."""
    headers = {}
    chosen = {}
    for name, prefs in SLOTS:
        for plane, fs in prefs:
            cands = [r for r in study_series if r["Anatomical_Plane"] == plane and r["Fluid_Sensitive"] == fs]
            if not cands:
                continue
            scored = []
            for r in cands:
                uid = r["SeriesInstanceUID"]
                if uid not in headers:
                    headers[uid] = _first_header(Path(series_root) / uid)
                ds, n = headers[uid]
                ps = float(ds.PixelSpacing[0]) if "PixelSpacing" in ds else 1.0
                scored.append(((not 15 <= n <= 60), ps, -n, uid))
            chosen[name] = min(scored)[3]
            break
    return chosen


def load_volume(series_dir, pool=None):
    """-> (float32 [D, H, W] sorted along the slice normal, info dict). All slices share a size after
    resampling, so volumes are assembled after the in-plane step."""
    files = sorted(Path(series_dir).glob("*.dcm"))
    dss = list(pool.map(_read, files)) if pool else [_read(f) for f in files]
    info = {"n_files": len(dss), "multiframe": False, "dup_positions": 0}
    d0 = dss[0]
    if len(dss) == 1 and int(d0.get("NumberOfFrames", 1) or 1) > 1:
        frames = apply_modality_lut(d0.pixel_array, d0).astype(np.float32)
        info["multiframe"] = True
        slices, spacing = list(frames), d0
    else:
        iop = np.array(d0.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
        normal = np.cross(iop[:3], iop[3:])
        keyed = {}
        for d in dss:
            pos = (round(float(normal @ np.array(d.ImagePositionPatient, float)), 2)
                   if "ImagePositionPatient" in d else float(d.get("InstanceNumber", 0)))
            echo = int(d.get("EchoNumbers", 1) or 1)
            if pos not in keyed or echo < keyed[pos][0]:  # multi-echo: keep the first echo per position
                keyed[pos] = (echo, d)
        info["dup_positions"] = len(dss) - len(keyed)
        ordered = [keyed[p][1] for p in sorted(keyed)]
        slices = [apply_modality_lut(d.pixel_array, d).astype(np.float32) for d in ordered]
        spacing = d0
    ps = [float(x) for x in spacing.get("PixelSpacing", [MM_PER_PX, MM_PER_PX])]
    vol = np.stack([_to_grid(s, ps) for s in slices])
    info.update({
        "n_slices": len(vol), "rows": int(d0.Rows), "cols": int(d0.Columns), "pixel_spacing": ps[0],
        "slice_thickness": float(d0.get("SliceThickness", 0) or 0),
        "desc": str(d0.get("SeriesDescription", "")), "laterality": str(d0.get("Laterality", "")),
        "manufacturer": str(d0.get("Manufacturer", "")),
        "field_strength": float(d0.get("MagneticFieldStrength", 0) or 0),
        "tr": float(d0.get("RepetitionTime", 0) or 0), "te": float(d0.get("EchoTime", 0) or 0),
    })
    return vol, info


def _to_grid(img, ps):
    """Resample to MM_PER_PX (row, col spacing ps) and centre crop/pad to SIZE x SIZE."""
    h, w = img.shape
    nh, nw = max(1, round(h * ps[0] / MM_PER_PX)), max(1, round(w * ps[1] / MM_PER_PX))
    interp = cv2.INTER_AREA if nh < h else cv2.INTER_LINEAR
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((SIZE, SIZE), np.float32)
    sy, sx = max(0, (nh - SIZE) // 2), max(0, (nw - SIZE) // 2)  # source crop start
    dy, dx = max(0, (SIZE - nh) // 2), max(0, (SIZE - nw) // 2)  # destination pad start
    ch, cw = min(SIZE, nh), min(SIZE, nw)
    out[dy:dy + ch, dx:dx + cw] = img[sy:sy + ch, sx:sx + cw]
    return out


def to_uint8(vol):
    """Evenly subsample depth to MAX_SLICES, then percentile-normalise the whole volume to uint8."""
    if len(vol) > MAX_SLICES:
        vol = vol[np.linspace(0, len(vol) - 1, MAX_SLICES).round().astype(int)]
    lo, hi = np.percentile(vol[:, ::4, ::4], PCT)
    return (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def process_study(study_uid, study_series, split_root):
    """-> ({slot_name: uint8 [D, SIZE, SIZE]}, [info dict per slot])."""
    series_root = Path(split_root) / study_uid
    chosen = choose_series(study_series, series_root)
    arrays, infos = {}, []
    with ThreadPoolExecutor(IO_THREADS) as pool:
        for name, uid in chosen.items():
            vol, info = load_volume(series_root / uid, pool)
            arrays[name] = to_uint8(vol)
            infos.append({"StudyInstanceUID": study_uid, "slot": name, "SeriesInstanceUID": uid,
                          "n_kept": len(arrays[name]), **info})
    return arrays, infos

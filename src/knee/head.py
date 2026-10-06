"""Study-level head over cached slice features (S3). Small enough to train on the Mac in minutes.

slice tokens [17, 384] --flatten, LayerNorm, Linear--> d   (flattening keeps the 4x4 grid layout: medial vs lateral)
  + slot embedding (sag_fs / cor_fs / ax_fs / t1) + depth embedding (relative slice position in its series)
  -> transformer over all slices of the study (padding mask)
  -> one attention-pooling query per label -> 12 logits
"""
import math

import numpy as np
import torch
import torch.nn as nn

SLOT_NAMES = ["sag_fs", "cor_fs", "ax_fs", "t1"]  # same order as preprocess.SLOTS
SLOT_PLANES = {"sag_fs": "Sagittal", "cor_fs": "Coronal", "ax_fs": "Axial"}  # t1 is Coronal or Sagittal, per study


def mirror(x, plane):
    """Lay one slot of a left knee out like a right knee, so medial is always on the same side (scripts/laterality.py).
    x [D, 1 + g*g, C]. Coronal / axial: reverse the grid columns (patient x). Sagittal: reverse the slice order.
    Token contents are not mirrored, only their layout. Labels are unchanged: Medial stays Medial."""
    if plane == "Sagittal":
        return x[::-1]
    g = math.isqrt(x.shape[1] - 1)
    grid = x[:, 1:].reshape(len(x), g, g, -1)[:, :, ::-1].reshape(len(x), g * g, -1)
    return np.concatenate([x[:, :1], grid], 1)


def fourier(depth, n=8):
    """depth in [0, 1], shape [B, N] -> [B, N, 2n]."""
    freqs = (2.0 ** torch.arange(n, device=depth.device)) * math.pi
    ang = depth.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class Head(nn.Module):
    """n_aux: extra report-derived outputs trained alongside the labels (S2); forward returns only the n_labels logits
    unless with_aux. slot_proj: one input projection per series slot instead of a shared one (the 4x4 grid means
    different anatomy in each plane). local: add each slice's neighbours in the same series before the transformer."""

    def __init__(self, tokens=17, dim=384, d=256, n_labels=12, layers=1, heads=4, drop=0.1, n_aux=0, slot_proj=False,
                 local=False):
        super().__init__()
        proj = lambda: nn.Sequential(nn.LayerNorm(tokens * dim), nn.Dropout(drop), nn.Linear(tokens * dim, d), nn.GELU())  # noqa: E731
        self.proj = nn.ModuleList([proj() for _ in SLOT_NAMES]) if slot_proj else proj()
        self.slot_emb = nn.Embedding(len(SLOT_NAMES), d)
        self.depth = nn.Linear(16, d)
        self.local = nn.Linear(2 * d, d) if local else None
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.n_labels = n_labels
        self.queries = nn.Parameter(torch.randn(n_labels + n_aux, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.w = nn.Parameter(torch.zeros(n_labels + n_aux, d))
        self.b = nn.Parameter(torch.zeros(n_labels + n_aux))

    @classmethod
    def from_config(cls, cfg):
        """A run's config.json -> an untrained Head of the same architecture (older configs lack the optional keys)."""
        return cls(cfg["tokens"], cfg["dim"], cfg["d"], len(cfg["labels"]), cfg["layers"], n_aux=len(cfg.get("aux", [])),
                   slot_proj=cfg.get("slot_proj", False), local=cfg.get("local", False))

    def forward(self, x, slot, depth, pad, return_attn=False, with_aux=False):
        """x [B, N, T, C] (float16 ok), slot [B, N] long, depth [B, N] float, pad [B, N] bool (True = padding).
        return_attn: also return each label's attention over slices [B, n_labels, N] (the review tool's 'where it looked')."""
        x = x.flatten(2).float()
        if isinstance(self.proj, nn.ModuleList):
            h = x.new_zeros(*x.shape[:2], self.slot_emb.embedding_dim)
            for s, proj in enumerate(self.proj):
                m = slot == s
                if m.any():
                    h[m] = proj(x[m])
        else:
            h = self.proj(x)
        if self.local is not None:  # previous / next slice of the same series; zero at series ends and padding
            same = (slot[:, 1:] == slot[:, :-1]) & ~pad[:, 1:] & ~pad[:, :-1]
            prev = torch.cat([h.new_zeros(len(h), 1, h.shape[2]), h[:, :-1] * same.unsqueeze(-1)], 1)
            nxt = torch.cat([h[:, 1:] * same.unsqueeze(-1), h.new_zeros(len(h), 1, h.shape[2])], 1)
            h = h + self.local(torch.cat([prev, nxt], -1))
        h = h + self.slot_emb(slot) + self.depth(fourier(depth))
        h = self.tf(h, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(len(h), -1, -1)
        z, attn = self.pool(q, h, h, key_padding_mask=pad, need_weights=return_attn)
        logits = (self.norm(z) * self.w).sum(-1) + self.b  # [B, n_labels + n_aux]
        if not with_aux:
            logits = logits[:, :self.n_labels]
            attn = attn[:, :self.n_labels] if return_attn else attn
        return (logits, attn) if return_attn else logits


class Studies(torch.utils.data.Dataset):
    """Study -> (tokens [N, T, C] float16, slot ids [N], depth [N], targets [12]).
    index: DataFrame(StudyInstanceUID, slot, shard, start, length); feat_paths: {shard: path to .npy [n, T, C]}.
    left: {uid: t1_plane} for left knees, which are mirrored into right-knee layout (None = no canonicalisation).
    slice_drop: training only, drop each slice with this probability (at least 4 kept per slot).
    Memmaps open lazily per worker: pickling an open memmap (spawned DataLoader workers) would copy it."""

    def __init__(self, uids, index, feat_paths, targets, train=False, slot_drop=0.1, seed=0, left=None, slice_drop=0.0):
        self.uids = list(uids)
        self.rows = {u: g.to_dict("records") for u, g in index[index.StudyInstanceUID.isin(set(self.uids))].groupby("StudyInstanceUID")}
        self.feat_paths, self.targets, self.train, self.slot_drop = dict(feat_paths), targets, train, slot_drop
        self.left = left or {}
        self.slice_drop = slice_drop
        self.seed, self.rng, self._feats = seed, None, None

    @property
    def feats(self):
        if self._feats is None:
            self._feats = {k: np.load(p, mmap_mode="r") for k, p in self.feat_paths.items()}
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed + (info.id if info else 0))
        return self._feats

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        feats = self.feats
        rows = self.rows.get(u, [])
        if self.train and len(rows) > 1:  # randomly drop a slot (never all of them)
            keep = [r for r in rows if self.rng.random() > self.slot_drop]
            rows = keep or rows
        xs, slots, depths = [], [], []
        for r in rows:
            a = np.asarray(feats[r["shard"]][r["start"]:r["start"] + r["length"]])
            if u in getattr(self, "left", {}):  # getattr: datasets pickled by an older version of this file
                a = np.ascontiguousarray(mirror(a, SLOT_PLANES.get(r["slot"], self.left[u])))
            depth = np.linspace(0, 1, len(a))
            if self.train and getattr(self, "slice_drop", 0) > 0:  # depth is kept, so dropped slices leave gaps
                keep = self.rng.random(len(a)) > self.slice_drop
                if keep.sum() >= 4:
                    a, depth = a[keep], depth[keep]
            xs.append(a)
            slots.append(np.full(len(a), SLOT_NAMES.index(r["slot"])))
            depths.append(depth)
        return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(slots)).long(),
                torch.from_numpy(np.concatenate(depths)).float(), torch.as_tensor(self.targets[u], dtype=torch.float32))


def slot_variants(x, slot, depth):
    """Leave-one-slot-out TTA: one study x [N, T, C], slot [N], depth [N] -> a padded batch (x, slot, depth, pad) of the
    full study plus the study without each of its slots (just the full study if it has one slot). Any device."""
    ss = slot.unique().tolist()
    keeps = [torch.ones_like(slot, dtype=torch.bool)] + ([slot != s for s in ss] if len(ss) > 1 else [])
    n = max(int(k.sum()) for k in keeps)
    xb, sb, db = x.new_zeros(len(keeps), n, *x.shape[1:]), slot.new_zeros(len(keeps), n), depth.new_zeros(len(keeps), n)
    pb = torch.ones(len(keeps), n, dtype=torch.bool, device=x.device)
    for i, k in enumerate(keeps):
        m = int(k.sum())
        xb[i, :m], sb[i, :m], db[i, :m], pb[i, :m] = x[k], slot[k], depth[k], False
    return xb, sb, db, pb


def collate(batch):
    n = max(len(b[0]) for b in batch)
    B, (T, C) = len(batch), batch[0][0].shape[1:]
    x = torch.zeros(B, n, T, C, dtype=batch[0][0].dtype)
    slot = torch.zeros(B, n, dtype=torch.long)
    depth = torch.zeros(B, n)
    pad = torch.ones(B, n, dtype=torch.bool)
    for i, (xi, si, di, _) in enumerate(batch):
        k = len(xi)
        x[i, :k], slot[i, :k], depth[i, :k], pad[i, :k] = xi, si, di, False
    return x, slot, depth, pad, torch.stack([b[3] for b in batch])

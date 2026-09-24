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

SLOTS = ["sag_fs", "cor_fs", "ax_fs", "t1"]


def fourier(depth, n=8):
    """depth in [0, 1], shape [B, N] -> [B, N, 2n]."""
    freqs = (2.0 ** torch.arange(n, device=depth.device)) * math.pi
    ang = depth.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class Head(nn.Module):
    def __init__(self, tokens=17, dim=384, d=256, n_labels=12, layers=1, heads=4, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(tokens * dim), nn.Dropout(drop), nn.Linear(tokens * dim, d), nn.GELU())
        self.slot_emb = nn.Embedding(len(SLOTS), d)
        self.depth = nn.Linear(16, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.w = nn.Parameter(torch.zeros(n_labels, d))
        self.b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x, slot, depth, pad):
        """x [B, N, T, C] (float16 ok), slot [B, N] long, depth [B, N] float, pad [B, N] bool (True = padding)."""
        h = self.proj(x.flatten(2).float()) + self.slot_emb(slot) + self.depth(fourier(depth))
        h = self.tf(h, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(len(h), -1, -1)
        z, _ = self.pool(q, h, h, key_padding_mask=pad)
        return (self.norm(z) * self.w).sum(-1) + self.b  # [B, n_labels]


class Studies(torch.utils.data.Dataset):
    """Study -> (tokens [N, T, C] float16, slot ids [N], depth [N], targets [12]).
    index: DataFrame(StudyInstanceUID, slot, shard, start, length); feat_paths: {shard: path to .npy [n, T, C]}.
    Memmaps open lazily per worker: pickling an open memmap (spawned DataLoader workers) would copy it."""

    def __init__(self, uids, index, feat_paths, targets, train=False, slot_drop=0.1, seed=0):
        self.uids = list(uids)
        self.rows = {u: g.to_dict("records") for u, g in index[index.StudyInstanceUID.isin(set(self.uids))].groupby("StudyInstanceUID")}
        self.feat_paths, self.targets, self.train, self.slot_drop = dict(feat_paths), targets, train, slot_drop
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
            xs.append(a)
            slots.append(np.full(len(a), SLOTS.index(r["slot"])))
            depths.append(np.linspace(0, 1, len(a)))
        return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(slots)).long(),
                torch.from_numpy(np.concatenate(depths)).float(), torch.as_tensor(self.targets[u], dtype=torch.float32))


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

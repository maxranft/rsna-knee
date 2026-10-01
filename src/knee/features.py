"""Frozen-encoder slice features (S3). uint8 [D, 256, 256] slices -> float16 [D, 1 + GRID*GRID, dim] tokens:
the CLS token plus the patch tokens average-pooled to a GRID x GRID map. The pooled grid keeps coarse spatial layout,
which medial vs lateral labels need. Shared by the feature-cache kernel and the submission notebook.
"""
import torch
import torch.nn.functional as F

ENCODER = "vit_small_patch14_dinov2.lvd142m"  # DINOv2 ViT-S/14, Apache-2.0
IMG = 224
GRID = 4
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(device="cuda", weights=None, half=True):
    """weights: path to a saved state_dict (offline submission); None downloads the pretrained weights.
    half=False keeps float32 (CPU use, e.g. the review tool's occlusion maps)."""
    import timm
    m = timm.create_model(ENCODER, pretrained=weights is None, num_classes=0, img_size=IMG)
    if weights:
        m.load_state_dict(torch.load(weights, map_location="cpu"))
    m = m.eval()
    return (m.half() if half else m).to(device)


@torch.no_grad()
def encode(model, slices_u8, device="cuda", batch=256, half=True):
    """slices_u8: uint8 tensor/array [N, H, W] -> float16 tensor [N, 1 + GRID*GRID, dim] on CPU (float32 if half=False)."""
    dt = torch.float16 if half else torch.float32
    x_all = torch.as_tensor(slices_u8)
    out = []
    for i in range(0, len(x_all), batch):
        x = x_all[i:i + batch].to(device, non_blocking=True).float().div_(255).unsqueeze(1)
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
        x = ((x.expand(-1, 3, -1, -1) - MEAN.to(device)) / STD.to(device)).to(dt)
        tok = model.forward_features(x)  # [B, prefix + P*P, dim]
        n_prefix = tok.shape[1] - (IMG // 14) ** 2
        cls, patches = tok[:, 0], tok[:, n_prefix:]
        p = IMG // 14
        grid = patches.transpose(1, 2).reshape(len(x), -1, p, p)
        grid = F.adaptive_avg_pool2d(grid.float(), GRID).flatten(2).transpose(1, 2).to(dt)  # [B, GRID*GRID, dim]
        out.append(torch.cat([cls.unsqueeze(1), grid], 1).cpu())
    return torch.cat(out)

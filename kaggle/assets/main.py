# ---- kernel entry point (appended to src/knee/features.py by scripts/build_kernels.py) ----
# Saves the frozen encoder's weights so the offline submission can load them (internet is off there).
m = load_encoder("cpu")
torch.save(m.float().state_dict(), "/kaggle/working/dinov2s_224.pt")
print(ENCODER, IMG, sum(p.numel() for p in m.parameters()), "params saved")

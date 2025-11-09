import numpy as np

data = np.load("ppe_rom_offline_data.npz", allow_pickle=True)
chi = data["chi"]; D = data["D"]

print("mean(chi) per mode (should be ~0):", np.abs(chi.mean(axis=0)).max())

s = np.linalg.svd(D, compute_uv=False)   # singular values only
cond = s.max() / max(s.min(), 1e-300)
print("cond(D):", cond)

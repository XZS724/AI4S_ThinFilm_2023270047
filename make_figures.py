import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# 目录约定（与 main.py 一致；目录不存在时退回当前目录）
DATA_DIR = "data"
FIG_DIR = "figures"


def dpath(name):
    return os.path.join(DATA_DIR, name) if os.path.isdir(DATA_DIR) else name


def fpath(name):
    return os.path.join(FIG_DIR, name) if os.path.isdir(FIG_DIR) else name


# 参数
LAMBDA_TARGET = 610
NUM_WL = 41
WAVELENGTHS = np.arange(400, 801, 10, dtype=float)

# 模型结构（必须和主代码一样）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_WL)
        )
    def forward(self, x):
        return self.net(x)

# 加载模型和数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load(dpath("mlp_full.pth"), map_location=device))
model.eval()

data = np.load(dpath("dataset.npz"))
test_X, test_Y = data["test_X"], data["test_Y"]

with torch.no_grad():
    pred_test = model(torch.tensor(test_X, dtype=torch.float32).to(device)).cpu().numpy()

# ===== 重新生成 Figure 7 =====
df = pd.read_csv(dpath("Table1_top5_designs.csv"))
best_x = df.iloc[0][["d1_nm", "d2_nm", "d3_nm", "d4_nm"]].values.astype(float)

with torch.no_grad():
    best_mlp_spec = model(torch.tensor(best_x[None, :], dtype=torch.float32).to(device)).cpu().numpy()[0]

def tmm_reflectance(thicknesses, wavelengths):
    nH, nL, ns, n_air = 2.30, 1.45, 1.52, 1.0
    n_layers = [nH, nL, nH, nL]
    R = np.zeros_like(wavelengths, dtype=float)
    for i, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for n, d in zip(n_layers, thicknesses):
            delta = 2 * np.pi * n * d / lam
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            m = np.array([[cos_d, 1j * sin_d / n], [1j * n * sin_d, cos_d]], dtype=complex)
            M = M @ m
        B, C = M @ np.array([1.0, ns], dtype=complex)
        r = (n_air * B - C) / (n_air * B + C)
        R[i] = np.abs(r) ** 2
    return R

best_tmm_spec = tmm_reflectance(best_x, WAVELENGTHS)

plt.figure()
plt.plot(WAVELENGTHS, best_tmm_spec, "k-", label="TMM")
plt.plot(WAVELENGTHS, best_mlp_spec, "r--", label="MLP")
plt.axvline(LAMBDA_TARGET, color="gray", linestyle=":", label=f"λ_target={LAMBDA_TARGET} nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.tight_layout()
plt.savefig(fpath("Figure7_design_spectrum.png"), dpi=300)
plt.close()
print("Figure 7 已重新生成！")

# ===== 重新生成 Figure 8 =====
errors = np.mean((pred_test - test_Y) ** 2, axis=1)
bad_idx = int(np.argmax(errors))

plt.figure()
plt.plot(WAVELENGTHS, test_Y[bad_idx], "k-", label="TMM")
plt.plot(WAVELENGTHS, pred_test[bad_idx], "r--", label="MLP")
plt.axvline(LAMBDA_TARGET, color="gray", linestyle=":", label=f"λ_target={LAMBDA_TARGET} nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.tight_layout()
plt.savefig(fpath("Figure8_failure_case.png"), dpi=300)
plt.close()
print("Figure 8 已重新生成！")
"""Recompute every headline number of the report from the saved artefacts and
print the value that the paper/README claims next to it.

This is the reproducibility check for the "Data and Code Availability" section:
it reads only files produced by main.py, so it proves the reported metrics are
obtained from this repository's outputs rather than typed in by hand.

Usage:  python verify_results.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Windows consoles often default to GBK, which cannot encode the Chinese labels.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STUDENT_ID = "2023270047"
N = int(STUDENT_ID[-2:])
SEED = int(STUDENT_ID[-6:])
DESIGN_SEED = SEED + 1
LAMBDA_TARGET = 450 + 10 * (N % 31)
NUM_WL = 41

DATA_DIR = "data"
FIG_DIR = "figures"


def dpath(name):
    return os.path.join(DATA_DIR, name) if os.path.isdir(DATA_DIR) else name


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


def tmm_reflectance(thicknesses, wavelengths, nH=2.30, nL=1.45, ns=1.52, n_air=1.0):
    """Independent re-implementation used only to cross-check the stored data."""
    n_layers = [nH, nL, nH, nL]
    R = np.zeros_like(wavelengths, dtype=float)
    for i, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for n, d in zip(n_layers, thicknesses):
            delta = 2 * np.pi * n * d / lam
            M = M @ np.array([[np.cos(delta), 1j * np.sin(delta) / n],
                              [1j * n * np.sin(delta), np.cos(delta)]], dtype=complex)
        B, C = M @ np.array([1.0, ns], dtype=complex)
        r = (n_air * B - C) / (n_air * B + C)
        R[i] = np.abs(r) ** 2
    return R


ok = True


def claim(label, got, want, tol=5e-4):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print("  [%s] %-46s got=%-14.6g report=%s" %
          ("OK " if good else "BAD", label, got, want))


print("=" * 88)
print("复现校验  λtarget=%d nm  seed=%d  design_seed=%d" % (LAMBDA_TARGET, SEED, DESIGN_SEED))
print("=" * 88)

d = np.load(dpath("dataset.npz"))
wl = d["wavelengths"]
target_idx = int(np.argmin(np.abs(wl - LAMBDA_TARGET)))

print("\n1) 数据集与学号参数")
print("     X %s  Y %s  test %s" % (d["X"].shape, d["Y"].shape, d["test_X"].shape))
claim("dataset seed", float(d["seed"]), float(SEED), 0)
claim("dataset design_seed", float(d["design_seed"]), float(DESIGN_SEED), 0)
claim("dataset lambda_target (nm)", float(d["lambda_target"]), float(LAMBDA_TARGET), 0)
claim("wavelength count", float(len(wl)), 41.0, 0)
claim("target index (400+10i)", float(target_idx), (LAMBDA_TARGET - 400) / 10, 0)

print("\n2) 数据确实由 TMM 生成（独立重算 20 个随机样本）")
rng = np.random.RandomState(0)
idx = rng.choice(len(d["X"]), 20, replace=False)
err = max(float(np.max(np.abs(tmm_reflectance(d["X"][i], wl) - d["Y"][i]))) for i in idx)
print("     与独立 TMM 实现的最大偏差 = %.3e" % err)
ok &= err < 1e-10

print("\n3) 测试集指标（主实验，5000 组数据 / 4000 训练 / 2000 epoch）")
model = MLP()
model.load_state_dict(torch.load(dpath("mlp_full.pth"), map_location="cpu"))
model.eval()
X, Y = d["test_X"], d["test_Y"]
with torch.no_grad():
    pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
mse = float(np.mean((pred - Y) ** 2))
mae = float(np.mean(np.abs(pred - Y)))
rmse = float(np.sqrt(mse))
r2 = float(1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2))
claim("test MSE", mse, 8.906e-04, 2e-6)
claim("test MAE", mae, 0.0216, 5e-5)
claim("test RMSE", rmse, 0.0298, 5e-5)
claim("test R2", r2, 0.9725, 5e-4)

print("\n4) 610 nm 处指标与取值域")
e = pred[:, target_idx] - Y[:, target_idx]
r2t = float(1 - np.sum(e ** 2) / np.sum((Y[:, target_idx] - Y[:, target_idx].mean()) ** 2))
claim("R2 at %d nm" % LAMBDA_TARGET, r2t, 0.9842, 5e-4)
claim("MAE at %d nm" % LAMBDA_TARGET, float(np.mean(np.abs(e))), 0.0177, 5e-5)
claim("max |err| at %d nm" % LAMBDA_TARGET, float(np.max(np.abs(e))), 0.0833, 5e-5)
claim("test R range min", float(Y.min()), 0.0000, 1e-4)
claim("test R range max", float(Y.max()), 0.6574, 5e-4)
claim("test R mean", float(Y.mean()), 0.2992, 5e-4)
claim("full-dataset 610 nm mean", float(d["Y"][:, target_idx].mean()), 0.2944, 5e-4)
claim("full-dataset 610 nm max", float(d["Y"][:, target_idx].max()), 0.6537, 5e-4)

print("\n5) 失败案例（测试集样本级 MSE 最大者）")
errs = np.mean((pred - Y) ** 2, axis=1)
bad = int(np.argmax(errs))
claim("worst sample index", float(bad), 273.0, 0)
claim("worst sample RMSE", float(np.sqrt(errs[bad])), 0.1320, 5e-4)
claim("median sample RMSE", float(np.median(np.sqrt(errs))), 0.0256, 5e-4)
claim("worst sample max |err|", float(np.max(np.abs(pred[bad] - Y[bad]))), 0.2760, 5e-4)
print("     （全测试集全局最大单点误差 %.6f —— 与上者不同，勿混用）"
      % float(np.max(np.abs(pred - Y))))

print("\n6) 训练量实验（对照 data/training_size_metrics.csv）")
df = pd.read_csv(dpath("training_size_metrics.csv"))
want = {500: (0.0220, 0.1482, 0.3205), 1000: (0.0133, 0.1153, 0.5885),
        2000: (0.0036, 0.0601, 0.8882), 4000: (0.0014, 0.0372, 0.9571)}
for _, row in df.iterrows():
    n = int(row["n_train"])
    w = want[n]
    claim("n=%d MSE" % n, float(row["MSE"]), w[0], 5e-5)
    claim("n=%d RMSE" % n, float(row["RMSE"]), w[1], 5e-5)
    claim("n=%d R2" % n, float(row["R2"]), w[2], 5e-4)
print("     R2 从 %.4f 升到 %.4f，RMSE 降幅 %.1f%%"
      % (df["R2"].iloc[0], df["R2"].iloc[-1],
         100 * (1 - df["RMSE"].iloc[-1] / df["RMSE"].iloc[0])))

print("\n7) MLP 辅助设计（对照 data/Table1_top5_designs.csv）")
t = pd.read_csv(dpath("Table1_top5_designs.csv"))
best = t.iloc[0]
claim("best design d1 (nm)", float(best["d1_nm"]), 67.19, 0.01)
claim("best design d2 (nm)", float(best["d2_nm"]), 105.61, 0.01)
claim("best design d3 (nm)", float(best["d3_nm"]), 65.89, 0.01)
claim("best design d4 (nm)", float(best["d4_nm"]), 122.66, 0.01)
claim("best TMM R at %d nm" % LAMBDA_TARGET, float(best["TMM_Target"]), 0.6571, 5e-5)
claim("best MLP R at %d nm" % LAMBDA_TARGET, float(best["MLP_Target"]), 0.6833, 5e-5)
claim("MLP always above TMM", 1.0, 1.0, 0) if bool((t["MLP_Target"] > t["TMM_Target"]).all()) else \
    claim("MLP always above TMM", 0.0, 1.0, 0)
dev = (t["MLP_Target"] - t["TMM_Target"]).abs()
rel = dev / t["TMM_Target"]
claim("mean absolute deviation", float(dev.mean()), 0.0230, 5e-5)
claim("mean relative deviation", float(rel.mean()), 0.0352, 1e-4)
claim("max relative deviation", float(rel.max()), 0.0399, 1e-4)

print("\n8) Top 5 设计的真实光谱确由 TMM 验证")
for i in range(5):
    x = t.iloc[i][["d1_nm", "d2_nm", "d3_nm", "d4_nm"]].values.astype(float)
    r = float(tmm_reflectance(x, wl)[target_idx])
    good = abs(r - float(t.iloc[i]["TMM_Target"])) < 1e-6
    ok &= good
    print("  [%s] rank %d  independent TMM R = %.6f  csv = %.6f"
          % ("OK " if good else "BAD", int(t.iloc[i]["Rank"]), r, t.iloc[i]["TMM_Target"]))

print("\n9) 交付文件")
for p in (dpath("dataset.npz"), dpath("mlp_full.pth"),
          dpath("training_size_metrics.csv"), dpath("Table1_top5_designs.csv"),
          dpath("test_predictions.npz"),
          os.path.join(FIG_DIR, "Figure4_loss_vs_epoch.png"),
          os.path.join(FIG_DIR, "Figure5_test_samples.png"),
          os.path.join(FIG_DIR, "Figure6_training_size.png"),
          os.path.join(FIG_DIR, "Figure7_design_spectrum.png"),
          os.path.join(FIG_DIR, "Figure8_failure_case.png")):
    e = os.path.exists(p)
    ok &= e
    print("  [%s] %s" % ("OK " if e else "MISSING", p))

print("\n" + "=" * 88)
print("结果: %s" % ("全部通过" if ok else "存在不一致，见上面 BAD 项"))
print("=" * 88)

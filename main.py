import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

# =====================
# 目录约定
#   data/     数据集与结果表
#   figures/  论文插图
#   两个目录不存在时退回脚本所在目录，便于直接在原工程目录里运行。
# =====================
DATA_DIR = "data"
FIG_DIR = "figures"


def dpath(name):
    return os.path.join(DATA_DIR, name) if os.path.isdir(DATA_DIR) else name


def fpath(name):
    return os.path.join(FIG_DIR, name) if os.path.isdir(FIG_DIR) else name


# =====================
# 个性化参数
# =====================
STUDENT_ID = "2023270047"
N = int(STUDENT_ID[-2:])                 # 47
SEED = int(STUDENT_ID[-6:])              # 270047
DESIGN_SEED = SEED + 1                   # 270048
LAMBDA_TARGET = 450 + 10 * (N % 31)      # 610 nm
GITHUB_USER = "XZS724"
REPO_NAME = "AI4S_ThinFilm_2023270047"

# =====================
# 光学与数据参数
# =====================
nH = 2.30
nL = 1.45
ns = 1.52
n_air = 1.0

THICKNESS_MIN = 40.0
THICKNESS_MAX = 180.0

WAVELENGTHS = np.arange(400, 801, 10, dtype=float)   # 41 点
NUM_WL = len(WAVELENGTHS)
TARGET_IDX = int(np.argmin(np.abs(WAVELENGTHS - LAMBDA_TARGET)))

# =====================
# 随机种子
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================
# TMM：Air / H / L / H / L / Glass
# =====================
def tmm_reflectance(thicknesses, wavelengths):
    """
    thicknesses: [d1, d2, d3, d4] nm
    层顺序：Air / H / L / H / L / Glass
    """
    n_layers = [nH, nL, nH, nL]
    R = np.zeros_like(wavelengths, dtype=float)

    for i, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)

        for n, d in zip(n_layers, thicknesses):
            delta = 2 * np.pi * n * d / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)

            # 正入射特征矩阵，eta = n
            m = np.array([
                [cos_d, 1j * sin_d / n],
                [1j * n * sin_d, cos_d]
            ], dtype=complex)

            M = M @ m

        # 出射介质为玻璃
        eta_s = ns
        B, C = M @ np.array([1.0, eta_s], dtype=complex)

        eta0 = n_air
        r = (eta0 * B - C) / (eta0 * B + C)
        R[i] = np.abs(r) ** 2

    return R

# =====================
# 数据生成
# =====================
def generate_dataset():
    set_seed(SEED)

    X = np.random.uniform(THICKNESS_MIN, THICKNESS_MAX, size=(5000, 4))
    Y = np.zeros((5000, NUM_WL), dtype=float)

    for i in range(5000):
        Y[i] = tmm_reflectance(X[i], WAVELENGTHS)

    # 固定随机排列
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(5000)
    X = X[perm]
    Y = Y[perm]

    # 固定划分 4000 / 500 / 500
    train_X, train_Y = X[:4000], Y[:4000]
    val_X, val_Y = X[4000:4500], Y[4000:4500]
    test_X, test_Y = X[4500:5000], Y[4500:5000]

    np.savez(
        dpath("dataset.npz"),
        X=X, Y=Y,
        train_X=train_X, train_Y=train_Y,
        val_X=val_X, val_Y=val_Y,
        test_X=test_X, test_Y=test_Y,
        wavelengths=WAVELENGTHS,
        seed=SEED,
        design_seed=DESIGN_SEED,
        lambda_target=LAMBDA_TARGET
    )

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# =====================
# MLP 模型
# =====================
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

# =====================
# 训练函数
# =====================
def train_mlp(train_X, train_Y, val_X, val_Y,
              seed=SEED, epochs=2000, batch_size=64, lr=1e-3):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_Y, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_Y, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)

        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, train_loss={train_loss:.6e}, val_loss={val_loss:.6e}")

    model.load_state_dict(best_state)
    return model, history

# =====================
# 评估函数
# =====================
def evaluate(model, X, Y):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    mse = np.mean((pred - Y) ** 2)
    mae = np.mean(np.abs(pred - Y))
    rmse = np.sqrt(mse)

    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return pred, {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

# =====================
# 主流程
# =====================
def main():
    # 1. 数据生成
    if not os.path.exists(dpath("dataset.npz")):
        print("生成数据集...")
        train_X, train_Y, val_X, val_Y, test_X, test_Y = generate_dataset()
    else:
        data = np.load(dpath("dataset.npz"))
        train_X, train_Y = data["train_X"], data["train_Y"]
        val_X, val_Y = data["val_X"], data["val_Y"]
        test_X, test_Y = data["test_X"], data["test_Y"]

    # 2. 训练完整 MLP
    print("训练完整 MLP...")
    model, history = train_mlp(
        train_X, train_Y, val_X, val_Y,
        seed=SEED, epochs=2000, batch_size=64, lr=1e-3
    )
    torch.save(model.state_dict(), dpath("mlp_full.pth"))

    # 3. 测试集评估
    pred_test, metrics = evaluate(model, test_X, test_Y)
    print("测试集指标:", metrics)
    np.savez(dpath("test_predictions.npz"), pred=pred_test, true=test_Y, X=test_X)

    # Figure 4: Loss vs epoch
    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath("Figure4_loss_vs_epoch.png"), dpi=300)
    plt.close()

    # Figure 5: 至少 3 个测试样本
    idxs = [0, 1, 2]
    plt.figure(figsize=(10, 8))
    for j, idx in enumerate(idxs):
        plt.subplot(3, 1, j + 1)
        plt.plot(WAVELENGTHS, test_Y[idx], "k-", label="TMM")
        plt.plot(WAVELENGTHS, pred_test[idx], "r--", label="MLP")
        plt.axvline(LAMBDA_TARGET, color="gray", linestyle=":", label=f"λ_target={LAMBDA_TARGET} nm")
        plt.ylabel("Reflectance")
        plt.legend()
        if j == 2:
            plt.xlabel("Wavelength (nm)")
    plt.tight_layout()
    plt.savefig(fpath("Figure5_test_samples.png"), dpi=300)
    plt.close()

    # 4. 训练量实验
    print("训练量实验...")
    sizes = [500, 1000, 2000, 4000]
    size_metrics = []

    for n in sizes:
        m, h = train_mlp(
            train_X[:n], train_Y[:n],
            val_X, val_Y,
            seed=SEED, epochs=1000, batch_size=64, lr=1e-3
        )
        _, met = evaluate(m, test_X, test_Y)
        size_metrics.append({"n_train": n, **met})
        print(n, met)
        torch.save(m.state_dict(), dpath(f"mlp_n{n}.pth"))

    pd.DataFrame(size_metrics).to_csv(dpath("training_size_metrics.csv"), index=False)

    # Figure 6
    plt.figure()
    plt.plot([m["n_train"] for m in size_metrics],
             [m["MSE"] for m in size_metrics], "o-")
    plt.xlabel("Training samples")
    plt.ylabel("Test MSE")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(fpath("Figure6_training_size.png"), dpi=300)
    plt.close()

    # 5. MLP 辅助设计
    print("MLP 辅助设计...")
    set_seed(DESIGN_SEED)

    candidates = np.random.uniform(
        THICKNESS_MIN, THICKNESS_MAX, size=(10000, 4)
    )

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        cand_pred = model(
            torch.tensor(candidates, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # 按 λ_target 处 MLP 预测反射率排序
    scores = cand_pred[:, TARGET_IDX]
    top10_idx = np.argsort(scores)[::-1][:10]

    top10_X = candidates[top10_idx]
    top10_mlp = scores[top10_idx]
    top10_tmm = np.array([
        tmm_reflectance(x, WAVELENGTHS)[TARGET_IDX] for x in top10_X
    ])

    # 用 TMM 重新验证后，按 TMM 目标反射率取 Top5
    order = np.argsort(top10_tmm)[::-1]
    top5_X = top10_X[order][:5]
    top5_mlp = top10_mlp[order][:5]
    top5_tmm = top10_tmm[order][:5]

    df = pd.DataFrame({
        "Rank": np.arange(1, 6),
        "d1_nm": top5_X[:, 0],
        "d2_nm": top5_X[:, 1],
        "d3_nm": top5_X[:, 2],
        "d4_nm": top5_X[:, 3],
        "MLP_Target": top5_mlp,
        "TMM_Target": top5_tmm,
    })
    df.to_csv(dpath("Table1_top5_designs.csv"), index=False)
    print(df)

    # Figure 7: 最佳设计光谱
    best_x = top5_X[0]
    best_mlp_spec = model(
        torch.tensor(best_x[None, :], dtype=torch.float32).to(device)
    ).cpu().numpy()[0]
    best_tmm_spec = tmm_reflectance(best_x, WAVELENGTHS)

    plt.figure()
    plt.plot(WAVELENGTHS, best_tmm_spec, "k-", label="TMM")
    plt.plot(WAVELENGTHS, best_mlp_spec, "r--", label="MLP")
    plt.axvline(LAMBDA_TARGET, color="gray", linestyle=":",
                label=f"λ_target={LAMBDA_TARGET} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath("Figure7_design_spectrum.png"), dpi=300)
    plt.close()

    # Figure 8: 失败案例，测试集中误差最大的样本
    errors = np.mean((pred_test - test_Y) ** 2, axis=1)
    bad_idx = int(np.argmax(errors))

    plt.figure()
    plt.plot(WAVELENGTHS, test_Y[bad_idx], "k-", label="TMM")
    plt.plot(WAVELENGTHS, pred_test[bad_idx], "r--", label="MLP")
    plt.axvline(LAMBDA_TARGET, color="gray", linestyle=":",
                label=f"λ_target={LAMBDA_TARGET} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath("Figure8_failure_case.png"), dpi=300)
    plt.close()

    # 保存配置
    config = {
        "student_id": STUDENT_ID,
        "N": N,
        "seed": SEED,
        "design_seed": DESIGN_SEED,
        "lambda_target_nm": LAMBDA_TARGET,
        "github_user": GITHUB_USER,
        "repo_name": REPO_NAME,
        "nH": nH,
        "nL": nL,
        "ns": ns,
        "thickness_range_nm": [THICKNESS_MIN, THICKNESS_MAX],
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "target_index": TARGET_IDX
    }

    with open(dpath("config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("完成。")

if __name__ == "__main__":
    main()
    
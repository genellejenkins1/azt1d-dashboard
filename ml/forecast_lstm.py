"""
forecast_lstm.py
----------------
Short-horizon glucose forecasting with a PyTorch LSTM. Given the last 60 minutes of
CGM + insulin + carb signal, predict the CGM trajectory 30 minutes ahead (6 steps).

Demonstrates: PyTorch, sequence modeling (LSTM), multivariate time-series forecasting,
sequence-to-sequence horizon prediction. Complements the sklearn classifier with a
deep-learning regressor. Research/education only.

Usage:
    python ml/forecast_lstm.py --data "sample_data/CGM Records" --epochs 15 --out ml/outputs
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IN_LEN, OUT_LEN = 12, 6   # 60 min in -> 30 min out (5-min cadence)
FEATURES = ["CGM", "TotalBolusInsulinDelivered", "CarbSize", "Basal"]


def load_all(data_dir):
    frames = []
    for sub in sorted(Path(data_dir).glob("Subject *")):
        for c in sub.glob("*.csv"):
            df = pd.read_csv(c, parse_dates=["EventDateTime"]).sort_values("EventDateTime")
            df["subject_id"] = int(sub.name.split()[-1])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_sequences(df):
    X, Y = [], []
    for _, g in df.groupby("subject_id"):
        arr = g[FEATURES].astype(float).values
        cgm = g["CGM"].astype(float).values
        for t in range(IN_LEN, len(arr) - OUT_LEN):
            X.append(arr[t - IN_LEN:t])
            Y.append(cgm[t:t + OUT_LEN])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class GlucoseLSTM(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=64, layers=2, horizon=OUT_LEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])     # last hidden state -> horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="ml/outputs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = load_all(a.data)
    X, Y = make_sequences(df)

    # Normalize features (fit on train only)
    n = len(X); idx = np.random.RandomState(0).permutation(n)
    cut = int(n * 0.8); tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
    Xn = (X - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlucoseLSTM().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.L1Loss()  # MAE in mg/dL
    dl = DataLoader(TensorDataset(torch.tensor(Xn[tr]), torch.tensor(Y[tr])),
                    batch_size=128, shuffle=True)

    for ep in range(a.epochs):
        model.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:2d}  train MAE {tot/len(tr):.2f} mg/dL")

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xn[te]).to(dev)).cpu().numpy()
    mae = np.abs(pred - Y[te]).mean()
    naive = np.abs(X[te][:, -1, 0:1] - Y[te]).mean()  # persistence baseline
    print(f"Test MAE: LSTM={mae:.2f} mg/dL  |  persistence baseline={naive:.2f} mg/dL")

    # Plot one example forecast
    k = 0
    hist = X[te][k][:, 0]
    plt.figure(figsize=(7, 4))
    plt.plot(range(-IN_LEN, 0), hist, "o-", label="history")
    plt.plot(range(0, OUT_LEN), Y[te][k], "s-", label="actual")
    plt.plot(range(0, OUT_LEN), pred[k], "x--", label="LSTM forecast")
    plt.axhspan(70, 180, color="green", alpha=0.08)
    plt.xlabel("steps (5 min)"); plt.ylabel("CGM (mg/dL)")
    plt.title("30-min glucose forecast (PyTorch LSTM)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{a.out}/forecast_example.png", dpi=130); plt.close()
    torch.save(model.state_dict(), f"{a.out}/glucose_lstm.pt")
    print(f"Artifacts -> {a.out}/forecast_example.png, glucose_lstm.pt")


if __name__ == "__main__":
    main()

"""
counterfactual_simulator.py
---------------------------
"What-if" glucose simulation. Learns a one-step-ahead CGM response model from data,
then rolls it forward autoregressively under a baseline scenario and a user-modified
counterfactual (e.g., "what if this meal's bolus were 20% larger?" or "what if carbs
were halved?"). This is the AZT1D roadmap's Phase-3 counterfactual feature.

Demonstrates: supervised response modeling + autoregressive simulation + scenario
analysis. Research/education only -- NOT clinical guidance.

Usage:
    python ml/counterfactual_simulator.py --data "sample_data/CGM Records" --subject 1 \
        --bolus-scale 1.2 --carb-scale 1.0 --out ml/outputs
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

LOOK = 6  # 30 min history


def load_subject(data_dir, sid):
    folder = Path(data_dir) / f"Subject {sid}"
    df = pd.concat([pd.read_csv(c, parse_dates=["EventDateTime"]) for c in folder.glob("*.csv")])
    return df.sort_values("EventDateTime").reset_index(drop=True)


def build_training(df):
    cgm = df["CGM"].astype(float).values
    bolus = df["TotalBolusInsulinDelivered"].astype(float).values
    carbs = df["CarbSize"].astype(float).values
    basal = df["Basal"].astype(float).values
    X, y = [], []
    for t in range(LOOK, len(cgm) - 1):
        X.append([
            cgm[t], cgm[t] - cgm[t - LOOK],                 # level, 30-min slope
            bolus[t - LOOK:t + 1].sum(), carbs[t - LOOK:t + 1].sum(), basal[t],
        ])
        y.append(cgm[t + 1] - cgm[t])                       # next-step delta
    return np.array(X), np.array(y)


def simulate(model, df, start, horizon, bolus_scale, carb_scale):
    cgm = df["CGM"].astype(float).values.copy()
    bolus = df["TotalBolusInsulinDelivered"].astype(float).values
    carbs = df["CarbSize"].astype(float).values
    basal = df["Basal"].astype(float).values
    traj = [cgm[start]]
    cur = cgm[start]
    for k in range(horizon):
        t = start + k
        recent_bolus = bolus[t - LOOK:t + 1].sum() * bolus_scale
        recent_carbs = carbs[t - LOOK:t + 1].sum() * carb_scale
        slope = traj[-1] - (traj[-LOOK] if len(traj) > LOOK else traj[0])
        feat = np.array([[cur, slope, recent_bolus, recent_carbs, basal[t]]])
        cur = float(np.clip(cur + model.predict(feat)[0], 40, 360))
        traj.append(cur)
    return np.array(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--bolus-scale", type=float, default=1.2)
    ap.add_argument("--carb-scale", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=24)  # 2 hours
    ap.add_argument("--out", default="ml/outputs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = load_subject(a.data, a.subject)
    X, y = build_training(df)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(X, y)

    # pick a window that contains a meal so the scenario is interesting
    carbs = df["CarbSize"].astype(float).values
    meal_idx = np.argmax(carbs[LOOK:len(df) - a.horizon]) + LOOK
    base = simulate(model, df, meal_idx, a.horizon, 1.0, 1.0)
    cf = simulate(model, df, meal_idx, a.horizon, a.bolus_scale, a.carb_scale)

    mins = np.arange(len(base)) * 5
    plt.figure(figsize=(7, 4))
    plt.axhspan(70, 180, color="green", alpha=0.08, label="target range")
    plt.plot(mins, base, "o-", label="baseline", ms=3)
    plt.plot(mins, cf, "s-", label=f"counterfactual (bolus×{a.bolus_scale}, carb×{a.carb_scale})", ms=3)
    plt.axhline(70, color="red", ls="--", alpha=0.5)
    plt.xlabel("minutes from meal"); plt.ylabel("CGM (mg/dL)")
    plt.title(f"Subject {a.subject}: what-if glucose response")
    plt.legend(); plt.tight_layout()
    out = f"{a.out}/counterfactual_subject{a.subject}.png"
    plt.savefig(out, dpi=130); plt.close()

    def tir(x): return float(((x >= 70) & (x <= 180)).mean() * 100)
    print(f"Baseline TIR={tir(base):.0f}%  min={base.min():.0f} | "
          f"Counterfactual TIR={tir(cf):.0f}%  min={cf.min():.0f}")
    print(f"Plot -> {out}")


if __name__ == "__main__":
    main()

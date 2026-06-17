"""
generate_synthetic_data.py
--------------------------
Create a small SYNTHETIC AZT1D-style dataset so the dashboard and ML pipeline
run end-to-end without the 875 MB DVC-tracked real data.

The output matches the real schema expected by ``src/data_loader.py``:

    sample_data/CGM Records/Subject <id>/synthetic.csv

Columns: EventDateTime, DeviceMode, BolusType, Basal, CorrectionDelivered,
         TotalBolusInsulinDelivered, FoodDelivered, CarbSize, CGM

This is fabricated data with realistic glucose dynamics (meals, boluses,
overnight drift, occasional hypo/hyper excursions). It is NOT real patient
data and must not be used for any clinical purpose.

Usage:
    python ml/generate_synthetic_data.py --subjects 3 --days 14 --out "sample_data/CGM Records"
"""
import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_subject(subject_id: int, days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = days * 24 * 12  # 5-minute cadence
    start = datetime(2023, 12, 1) + timedelta(days=subject_id)
    times = [start + timedelta(minutes=5 * i) for i in range(steps)]

    # Baseline glucose with circadian drift + AR(1) noise
    t = np.arange(steps)
    circadian = 25 * np.sin(2 * np.pi * (t % 288) / 288 - 0.5)
    cgm = np.zeros(steps)
    cgm[0] = rng.normal(140, 10)
    for i in range(1, steps):
        cgm[i] = 0.92 * cgm[i - 1] + 0.08 * (130 + circadian[i]) + rng.normal(0, 6)

    basal = np.full(steps, round(rng.uniform(0.6, 1.1), 2))
    total_bolus = np.zeros(steps)
    correction = np.zeros(steps)
    food = np.zeros(steps)
    carbs = np.zeros(steps)
    bolus_type = np.array([""] * steps, dtype=object)

    # Meals: ~3-4/day -> carb load + bolus -> glucose rises then insulin pulls down
    n_meals = int(days * rng.uniform(3, 4))
    for _ in range(n_meals):
        idx = rng.integers(12, steps - 60)
        carb = rng.uniform(30, 80)
        carbs[idx] = round(carb, 1)
        food[idx] = round(carb, 1)
        bolus = carb / rng.uniform(8, 12)  # insulin-to-carb ratio
        total_bolus[idx] = round(bolus, 2)
        bolus_type[idx] = "Meal"
        rise = np.exp(-np.arange(0, 48) / 18.0) * carb * 1.4
        cgm[idx:idx + len(rise)] += rise[: max(0, steps - idx)][: len(cgm[idx:idx + len(rise)])]
        drop_start = idx + 6
        drop = np.exp(-np.arange(0, 60) / 25.0) * bolus * 22
        end = min(steps, drop_start + len(drop))
        cgm[drop_start:end] -= drop[: end - drop_start]

    # Occasional corrections when high
    high_idx = np.where(cgm > 220)[0]
    for idx in high_idx[:: max(1, len(high_idx) // (days * 2 + 1))]:
        correction[idx] = round(rng.uniform(0.5, 2.0), 2)
        total_bolus[idx] += correction[idx]
        bolus_type[idx] = "Correction"
        end = min(steps, idx + 40)
        cgm[idx:end] -= np.exp(-np.arange(0, end - idx) / 22.0) * correction[idx] * 20

    cgm = np.clip(cgm, 40, 360)
    df = pd.DataFrame({
        "EventDateTime": times,
        "DeviceMode": "Auto",
        "BolusType": bolus_type,
        "Basal": basal,
        "CorrectionDelivered": correction.round(2),
        "TotalBolusInsulinDelivered": total_bolus.round(2),
        "FoodDelivered": food.round(1),
        "CarbSize": carbs.round(1),
        "CGM": cgm.round(0),
    })
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default="sample_data/CGM Records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for sid in range(1, args.subjects + 1):
        folder = os.path.join(args.out, f"Subject {sid}")
        os.makedirs(folder, exist_ok=True)
        df = generate_subject(sid, args.days, seed=args.seed + sid)
        path = os.path.join(folder, "synthetic.csv")
        df.to_csv(path, index=False)
        tir = ((df.CGM >= 70) & (df.CGM <= 180)).mean() * 100
        print(f"Subject {sid}: {len(df)} rows  TIR={tir:4.1f}%  -> {path}")


if __name__ == "__main__":
    main()

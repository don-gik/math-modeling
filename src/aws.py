import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    def sine_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    data = ["일시", "일강수량(mm)"]

    var_dict = {"일강수량(mm)": "precipitation"}

    df = pd.read_csv(
        "SURFACE_AWS_678_DAY_2019_2019_2025.csv",
        encoding="utf-8",
        usecols=data,
    )

    df["date"] = pd.to_datetime(df["일시"])
    df = df.set_index("date")

    numeric_cols = ["일강수량(mm)"]

    window_size = 120
    par: dict[str, dict[str, float]] = {}

    for var in numeric_cols:
        s = pd.to_numeric(df[var], errors="coerce").fillna(0)

        s = s.interpolate(method="time", limit_direction="both")

        if s.notna().sum() < window_size + 4:
            logging.error(f"{var}: not enough data after interpolation, skipping.")
            continue

        rolling = s.rolling(window=window_size, min_periods=window_size // 2).mean()

        rolling_valid = rolling.dropna()
        if len(rolling_valid) < 4:
            logging.error(f"{var}: not enough valid rolling values, skipping.")
            continue

        idx = rolling_valid.index
        x_data = np.arange(len(idx), dtype=float)
        y_data = rolling_valid.to_numpy()

        A0 = (y_data.max() - y_data.min()) / 2.0
        D0 = y_data.mean()

        B0 = 2 * np.pi / 365.0
        C0 = 0.0
        initial_guess = [A0, B0, C0, D0]

        B_min = 2.0 * np.pi / (365.0 * 10.0)  # 10년
        B_max = 2.0 * np.pi / 10.0  # 10일
        lower_bounds = [0.0, B_min, -2.0 * np.pi, y_data.min() - abs(A0) * 2]
        upper_bounds = [np.inf, B_max, 2.0 * np.pi, y_data.max() + abs(A0) * 2]

        initial_guess = [A0, B0, C0, D0]

        try:
            params, params_covariance = curve_fit(
                sine_func,
                x_data,
                y_data,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )

            par[var_dict[var]] = {
                "A": params[0],
                "B": params[1],
                "C": params[2],
                "D": params[3],
            }
        except RuntimeError:
            logging.error(f"{var}: curve_fit failed.")
            continue

        y_fit = sine_func(x_data, *params)
        x_time = idx

        r2 = r2_score(y_data, y_fit)

        par[var_dict[var]]["r2"] = r2

        plt.figure(figsize=(10, 6))
        plt.plot(x_time, y_data, label=f"{var} ({window_size}일 이동 평균)", linewidth=2)
        plt.plot(x_time, y_fit, label=f"사인 곡선 피팅 R^2={r2:.3f}", linewidth=2)

        plt.title(f"{window_size}일 이동 평균 - {var}")
        plt.xlabel("날짜")
        plt.ylabel(f"{var} 값")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    with open("aws.json", "w") as json_file:
        json.dump(par, json_file, indent=4)

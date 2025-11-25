import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs("plots/", exist_ok=True)

    def sine_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    data = [
        "일시",
        "평균기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "합계 대형증발량(mm)",
        "합계 소형증발량(mm)",
    ]

    var_dict = {
        "평균기온(°C)": "temperature",
        "일강수량(mm)": "precipitation",
        "평균 풍속(m/s)": "wind",
        "평균 상대습도(%)": "relative-humidity",
        "합계 대형증발량(mm)": "large-evaporation",
        "합계 소형증발량(mm)": "small-evaporation",
    }

    df = pd.read_csv(
        "data/SURFACE_ASOS_105_DAY_2019_2019_2025.csv",
        encoding="utf-8",
        usecols=data,
    )

    df["date"] = pd.to_datetime(df["일시"])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    numeric_cols = [
        "평균기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "합계 대형증발량(mm)",
        "합계 소형증발량(mm)",
    ]

    window_size = 120

    logger = logging.getLogger(__name__)

    for var in numeric_cols:
        par: dict[str, float] = {}

        if var_dict[var] is None:
            continue

        s = pd.to_numeric(df[var], errors="coerce").fillna(0)

        if s.notna().sum() < window_size + 4:
            logger.error(f"{var}: not enough data after interpolation, skipping.")
            continue

        rolling = s.rolling(window=window_size, min_periods=window_size // 2, center=True).mean()

        rolling_valid = rolling.dropna()
        if len(rolling_valid) < 4:
            logger.error(f"{var}: not enough valid rolling values, skipping.")
            continue

        idx = rolling_valid.index
        x_data = np.arange(len(idx), dtype=float)
        y_data = rolling_valid.to_numpy()

        A0 = (y_data.max() - y_data.min()) / 2.0
        D0 = y_data.mean()

        B0 = 2 * np.pi / 365.0
        C0 = 0.0
        initial_guess = [A0, B0, C0, D0]

        B_min = 2.0 * np.pi / (365.0 * 10.0)
        B_max = 2.0 * np.pi / 10.0
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

            par = {
                "A": params[0],
                "B": params[1],
                "C": params[2],
                "D": params[3],
            }
        except RuntimeError:
            logger.error(f"{var}: curve_fit failed.")
            continue

        y_fit = sine_func(x_data, *params)
        x_time = idx

        r2 = r2_score(y_data, y_fit)

        par["r2"] = r2

        def get_season(month):
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            else:
                return "fall"

        df["season"] = df.index.month.map(get_season)
        seasonal_avg = df.groupby("season")[var].mean().reindex(["spring", "summer", "fall", "winter"])

        season_to_md = {
            "winter": (1, 15),
            "spring": (4, 15),
            "summer": (7, 15),
            "fall": (10, 15),
        }

        season_order = ["spring", "summer", "fall", "winter"]
        years = sorted(df.index.year.unique())

        x_list = []
        y_list = []

        for year in years:
            for season in season_order:
                if pd.isna(seasonal_avg.loc[season]):
                    continue
                month, day = season_to_md[season]
                x_list.append(pd.Timestamp(year=year, month=month, day=day))
                y_list.append(seasonal_avg.loc[season])

        x_bar = pd.to_datetime(x_list)
        y_bar = np.array(y_list, dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(x_time, y_data, label=f"{var} ({window_size}일 이동 평균)", linewidth=2)
        # plt.plot(x_time, y_fit, label=f"사인 곡선 피팅 R^2={r2:.3f}", linewidth=2)

        plt.bar(x_bar, y_bar, color=["green"], width=60, alpha=0.3)

        plt.title(f"{window_size}일 이동 평균 - {var}")
        plt.xlabel("날짜")
        plt.ylabel(f"{var} 값")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/asos-{var_dict[var]}")

        with open(f"data/asos-season-{var_dict[var]}.json", "w") as json_file:
            json.dump(seasonal_avg.to_dict(), json_file, indent=4)

        with open(f"data/asos-sine-{var_dict[var]}.json", "w") as json_file:
            json.dump(par, json_file, indent=4)

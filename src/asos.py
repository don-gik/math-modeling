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
    os.makedirs("data/processed", exist_ok=True)

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
        "data/raw/SURFACE_ASOS_105_DAY_2019_2019_2025.csv",
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

        B_min = 2.0 * np.pi / (365.0 * 10.0)
        B_max = 2.0 * np.pi / 10.0
        lower_bounds = [0.0, B_min, -2.0 * np.pi]
        upper_bounds = [np.inf, B_max, 2.0 * np.pi]

        initial_guess = [A0, B0, C0]

        try:
            def sine_func(x, A, B, C):
                s = np.sin(B * x + C)
                return A * s + D0
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
                "D": D0,
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
        seasonal_by_year = df.groupby([df.index.year, "season"])[var].mean().unstack()
        monthly_avg = df.groupby(df.index.month)[var].mean()

        season_to_md = {
            "winter": (1, 15),
            "spring": (4, 15),
            "summer": (7, 15),
            "fall": (10, 15),
        }

        season_order = ["spring", "summer", "fall", "winter"]
        years = sorted(df.index.year.unique())

        monthly_payload: dict[str, float] = {str(k): float(v) for k, v in monthly_avg.items()}

        plt.figure(figsize=(10, 6))
        plt.plot(x_time, y_data, label=f"{var} ({window_size}일 이동 평균)", linewidth=2)
        # plt.plot(x_time, y_fit, label=f"사인 곡선 피팅 R^2={r2:.3f}", linewidth=2)

        x_time = rolling_valid.index
        start_year = x_time.min().year
        end_year   = x_time.max().year

        months = [int(m) for m in monthly_avg.index]
        base_height = monthly_avg.to_numpy(dtype=float)

        xs: list[pd.Timestamp] = []
        hs: list[float] = []

        for year in range(start_year, end_year + 1):
            xs.extend(pd.Timestamp(year=year, month=m, day=15) for m in months)
            hs.extend(base_height)

        x = xs
        height = np.array(hs, dtype=float)

        plt.bar(
            x,  # type: ignore
            height,
            color=["green"],
            width=20,
            alpha=0.3,
            label="월별 평균",
        )

        plt.title(f"{window_size}일 이동 평균 - {var}")
        plt.xlabel("날짜")
        plt.ylabel(f"{var} 값")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/asos-{var_dict[var]}-monthly")

        plt.figure(figsize=(10, 6))
        plt.plot(x_time, y_data, label=f"{var} ({window_size}일 이동 평균)", linewidth=2)
        plt.plot(x_time, y_fit, label=f"사인 곡선 피팅 R^2={r2:.3f}", linewidth=2)

        plt.title(f"{window_size}일 이동 평균 - {var}")
        plt.xlabel("날짜")
        plt.ylabel(f"{var} 값")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/asos-{var_dict[var]}-sine")

        with open(f"data/processed/asos-season-{var_dict[var]}.json", "w") as json_file:
            json.dump(monthly_payload, json_file, indent=4)

        par.update(monthly_payload)
        with open(f"data/processed/asos-sine-{var_dict[var]}.json", "w") as json_file:
            json.dump(par, json_file, indent=4)

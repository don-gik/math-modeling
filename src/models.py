# A(t)=A(t-1)+0.01848*(r-v)*d+0.07*o-(1-p)*0.0055*d
# A(t): 저수율 (%)
# r: 강수량 (mm)
# v: 증발량 (mm)
# d: 일수 (day)
# o: 외부급수량 (만톤)
# p: 절수정책 비율

import datetime
import json
from abc import ABC, abstractmethod

import numpy as np
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        water_usage: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
    ): ...

    @abstractmethod
    def _next(
        self,
        current_capacity: float,
        date: datetime.date,
        precipitation_scale: float = 1.0,
        water_usage_scale: float = 1.0,
        external_supply_addition: float = 0.0,
    ) -> float: ...

    @abstractmethod
    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]: ...

    def predict(
        self,
        start_capacity: float,
        start_time: datetime.date,
        end_time: datetime.date,
        mid_date: datetime.date,
        precipitation_scale: float = 1.0,
        water_usage_scale: float = 1.0,
        external_supply_addition: float = 0.0,
    ) -> list[float]:
        """Simulate with scenario adjustments applied from `mid_date` onward."""
        current_capacity: list[float] = [start_capacity]
        current_time = start_time

        effective_mid = max(mid_date, start_time)

        if getattr(self, "reference_year", None) is None:
            self.reference_year = start_time.year  # type: ignore[attr-defined]

        while current_time <= end_time:
            apply_adjustment = current_time >= effective_mid
            next_capacity = self._next(
                current_capacity=current_capacity[-1],
                date=current_time,
                precipitation_scale=precipitation_scale if apply_adjustment else 1.0,
                water_usage_scale=water_usage_scale if apply_adjustment else 1.0,
                external_supply_addition=external_supply_addition if apply_adjustment else 0.0,
            )

            current_capacity.append(next_capacity)
            current_time = current_time + datetime.timedelta(days=1)

        return current_capacity


class SeasonModel(BaseModel):
    PARAM_DIM = 8
    PARAM_NAMES = ["p", "m5", "m6", "m7", "m8", "m9", "r_trend", "v_trend"]
    P_MAX = [2.0, 1e6, 1e6, 1e6, 1e6, 1e6, 5.0, 100.0]

    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        water_usage: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
        x: list[float] | None = None,
    ):
        self.json: list[dict[str, Any]] = []
        for paths in json_paths:
            with open(paths, "r") as f:
                self.json.append(json.load(f))

        self.precipitation_index = precipitation
        self.large_evaporation_index = large_evaporation
        self.water_usage_index = water_usage
        self.external_supply = external_supply
        self.water_policy = water_policy

        default_x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        chosen_x = default_x if x is None else list(x)
        if len(chosen_x) < self.PARAM_DIM:
            chosen_x += [default_x[i] for i in range(len(chosen_x), self.PARAM_DIM)]
        self.x = chosen_x[: self.PARAM_DIM]
        self.reference_year: int = 2019

    @staticmethod
    def _get_season(date: datetime.date):
        if date.month in [12, 1, 2]:
            return "winter"
        elif date.month in [3, 4, 5]:
            return "spring"
        elif date.month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _value_with_trend(self, entry: Any, year: int) -> float:
        """Handles both plain numbers and dicts with base/trend/base_year."""
        if isinstance(entry, dict):
            base_year = entry.get("base_year", self.reference_year if self.reference_year is not None else year)
            try:
                base_year_int = int(base_year)
            except (TypeError, ValueError):
                base_year_int = year

            base_value = float(entry.get("base", entry.get("value", 0.0)))
            trend = float(entry.get("trend", 0.0))
            return base_value + trend * (year - base_year_int)

        try:
            return float(entry)
        except (TypeError, ValueError):
            return 0.0

    def _value_for_date(self, data: dict[str, Any], date: datetime.date) -> float:
        """Prefer monthly values if present, fall back to seasonal buckets."""
        # Prefer exact month as string ("1".."12") or zero-based ("0".."11")
        month_keys = [str(date.month), str(date.month - 1)]
        for key in month_keys:
            if key in data:
                return self._value_with_trend(data[key], date.year)

        # Fallback to seasonal mapping
        season = self._get_season(date)
        if season in data:
            return self._value_with_trend(data[season], date.year)

        return 0.0

    def _get(self, date: datetime.date):
        r: float = self._value_for_date(self.json[self.precipitation_index], date)
        v: float = self._value_for_date(self.json[self.large_evaporation_index], date)
        d: float = 1.0
        o: float = 0.0 if self.external_supply is None else self.external_supply
        p: float = 0.0 if self.water_policy is None else self.water_policy

        date_first = datetime.datetime(date.year, date.month, 1)
        if date.month == 12:
            date_last = datetime.datetime(date.year + 1, 1, 1)
        else:
            date_last = datetime.datetime(date.year, date.month + 1, 1)
        date_delta = date_last - date_first - datetime.timedelta(days=1)

        w: float = self.json[self.water_usage_index][str(date.month - 1)] / date_delta.days

        return (r, v, d, o, p, w)

    def _next(
        self,
        current_capacity: float,
        date: datetime.date,
        precipitation_scale: float = 1.0,
        water_usage_scale: float = 1.0,
        external_supply_addition: float = 0.0,
    ):
        r, v, d, o, p, w = self._get(date)
        p, m5, m6, m7, m8, m9, r_trend, v_trend = (
            self.x[0],
            self.x[1],
            self.x[2],
            self.x[3],
            self.x[4],
            self.x[5],
            self.x[6],
            self.x[7],
        )

        r = r * precipitation_scale
        w = w * water_usage_scale
        o = o + external_supply_addition

        if date.month == 5:
            m = m5
        elif date.month == 6:
            m = m6
        elif date.month == 7:
            m = m7
        elif date.month == 8:
            m = m8
        elif date.month == 9:
            m = m9
        else:
            m = 0.0

        m = m * water_usage_scale

        year_offset = date.year - (self.reference_year if self.reference_year is not None else date.year)
        r_adjusted = r + r_trend * year_offset
        v_adjusted = v + v_trend * year_offset

        next_capacity = current_capacity + 7.6e-1 * p * r_adjusted * d - ((current_capacity * 0.01) ** (0.6666)) * 850000 / 143290.0 * v_adjusted / 1000 * d + 0.07 * o - ((w + m) / 143290.0)
        next_capacity = min(max(next_capacity, 0.0), 100.0)

        return next_capacity

    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]:
        current_capacity: list[float] = [start_capacity]
        current_time = start_time

        if self.reference_year is None:
            self.reference_year = start_time.year

        while current_time <= end_time:
            current_capacity.append(
                self._next(current_capacity=current_capacity[len(current_capacity) - 1], date=current_time)
            )

            current_time = current_time + datetime.timedelta(days=1)

        return current_capacity


class SineModel(BaseModel):
    PARAM_DIM = 8
    PARAM_NAMES = ["p", "m5", "m6", "m7", "m8", "m9", "r_trend", "v_trend"]
    P_MAX = [2.0, 1e6, 1e6, 1e6, 1e6, 1e6, 5.0, 100.0]

    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        water_usage: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
        x: list[float] | None = None,
    ):
        self.json: list[dict[str, Any]] = []
        for paths in json_paths:
            with open(paths, "r") as f:
                self.json.append(json.load(f))

        self.precipitation_index = precipitation
        self.large_evaporation_index = large_evaporation
        self.water_usage_index = water_usage
        self.external_supply = external_supply
        self.water_policy = water_policy

        default_x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        chosen_x = default_x if x is None else list(x)
        if len(chosen_x) < self.PARAM_DIM:
            chosen_x += [default_x[i] for i in range(len(chosen_x), self.PARAM_DIM)]
        self.x = chosen_x[: self.PARAM_DIM]
        self.reference_year: int | None = None
        self.monthly_r = self._extract_monthly_component(self.json[self.precipitation_index])
        self.monthly_v = self._extract_monthly_component(self.json[self.large_evaporation_index])

    @staticmethod
    def _extract_monthly_component(raw: dict[str, Any]) -> dict[int, float]:
        monthly = {}
        for k, v in raw.items():
            try:
                month_int = int(k)
            except (TypeError, ValueError):
                continue
            monthly[month_int] = float(v)
        return monthly

    @staticmethod
    def _sine_function(x: float, A: float, B: float, C: float, D: float) -> float:
        return A * np.sin(B * x + C) + D

    def _get(self, date: datetime.date):
        r: float = self._sine_function(
            x=float((date - datetime.date(2019, 1, 1)).days),
            A=self.json[self.precipitation_index]["A"],
            B=self.json[self.precipitation_index]["B"],
            C=self.json[self.precipitation_index]["C"],
            D=self.json[self.precipitation_index]["D"],
        )
        v: float = self._sine_function(
            x=float((date - datetime.date(2019, 1, 1)).days),
            A=self.json[self.large_evaporation_index]["A"],
            B=self.json[self.large_evaporation_index]["B"],
            C=self.json[self.large_evaporation_index]["C"],
            D=self.json[self.large_evaporation_index]["D"],
        )
        month_idx = date.month
        if month_idx in self.monthly_r:
            r += self.monthly_r[month_idx]
        if month_idx in self.monthly_v:
            v += self.monthly_v[month_idx]

        year_offset = date.year - (self.reference_year if self.reference_year is not None else date.year)
        r_trend, v_trend = self.x[6], self.x[7]
        r = r + r_trend * year_offset
        v = v + v_trend * year_offset

        d: float = 1.0
        o: float = 0.0 if self.external_supply is None else self.external_supply
        p: float = 0.0 if self.water_policy is None else self.water_policy

        date_first = datetime.datetime(date.year, date.month, 1)
        if date.month == 12:
            date_last = datetime.datetime(date.year + 1, 1, 1)
        else:
            date_last = datetime.datetime(date.year, date.month + 1, 1)
        date_delta = date_last - date_first - datetime.timedelta(days=1)

        w: float = float(str(self.json[self.water_usage_index][str(date.month - 1)]).replace(",", "")) / date_delta.days

        return (r, v, d, o, p, w)

    def _next(
        self,
        current_capacity: float,
        date: datetime.date,
        precipitation_scale: float = 1.0,
        water_usage_scale: float = 1.0,
        external_supply_addition: float = 0.0,
    ):
        r, v, d, o, p, w = self._get(date)
        p, m5, m6, m7, m8, m9 = (
            self.x[0],
            self.x[1],
            self.x[2],
            self.x[3],
            self.x[4],
            self.x[5],
        )

        r = r * precipitation_scale
        w = w * water_usage_scale
        o = o + external_supply_addition

        if date.month == 5:
            m = m5
        elif date.month == 6:
            m = m6
        elif date.month == 7:
            m = m7
        elif date.month == 8:
            m = m8
        elif date.month == 9:
            m = m9
        else:
            m = 0.0

        m = m * water_usage_scale

        next_capacity = current_capacity + 7.6e-1 * p * r * d - ((current_capacity * 0.01) ** (0.6666)) * 850000 / 143290.0 * v / 1000 * d + 0.07 * o - ((w + m) / 143290.0)
        next_capacity = min(max(next_capacity, 0.0), 100.0)

        return next_capacity

    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]:
        current_capacity: list[float] = [start_capacity]
        current_time = start_time

        if self.reference_year is None:
            self.reference_year = start_time.year

        while current_time <= end_time:
            current_capacity.append(
                self._next(current_capacity=current_capacity[len(current_capacity) - 1], date=current_time)
            )

            current_time = current_time + datetime.timedelta(days=1)

        return current_capacity

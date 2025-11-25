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


class BaseModel(ABC):
    @abstractmethod
    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
    ): ...

    @abstractmethod
    def _next(self, current_capacity: float, date: datetime.date) -> float: ...

    @abstractmethod
    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]: ...


class SeasonModel(BaseModel):
    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
    ):
        self.json: list[dict[str, float]] = []
        for paths in json_paths:
            with open(paths, "r") as f:
                self.json.append(json.load(f))

        self.precipitation_index = precipitation
        self.large_evaporation_index = large_evaporation
        self.external_supply = external_supply
        self.water_policy = water_policy

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

    def _get(self, date: datetime.date):
        season = self._get_season(date)

        r: float = self.json[self.precipitation_index][season]
        v: float = self.json[self.large_evaporation_index][season]
        d: float = 1.0
        o: float = 0.0 if self.external_supply is None else self.external_supply
        p: float = 0.0 if self.water_policy is None else self.water_policy

        return (r, v, d, o, p)

    def _next(self, current_capacity: float, date: datetime.date):
        r, v, d, o, p = self._get(date)

        next_capacity = current_capacity + 0.01848 * (r - v) * d + 0.07 * o - (1 - p) * 0.0055 * d

        return next_capacity

    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]:
        current_capacity: list[float] = [start_capacity]
        current_time = start_time

        while current_time <= end_time:
            current_capacity.append(
                self._next(current_capacity=current_capacity[len(current_capacity) - 1], date=current_time)
            )

            current_time = current_time + datetime.timedelta(days=1)

        return current_capacity


class SineModel(BaseModel):
    def __init__(
        self,
        json_paths: list[str],
        precipitation: int,
        large_evaporation: int,
        external_supply: float | None = None,
        water_policy: float | None = None,
    ):
        self.json: list[dict[str, float]] = []
        for paths in json_paths:
            with open(paths, "r") as f:
                self.json.append(json.load(f))

        self.precipitation_index = precipitation
        self.large_evaporation_index = large_evaporation
        self.external_supply = external_supply
        self.water_policy = water_policy

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
        d: float = 1.0
        o: float = 0.0 if self.external_supply is None else self.external_supply
        p: float = 0.0 if self.water_policy is None else self.water_policy

        return (r, v, d, o, p)

    def _next(self, current_capacity: float, date: datetime.date):
        r, v, d, o, p = self._get(date)

        next_capacity = current_capacity + 0.01848 * (r - v) * d + 0.07 * o - (1 - p) * 0.0055 * d

        return next_capacity

    def run(self, start_capacity: float, start_time: datetime.date, end_time: datetime.date) -> list[float]:
        current_capacity: list[float] = [start_capacity]
        current_time = start_time

        while current_time <= end_time:
            current_capacity.append(
                self._next(current_capacity=current_capacity[len(current_capacity) - 1], date=current_time)
            )

            current_time = current_time + datetime.timedelta(days=1)

        return current_capacity

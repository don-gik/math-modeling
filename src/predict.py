# Simulates Model
#
# Uses Simulator class for simulating general models in models.py
# plots differences between real data and predicted data
# for analyzing model accuracy

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

import src.models as models


class Simulator:
    def __init__(self, config: DictConfig | ListConfig):
        self.jsons: ListConfig = self._safe_fetch_config("jsons", config)
        self.args: DictConfig = self._safe_fetch_config("args", config)
        self.model_config: DictConfig = self._safe_fetch_config("model", config)

    @staticmethod
    def _safe_fetch_config(attribute: str, config: DictConfig | ListConfig):
        if attribute in config:
            return getattr(config, attribute)
        raise ValueError(f"{__name__} Simulator config has no {attribute} attribute.")

    @staticmethod
    def _scenario_slug(name: str) -> str:
        slug = "".join(ch if ch.isalnum() else "_" for ch in name)
        slug = "_".join(filter(None, slug.split("_")))
        return slug if slug else "scenario"

    @staticmethod
    def _plot(
        start_date: datetime,
        end_date: datetime,
        result: list[float],
        title: str,
        model_name: str,
        plot_start_date: datetime | None = None,
        mid_date: datetime | None = None,
    ):
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        plot_start = plot_start_date or start_date
        start_offset = max(0, (plot_start - start_date).days)
        x_data = pd.date_range(start=plot_start, end=end_date + timedelta(days=1))
        end_offset = start_offset + len(x_data)
        y_data = np.array(result[start_offset:end_offset])

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="저수량 예측", linewidth=2)

        if mid_date is not None:
            plt.axvline(
                pd.Timestamp(mid_date),  # type: ignore
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="mid_date",
            )

        plt.title(f"계절 저수량 예측 - {model_name}")
        plt.xlabel("날짜")
        plt.ylabel("저수량")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots/prediction/", exist_ok=True)

        plt.savefig(f"plots/prediction/{title}")
        plt.close()

    @staticmethod
    def _plot_both(
        start_date: datetime,
        end_date: datetime,
        result: list[float],
        title: str,
        model_name: str,
        data_path: str,
        data_name: str,
        data_loc: str,
        plot_start_date: datetime | None = None,
        mid_date: datetime | None = None,
    ):
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        plot_start = plot_start_date or start_date
        start_offset = max(0, (plot_start - start_date).days)
        x_data = pd.date_range(start=plot_start, end=end_date + timedelta(days=1))
        end_offset = start_offset + len(x_data)
        y_data = np.array(result[start_offset:end_offset])

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="저수량 예측", linewidth=2)

        df = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
        )
        df["계측일자"] = pd.to_datetime(df["계측일자"], format="%Y%m%d")
        df = df.set_index("계측일자")

        s = df["저수율"]

        start = pd.Timestamp(plot_start)
        end = pd.Timestamp(end_date)
        last_idx = s.index.max()
        slice_end = min(end, last_idx)

        s = s.loc[start:slice_end]

        if mid_date is not None:
            actual_end = min(pd.Timestamp(mid_date), slice_end)
            s_actual = s.loc[start:actual_end]
        else:
            s_actual = s

        s_actual.plot(label="실제 계절 저수량", linewidth=2)

        if mid_date is not None:
            plt.axvline(
                pd.Timestamp(mid_date),  # type: ignore
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="mid_date",
            )

        plt.title(f"계절 저수량 예측 - {model_name}")
        plt.xlabel("날짜")
        plt.ylabel("저수량")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots/prediction/", exist_ok=True)

        plt.savefig(f"plots/prediction/{title}_both")
        plt.close()

    @staticmethod
    def load_observed_series(
        data_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.Series:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        df["계측일자"] = pd.to_datetime(df["계측일자"], format="%Y%m%d")
        df = df.set_index("계측일자")

        s = df["저수율"]
        s = s.loc[start_date.strftime("%Y-%m-%d") : end_date.strftime("%Y-%m-%d")]
        return s

    def _load_model_parameters(self, model_type: str, param_names: list[str]) -> list[float]:
        proportion_path = getattr(self.args, "proportion_path", f"data/processed/{model_type}_proportion.json")
        if not os.path.exists(proportion_path):
            raise FileNotFoundError(f"Expected proportion file at {proportion_path}")

        with open(proportion_path, "r", encoding="utf-8") as f:
            proportion_json: dict[str, Any] = json.load(f)

        param_source: dict[str, Any] = proportion_json.get("proportion", proportion_json)
        return [float(param_source.get(name, 0.0)) for name in param_names]

    def _normalize_scenarios(self) -> list[dict[str, Any]]:
        scenarios_raw = getattr(self.args, "scenarios", [])
        if isinstance(scenarios_raw, (DictConfig, ListConfig)):
            scenarios_raw = OmegaConf.to_container(scenarios_raw, resolve=True)

        scenarios_list: list[dict[str, Any]] = [] if scenarios_raw is None else list(scenarios_raw)  # type: ignore

        if not scenarios_list:
            return [
                {
                    "name": "baseline",
                    "precipitation_scale": 1.0,
                    "water_usage_scale": 1.0,
                    "external_supply_addition": 0.0,
                }
            ]

        normalized = []
        for scenario in scenarios_list:
            name = str(scenario.get("name", "scenario"))
            normalized.append(
                {
                    "name": name,
                    "precipitation_scale": float(scenario.get("precipitation_scale", 1.0)),
                    "water_usage_scale": float(scenario.get("water_usage_scale", 1.0)),
                    "external_supply_addition": float(scenario.get("external_supply_addition", 0.0)),
                }
            )

        return normalized

    def run(self):
        logger = logging.getLogger(__name__)

        logger.info(f"Setting up model, with type of {getattr(self.model_config, 'type')}")

        model_type: str = getattr(self.model_config, "type")
        start_date = datetime.strptime(getattr(self.args, "start_date"), "%Y-%m-%d")
        end_date = datetime.strptime(getattr(self.args, "end_date"), "%Y-%m-%d")
        mid_date_raw = getattr(self.args, "mid_date", getattr(self.args, "start_date"))
        mid_date = datetime.strptime(mid_date_raw, "%Y-%m-%d")
        if mid_date < start_date:
            mid_date = start_date

        data_path = getattr(self.args, "data")
        data_name = getattr(self.args, "name")
        data_loc = getattr(self.args, "loc")

        model_class = getattr(models, model_type)
        param_names = getattr(model_class, "PARAM_NAMES", None)
        if param_names is None:
            param_dim = getattr(model_class, "PARAM_DIM", 0)
            param_names = [f"x{i}" for i in range(param_dim)]

        parameters = self._load_model_parameters(model_type, param_names)

        model: models.BaseModel = model_class(
            json_paths=OmegaConf.to_container(self.jsons, resolve=True),
            precipitation=getattr(self.args, "precipitation"),
            large_evaporation=getattr(self.args, "large_evaporation"),
            water_usage=getattr(self.args, "water_usage"),
            x=parameters,
        )

        logger.info(f"Using parameter set from processed file: {parameters}")

        scenarios = self._normalize_scenarios()
        logger.info("Running %d scenario(s)", len(scenarios))

        for scenario in scenarios:
            logger.info(
                "Scenario '%s': precipitation x%.2f, usage x%.2f, external +%.2f mm/day",
                scenario["name"],
                scenario["precipitation_scale"],
                scenario["water_usage_scale"],
                scenario["external_supply_addition"],
            )

            result = model.predict(
                start_capacity=getattr(self.args, "start_capacity"),
                start_time=start_date.date(),
                end_time=end_date.date(),
                mid_date=mid_date.date(),
                precipitation_scale=scenario["precipitation_scale"],
                water_usage_scale=scenario["water_usage_scale"],
                external_supply_addition=scenario["external_supply_addition"],
            )

            title = (
                f"{getattr(self.model_config, 'type')}_"
                f"{self._scenario_slug(scenario['name'])}_"
                f"{start_date.date()}_to_{end_date.date()}"
            )

            plot_start = start_date

            self._plot(
                start_date=start_date,
                end_date=end_date,
                result=result,
                title=title,
                model_name=getattr(self.model_config, "type"),
                plot_start_date=plot_start,
                mid_date=mid_date,
            )
            self._plot_both(
                start_date=start_date,
                end_date=end_date,
                result=result,
                title=title,
                model_name=getattr(self.model_config, "type"),
                data_path=data_path,
                data_name=data_name,
                data_loc=data_loc,
                plot_start_date=plot_start,
                mid_date=mid_date,
            )

        logger.info("Ignore incompatible converter warnings. It is a pandas plotting warning.")

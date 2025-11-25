# Simulates Model

# Uses Simulator class for simulating general models in models.py
# plots differences between real data and predicted data
# for analyzing model accuracy


import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas
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
        else:
            raise ValueError(f"{__name__} Simulator config has no {attribute} attribute.")

    @staticmethod
    def _plot(start_date: datetime, end_date: datetime, result: list[float], title: str, model_name: str):
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        x_data = pandas.date_range(
            start=start_date,
            end=end_date + timedelta(days=1),
        )
        y_data = np.array(result)

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="저수량 예측", linewidth=2)

        plt.title(f"계절 저수량 예측 - {model_name}")
        plt.xlabel("날짜")
        plt.ylabel("저수량")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots/prediction/", exist_ok=True)

        plt.savefig(f"plots/prediction/{title}")

    def run(self):
        logger = logging.getLogger(__name__)

        logger.info(f"Setting up model, with type of {getattr(self.model_config, 'type')}")

        model_type: str = getattr(self.model_config, "type")
        model: models.BaseModel = getattr(models, model_type)(
            json_paths=OmegaConf.to_container(self.jsons),
            precipitation=getattr(self.args, "precipitation"),
            large_evaporation=getattr(self.args, "large_evaporation"),
        )

        start_date = datetime.strptime(getattr(self.args, "start_date"), "%Y-%m-%d")
        end_date = datetime.strptime(getattr(self.args, "end_date"), "%Y-%m-%d")

        logger.info(f"Calculating results from {str(start_date.date())} to {str(end_date.date())}")

        result = model.run(
            start_capacity=getattr(self.args, "start_capacity"), start_time=start_date.date(), end_time=end_date.date()
        )

        title = str(getattr(self.model_config, "type")) + "_" + str(start_date.date()) + "_to_" + str(end_date.date())

        logger.info("Plotting...")

        self._plot(start_date=start_date, end_date=end_date, result=result, title=title, model_name=getattr(self.model_config, "type"))

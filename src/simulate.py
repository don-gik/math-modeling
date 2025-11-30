# Simulates Model

# Uses Simulator class for simulating general models in models.py
# plots differences between real data and predicted data
# for analyzing model accuracy

import cma
import logging
import os
import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import minimize, least_squares, differential_evolution, dual_annealing, minimize_scalar
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

    @staticmethod
    def _plot_both(start_date: datetime, end_date: datetime, result: list[float], title: str, model_name: str, data_path: str, data_name: str, data_loc: str):
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        x_data = pandas.date_range(
            start=start_date,
            end=end_date + timedelta(days=1),
        )
        y_data = np.array(result)

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label="저수량 예측", linewidth=2)

        df = pandas.read_csv(
            data_path,
            encoding="utf-8-sig",
        )
        df["계측일자"] = pandas.to_datetime(df["계측일자"], format="%Y%m%d")
        df = df.set_index("계측일자")

        s = df["저수율"]
        s = s.loc[start_date.strftime("%Y-%m-%d") : end_date.strftime("%Y-%m-%d")]

        s.plot(label="실제 계절 저수량", linewidth=2)

        plt.title(f"계절 저수량 예측 - {model_name}")
        plt.xlabel("날짜")
        plt.ylabel("저수량")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots/prediction/", exist_ok=True)

        plt.savefig(f"plots/prediction/{title}_both")

    @staticmethod
    def load_observed_series(
        data_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pandas.Series:
        df = pandas.read_csv(data_path, encoding="utf-8-sig")
        df["계측일자"] = pandas.to_datetime(df["계측일자"], format="%Y%m%d")
        df = df.set_index("계측일자")

        s = df["저수율"]
        s = s.loc[start_date.strftime("%Y-%m-%d") : end_date.strftime("%Y-%m-%d")]
        return s
    
    @staticmethod
    def loss_for_P(x, model_type, jsons, args, y_true, start_date, end_date):
        y_true_arr = np.asarray(y_true, dtype=float)

        def simulate_with_proportion(
            x: list[float],
            model_type: str,
            jsons,
            args,
            start_date: datetime,
            end_date: datetime,
        ) -> np.ndarray:
            model: models.BaseModel = getattr(models, model_type)(
                json_paths=OmegaConf.to_container(jsons),
                precipitation=getattr(args, "precipitation"),
                large_evaporation=getattr(args, "large_evaporation"),
                water_usage=getattr(args, "water_usage"),
                x=x,
            )
            result: list[float] = model.run(
                start_capacity=getattr(args, "start_capacity"), start_time=start_date.date(), end_time=end_date.date()
            )
            return np.array(result, dtype=float)

        y_pred = simulate_with_proportion(x, model_type, jsons, args, start_date, end_date)
        if not np.all(np.isfinite(y_pred)):
            n = len(y_true_arr)
            return np.full(n, 1e6, dtype=float)
        n = min(len(y_true_arr), len(y_pred))

        scale = np.std(y_true_arr[:n]) + 1e-8

        q_low = np.quantile(y_true_arr[:n], 0.15)
        q_high = np.quantile(y_true_arr[:n], 0.85)
        extreme_mask = (y_true_arr[:n] <= q_low) | (y_true_arr[:n] >= q_high)
        weights = 1.0 + 2.0 * extreme_mask.astype(float)

        residual = ((y_true_arr[:n] - y_pred[:n]) / scale) * weights

        return residual

    def run(self):
        logger = logging.getLogger(__name__)

        logger.info(f"Setting up model, with type of {getattr(self.model_config, 'type')}")

        model_type: str = getattr(self.model_config, "type")
        start_date = datetime.strptime(getattr(self.args, "start_date"), "%Y-%m-%d")
        end_date = datetime.strptime(getattr(self.args, "end_date"), "%Y-%m-%d")

        data_path = getattr(self.args, "data")
        data_name = getattr(self.args, "name")
        data_loc = getattr(self.args, "loc")

        model_class = getattr(models, model_type)
        param_names = getattr(model_class, "PARAM_NAMES", None)
        param_dim = getattr(model_class, "PARAM_DIM", len(param_names) if param_names is not None else 6)
        if param_dim <= 0:
            raise ValueError(f"{model_type} must define a positive PARAM_DIM (got {param_dim}).")

        P_MAX_fallback = [2.0, 1e6, 1e6, 1e6, 1e6, 1e6]
        P_MAX_raw = getattr(model_class, "P_MAX", P_MAX_fallback if param_dim <= len(P_MAX_fallback) else [1.0] * param_dim)
        P_MAX = np.array(P_MAX_raw, dtype=float)

        if P_MAX.shape[0] != param_dim:
            raise ValueError(f"P_MAX length ({len(P_MAX)}) does not match PARAM_DIM ({param_dim}) for {model_type}.")

        if param_names is None:
            param_names = [f"x{i}" for i in range(param_dim)]

        y_true = self.load_observed_series(data_path, start_date, end_date)

        common_args = (model_type, self.jsons, self.args, y_true, start_date, end_date)

        def residual_scaled(u, model_type, jsons, args, y_true, start_date, end_date):
            x = u * P_MAX
            return self.loss_for_P(x, model_type, jsons, args, y_true, start_date, end_date)

        def scalar_objective(theta, model_type, jsons, args, y_true, start_date, end_date):
            res = residual_scaled(theta, model_type, jsons, args, y_true, start_date, end_date)
            return float(np.mean(res**2))
        
        def scalar_objective_cma(theta):
            return scalar_objective(
                np.array(theta),
                model_type, self.jsons, self.args, y_true, start_date, end_date,
            )
        
        x0 = [0.5] * param_dim
        sigma0 = 0.2
        lb_cma = [0.0] * param_dim
        ub_cma = [1.0] * param_dim
        cma_opts = {
            "bounds": [lb_cma, ub_cma],
            "CMA_stds": [sigma0] * param_dim,
            "scaling_of_variables": [1.0] * param_dim,
            "typical_x": [0.0] * param_dim,
            "fixed_variables": None,
            "transformation": None,
            "maxiter": 2000,
        }

        if param_dim == 1:
            def objective_scalar(u):
                u_clipped = float(np.clip(u, 0.0, 1.0))
                return scalar_objective(
                    np.array([u_clipped]),
                    model_type, self.jsons, self.args, y_true, start_date, end_date,
                )

            res_scalar = minimize_scalar(
                objective_scalar,
                bounds=(0.0, 1.0),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 2000},
            )
            theta_best = np.array([float(np.clip(res_scalar.x, 0.0, 1.0))])
        else:
            es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
            es.optimize(scalar_objective_cma)
            theta_best = np.array(es.result.xbest)

        x_opt = P_MAX * theta_best

        model: models.BaseModel = getattr(models, model_type)(
            json_paths=OmegaConf.to_container(self.jsons),
            precipitation=getattr(self.args, "precipitation"),
            large_evaporation=getattr(self.args, "large_evaporation"),
            water_usage=getattr(self.args, "water_usage"),
            x=x_opt
        )

        result_json = {name: float(val) for name, val in zip(param_names, x_opt)}
        with open(f"data/processed/{model_type}_proportion.json", "w") as f:
            o = {"proportion" : result_json}
            json.dump(o, f, indent=4)

        logger.info(f"Proportion value : {x_opt}")

        logger.info(f"Calculating results from {str(start_date.date())} to {str(end_date.date())}")

        result = model.run(
            start_capacity=getattr(self.args, "start_capacity"), start_time=start_date.date(), end_time=end_date.date()
        )

        title = str(getattr(self.model_config, "type")) + "_" + str(start_date.date()) + "_to_" + str(end_date.date())

        logger.info("Plotting...")

        self._plot(start_date=start_date, end_date=end_date, result=result, title=title, model_name=getattr(self.model_config, "type"))
        self._plot_both(start_date=start_date, end_date=end_date, result=result, title=title, model_name=getattr(self.model_config, "type"), data_path=data_path, data_name=data_name, data_loc=data_loc)

        logger.info("Ignore incompatible converter warnings. It is a pandas plotting warning.")

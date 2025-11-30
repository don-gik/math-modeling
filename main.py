from __future__ import annotations

import logging
import logging.config

import hydra

from src.asos import main as asos_main
from src.aws import main as aws_main
from src.water import main as water_main
from src.simulate import Simulator as SimulateSimulator
from src.predict import Simulator as PredictSimulator


@hydra.main(config_path="config", version_base=None)
def main(config):
    logging.config.fileConfig("logging.conf", encoding="utf-8")
    logger = logging.getLogger(__name__)

    logger.info("Starting Program...")

    simulate_enabled = bool(getattr(config.args, "simulate", False))
    predict_enabled = bool(getattr(config.args, "predict", False))

    water_args = None
    if simulate_enabled and hasattr(config, "simulate"):
        water_args = config.simulate.args
    elif predict_enabled and hasattr(config, "predict"):
        water_args = config.predict.args

    if water_args is not None:
        logger.info("Precalculating water usage into json...")
        water_main(water_args)
        logger.info("Water usage saved into json file.")

    if config.args.data_process == "aws":
        aws_main()
    if config.args.data_process == "asos":
        asos_main()
    if simulate_enabled and hasattr(config, "simulate"):
        simulator = SimulateSimulator(config.simulate)
        simulator.run()
    if predict_enabled and hasattr(config, "predict"):
        predictor = PredictSimulator(config.predict)
        predictor.run()

    logger.info(f"Finishing data process : {config.args.data_process}")
    logger.info("Program ended.")


if __name__ == "__main__":
    main()

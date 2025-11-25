from __future__ import annotations

import logging
import logging.config

import hydra

from src.asos import main as asos_main
from src.aws import main as aws_main
from src.simulate import Simulator


@hydra.main(config_path="config", version_base=None)
def main(config):
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger(__name__)

    logger.info("Starting Program...")
    logger.info(f"Using option : {config}")

    if config.args.data_process == "aws":
        aws_main()
    if config.args.data_process == "asos":
        asos_main()
    if config.args.simulate:
        simulator = Simulator(config.simulate)
        simulator.run()

    logger.info(f"Finishing data process : {config.args.data_process}")
    logger.info("Program ended.")


if __name__ == "__main__":
    main()

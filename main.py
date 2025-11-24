from __future__ import annotations

import numpy as np
import json

import logging
import logging.config

import hydra


@hydra.main(config_path="config", version_base=None)
def main(config):
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger(__name__)

    logger.info("Starting Program...")
    logger.info(f"Using option : {config}")

    logger.info("Program ended.")


if __name__ == "__main__":
    main()
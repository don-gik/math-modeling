import pandas as pd
import os

from omegaconf import DictConfig, ListConfig

def main(config: DictConfig | ListConfig):
    df = pd.read_csv(getattr(config, "water_use"))
    df = df["물 사용량(m^3)"].str.replace(',', '').astype(float)
    os.makedirs("data/processed", exist_ok=True)
    df.to_json("data/processed/water_use.json", indent=4)
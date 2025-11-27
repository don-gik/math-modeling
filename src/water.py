import pandas as pd

from omegaconf import DictConfig, ListConfig

def main(config: DictConfig | ListConfig):
    df = pd.read_csv(getattr(config, "water_use"))
    df = df["Water Usage(m^3)"].str.replace(',', '').astype(float)
    df.to_json("data/processed/water_use.json", indent=4)
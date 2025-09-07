import pandas as pd

def load_twitter(path="./data/twitter-clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

import pandas as pd

def load_data(path="data/iris.csv") -> pd.DataFrame:
    """Wczytuje dataset Iris z pliku CSV"""
    df = pd.read_csv(path)
    return df


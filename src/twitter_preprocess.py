import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_twitter_data(path="./twitter-clean.csv"):
    df = pd.read_csv(path)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 60% train, 20% valid, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

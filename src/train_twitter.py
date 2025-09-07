import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parametry z zadania
model_params = {
    'classifier': LogisticRegression(),
    'classifier__C': 1,
    'classifier__max_iter': 1000,
    'vectorizer__max_df': 0.8,
    'vectorizer__max_features': 7000,
    'vectorizer__min_df': 5,
    'vectorizer__ngram_range': (1, 2)
}

def make_pipeline_nlp() -> Pipeline:
    """Tworzy pipeline: TF-IDF + klasyfikator"""
    return Pipeline(steps=[
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

def train_twitter():
    logger.info("Loading Twitter data...")

    # 👇 wczytanie CSV
    df = pd.read_csv("./twitter-clean.csv")
    X = df["text"]
    y = df["sentiment"]

    # sprawdzamy unikalne klasy
    classes = y.unique()
    logger.info(f"Unique classes in dataset: {classes}")

    # jeżeli tylko jedna klasa → dodajemy sztuczną drugą
    stratify_arg = None
    if len(classes) < 2:
        logger.warning("Dataset has only one class! Dodaję sztuczne próbki i wyłączam stratify.")
        row = df.iloc[0].copy()
        row["sentiment"] = 1 if row["sentiment"] == 0 else 0
        df = pd.concat([df, pd.DataFrame([row, row, row, row])], ignore_index=True)  # dodajemy kilka kopii
        X = df["text"]
        y = df["sentiment"]
        stratify_arg = None  # 🚨 bez stratify
    else:
        stratify_arg = y

    # podział na train/valid/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=stratify_arg
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=y_temp if stratify_arg is not None else None
    )

    logger.info("Building pipeline...")
    pipeline = make_pipeline_nlp()
    pipeline.set_params(**model_params)

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    y_preds = pipeline.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_preds)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_preds)}")

    # zapis modelu
    joblib.dump(pipeline, "model.pkl")
    logger.info("Model saved to model.pkl")

if __name__ == "__main__":
    train_twitter()

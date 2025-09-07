from src.data_loader import load_data
from src.pipeline import make_pipeline_nlp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parametry z zadania
model_params =  {
    'classifier': LogisticRegression(),
    'classifier__C': 1,
    'classifier__max_iter': 1000,
    'vectorizer__max_df': 0.8,
    'vectorizer__max_features': 7000,
    'vectorizer__min_df': 5,
    'vectorizer__ngram_range': (1, 2)
}

def train_twitter():
    logger.info("Loading Twitter data...")
    twitter_params = {
        "path": "./twitter-clean.csv",   # 👈 poprawiona ścieżka
        "data_columns": 'text',
        "target_column": 'sentiment',
        "sample_size": 1.0,   # pełne dane
    }

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(**twitter_params)

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

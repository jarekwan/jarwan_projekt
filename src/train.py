from src.data_loader import load_data
from src.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import logging
logging.basicConfig(level=logging.WARN,filename="output.log",filemode="a")
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting training model...")

    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    logger.debug(f"Dataset loaded: {len(X)} items.")

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = make_pipeline(model='randomforest')
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)

    logger.info(f"Accuracy: {accuracy_score(y_test, y_preds)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_preds)}")


if __name__ == "__main__":
    train_model()
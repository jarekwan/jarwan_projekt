from src.data_loader import load_data
from src.pipeline import make_pipeline
feature-grid_search
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, filename="output.log", filemode="a")
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting training model with GridSearch...")

    # 1. Dane
    df = load_data()
    X = df.drop(columns="species")
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Pipeline (LogisticRegression domyślnie)
    pipeline = make_pipeline(model="logreg")

    # 3. Siatka parametrów
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    # 4. GridSearch
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    y_preds = grid.predict(X_test)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_preds)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_preds)}")

if __name__ == "__main__":
    train_model()


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
main

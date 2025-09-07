from src.data_loader import load_data
from src.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("Starting training model...")

    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    logger.debug(f"Dataset loaded: {len(X)} items.")

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = make_pipeline(model='tree')

    param_grid = get_grid_param()

    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    y_preds = grid.predict(X_test)

    logger.info(f"Accuracy: {accuracy_score(y_test, y_preds)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_preds)}")


def get_grid_param():
    param_grid = [{
            "classifier": [SVC()],
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
        },
        {
            "classifier": [DecisionTreeClassifier()],
            "classifier__max_depth": [2, 4, 6],
            "classifier__min_samples_split": [2, 4, 6],
            "classifier__min_samples_leaf": [2, 4, 6],
        },
        {
            "classifier": [RandomForestClassifier()],
            "classifier__max_depth": [2, 4, 6],
            "classifier__min_samples_split": [2, 4, 6],
            "classifier__min_samples_leaf": [2, 4, 6],
        },
        {
            "classifier": [LogisticRegression()],
            "classifier__C": [0.1, 1, 10, 100],
        }
    ]
    return param_grid


if __name__ == "__main__":
    train_model()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def make_pipeline(model='logistic') -> Pipeline:

    if model == 'tree':
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Creating pipeline with model {model}.")
    models = {
        'logistic': LogisticRegression(max_iter=100),
        'tree': DecisionTreeClassifier(),
        'randomforest': RandomForestClassifier(),
        'svc': SVC(kernel='linear'),
    }

    if model not in models:
        logger.error(f"Requested model {model} not supported.")
        raise ValueError(f'Invalid model: {model}')

    clf = models[model]

    logger.info(f"Training model with {clf.__class__.__name__} classifier")
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

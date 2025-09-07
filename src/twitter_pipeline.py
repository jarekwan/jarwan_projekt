from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_twitter_pipeline(model="logistic") -> Pipeline:
    """
    Pipeline for Twitter text data (TF-IDF + classifier).
    """
    classifiers = {
        "logistic": LogisticRegression(max_iter=200),
    }

    if model not in classifiers:
        logger.error(f"Model {model} not supported!")
        raise ValueError(f"Invalid model: {model}")

    clf = classifiers[model]

    logger.info(f"Creating Twitter pipeline with {clf.__class__.__name__}")
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("classifier", clf)
    ])

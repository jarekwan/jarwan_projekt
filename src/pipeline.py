from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def make_pipeline(use_tree=False) -> Pipeline:
    """Tworzy pipeline: standaryzacja + klasyfikator"""
    if use_tree:
        classifier = DecisionTreeClassifier(random_state=42)
    else:
        classifier = LogisticRegression(max_iter=100)

    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

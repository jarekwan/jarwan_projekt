from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def make_pipeline() -> Pipeline:
    """Tworzy pipeline: standaryzacja + klasyfikacja"""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=100))
    ])
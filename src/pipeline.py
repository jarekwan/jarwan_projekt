from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def make_pipeline(model="logreg") -> Pipeline:
    """Tworzy pipeline: standaryzacja + klasyfikator"""
    if model == "tree":
        classifier = DecisionTreeClassifier(random_state=42)
    elif model == "forest":
        classifier = RandomForestClassifier(random_state=42)
    else:
        classifier = LogisticRegression(max_iter=100)

    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

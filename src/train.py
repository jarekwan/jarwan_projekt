from src.data_loader import load_data
from src.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
    print("Training model...")

    # 1. Wczytanie danych
    df = load_data()
    X = df.drop(columns="species")   # kolumny wejściowe
    y = df["species"]                # etykiety (target)

    # 2. Podział na trening/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Pipeline
    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Predykcja i ocena
    y_preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_preds))

if __name__ == "__main__":
    train_model()

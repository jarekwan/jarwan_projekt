import joblib
import logging

# konfiguracja logów
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# wczytaj wytrenowany model z pliku
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def predict_text(text: str):
    """
    Funkcja przyjmuje string i zwraca predykcję modelu.
    """
    prediction = model.predict([text])[0]

    # zakładam, że Twoja kolumna 'sentiment' to 1/0 (albo :-)/:-()
    if prediction == 1:
        return ":-)"
    else:
        return ":-("

if __name__ == "__main__":
    example = "This movie was fantastic!"
    result = predict_text(example)
    logger.info(f"Prediction for '{example}': {result}")

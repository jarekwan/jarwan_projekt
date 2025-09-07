import joblib
import logging
from fastapi import FastAPI, Query
app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading model...")
model = joblib.load('models/twitter-model.pkl')
logger.info("Model loaded.")

mapping = {
        1: "\U0001F600 \U0001F929 ٩(◕‿◕｡)۶  party time!",
        0: "\U0001F62D \U0001F494 (╯︵╰,) so sad..."
    }

def predict_text(text:str) -> str:
    prediction = model.predict([text])[0] # zwraca 0 lub 1
    return mapping[prediction]


@app.get("/predict")
def get_prediction(text: str = Query(..., description="Evaluates sentiment of the provided text")):
    sentiment = predict_text(text)
    return {"text": text, "sentiment": sentiment}


if __name__ == '__main__':

    X = ['let us go party tonight this is wonderful','sad story lost work', 'well i did not pass the exam shame']

    for text in X:
        logger.info(f"{text} -> {predict_text(text)}")
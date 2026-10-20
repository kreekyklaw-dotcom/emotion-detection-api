import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="Telugu-English Emotion Detection API",
    description="Predict emotions in Telugu-English code-mixed text.",
    version="1.0.0",
)

DATA_PATH = Path(__file__).parent / "model_data.json"

EMOTIONS = [
    "Anger",
    "Disgust",
    "Fear",
    "Joy",
    "Love_Affection",
    "Sadness",
    "Surprise",
]

# Load the saved model parameters.
with open(DATA_PATH, "r", encoding="utf-8") as file:
    model_data = json.load(file)


# --------------------------------------------------
# RECONSTRUCT TF-IDF
# --------------------------------------------------

tfidf_data = model_data["tfidf"]

vocabulary = {
    word: index
    for index, word in enumerate(tfidf_data["vocabulary"])
}

vectorizer = CountVectorizer(
    vocabulary=vocabulary,
    ngram_range=(
        tfidf_data["ngram_min"],
        tfidf_data["ngram_max"],
    ),
    lowercase=tfidf_data["lowercase"],
    token_pattern=tfidf_data["token_pattern"],
)

tfidf_transformer = TfidfTransformer(
    norm=tfidf_data["norm"],
    use_idf=True,
    smooth_idf=False,
    sublinear_tf=tfidf_data["sublinear_tf"],
)

# Restore the saved IDF weights.
tfidf_transformer.idf_ = np.asarray(
    tfidf_data["idf"],
    dtype=np.float64,
)


def transform_text(text: str):
    """Convert input text into the saved TF-IDF feature space."""
    word_counts = vectorizer.transform([text])
    return tfidf_transformer.transform(word_counts)


# --------------------------------------------------
# LOAD CLASSIFIER PARAMETERS
# --------------------------------------------------

def load_classifier(name: str):
    """Load a classifier's saved coefficients and intercepts."""
    if name not in ("logistic_regression", "svm"):
        raise ValueError(f"Unsupported model: {name}")

    data = model_data[name]

    classes = data["classes"]
    coefficients = np.asarray(data["coef"], dtype=np.float64)
    intercepts = np.asarray(data["intercept"], dtype=np.float64)

    if coefficients.ndim != 2:
        raise ValueError(f"Invalid coefficient dimensions for {name}")

    if coefficients.shape[0] != len(classes):
        raise ValueError(f"Class count mismatch for {name}")

    if coefficients.shape[1] != len(vocabulary):
        raise ValueError(f"Feature count mismatch for {name}")

    if intercepts.shape != (len(classes),):
        raise ValueError(f"Intercept count mismatch for {name}")

    return {
        "classes": classes,
        "coef": coefficients,
        "intercept": intercepts,
    }


classifiers = {
    "logistic_regression": load_classifier("logistic_regression"),
    "svm": load_classifier("svm"),
}


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Telugu-English text to classify",
    )

    model: str = Field(
        default="logistic_regression",
        description="Choose logistic_regression or svm",
    )


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Telugu-English Emotion Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(classifiers.keys()),
        "emotion_classes": EMOTIONS,
    }


@app.get("/models")
def get_models():
    return {
        "available_models": [
            {
                "name": "logistic_regression",
                "description": "Logistic Regression classifier",
            },
            {
                "name": "svm",
                "description": "Support Vector Machine classifier",
            },
        ],
        "default_model": "logistic_regression",
        "emotions": EMOTIONS,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()
    model_name = request.model.strip().lower()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    if model_name not in classifiers:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. Choose "
                "'logistic_regression' or 'svm'."
            ),
        )

    # Convert input text into TF-IDF features.
    features = transform_text(text)

    # Calculate the classifier's decision scores.
    classifier = classifiers[model_name]

    scores = np.asarray(
        features @ classifier["coef"].T
    ).ravel() + classifier["intercept"]

    # The highest decision score determines the predicted class.
    predicted_index = int(np.argmax(scores))
    predicted_emotion = classifier["classes"][predicted_index]

    return {
        "text": text,
        "model": model_name,
        "predicted_emotion": predicted_emotion,
    }
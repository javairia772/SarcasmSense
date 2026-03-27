from transformers import pipeline
import os

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        repo = os.getenv("HF_REPO_ID", "Backened/sarcasm-model")
        print(f"Loading model from {repo} ...")
        _classifier = pipeline(
            "text-classification",
            model=repo,
            device=-1,       # CPU — change to 0 for GPU
        )
        print("Model ready.")
    return _classifier

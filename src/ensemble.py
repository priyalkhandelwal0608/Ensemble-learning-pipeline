import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from src.train_models import train_base_models
from src.preprocess import load_and_preprocess

def build_and_save_ensemble():
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess("data/data.csv")

    # Train base models
    log_reg, rf, gb = train_base_models(X_train, y_train)

    # Build stacked features
    stacked_train = np.column_stack((
        rf.predict_proba(X_train)[:,1],
        gb.predict_proba(X_train)[:,1]
    ))

    meta = LogisticRegression()
    meta.fit(stacked_train, y_train)

    # ✅ Save ensemble
    joblib.dump((rf, gb, meta), "models/ensemble.pkl")
    print("✅ Ensemble model saved at models/ensemble.pkl")

if __name__ == "__main__":
    build_and_save_ensemble()
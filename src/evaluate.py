from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

def evaluate_models(log_reg, rf, gb, meta, X_train, X_test, y_train, y_test):
    # Base predictions
    log_pred = log_reg.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)

    # Ensemble predictions
    stacked_test = np.column_stack((
        rf.predict_proba(X_test)[:,1],
        gb.predict_proba(X_test)[:,1]
    ))
    meta_pred = meta.predict(stacked_test)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
    print("Ensemble Accuracy:", accuracy_score(y_test, meta_pred))

    print("\nClassification Report (Ensemble):\n", classification_report(y_test, meta_pred))
    print("Ensemble ROC-AUC:", roc_auc_score(y_test, meta_pred))
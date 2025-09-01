import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, core_acts):
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Accuracy :", accuracy_score(y_true, y_pred_classes))
    print("Precision:", precision_score(y_true, y_pred_classes, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred_classes, average="macro"))
    print("Macro F1 :", f1_score(y_true, y_pred_classes, average="macro"))

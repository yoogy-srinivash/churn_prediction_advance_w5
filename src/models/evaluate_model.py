from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, X_test, y_test):

    probs = model.predict_proba(X_test)[:,1]

    preds = (probs > 0.35).astype(int)

    print("Confusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print("\nROC-AUC:", auc)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # Precision Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    # Find best threshold by F1
    f1_scores = 2*(precision*recall)/(precision+recall+1e-6)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    print("\nBest Threshold based on F1:", best_threshold)
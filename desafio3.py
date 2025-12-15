import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix
from xgboost import XGBClassifier

X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.99, 0.01],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# MODELO 1: sem ajuste de desbalanceamento
model_default = XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

model_default.fit(X_train, y_train)

y_pred_default = model_default.predict(X_test)

recall_default = recall_score(y_test, y_pred_default, pos_label=1)

print("MODELO SEM AJUSTE")
print("Recall (Fraude):", recall_default)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_default))


# MODELO 2: com scale_pos_weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

model_weighted = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

model_weighted.fit(X_train, y_train)

y_prob = model_weighted.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_weighted = (y_prob >= threshold).astype(int)

recall_weighted = recall_score(y_test, y_pred_weighted, pos_label=1)

print("\nMODELO COM scale_pos_weight")
print("Recall (Fraude):", recall_weighted)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_weighted))

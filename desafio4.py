from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MODELO 1: Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)

# MODELO 2: XGBoost
xgb_model = XGBClassifier(
    objective="binary:logistic",
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)

print("Comparativo (Acurácia)")
print("--------------------------------")
print(f"Decision Tree (max_depth=3): {acc_dt:.4f}")
print(f"XGBoost       (max_depth=3): {acc_xgb:.4f}")

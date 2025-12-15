import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.1,
    "eval_metric": "logloss",
    "seed": 42
}

evallist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=10,
    verbose_eval=True
)

print(f"\nNúmero de árvores usadas: {bst.best_iteration + 1}")

bst.save_model("meu_modelo.json")
print("Modelo salvo com sucesso!")

bst_loaded = xgb.Booster()
bst_loaded.load_model("meu_modelo.json")

y_pred_prob = bst_loaded.predict(dtest)
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo carregado: {acc:.4f}")

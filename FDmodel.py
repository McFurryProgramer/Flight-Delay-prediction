import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =======================
# Загрузка данных
# =======================
df = pd.read_csv("airlines_delay.csv")
print(df.head())

# =======================
# Первичный анализ данных
# =======================
df["Class"].value_counts().plot(kind="bar")
plt.xlabel("Класс")
plt.ylabel("Количество рейсов")
plt.title("Распределение задержанных и незадержанных рейсов")
plt.show()

delay_by_day = df.groupby("DayOfWeek")["Class"].mean()
delay_by_day.plot(marker="o")
plt.xlabel("День недели")
plt.ylabel("Доля задержанных рейсов")
plt.title("Зависимость задержек рейсов от дня недели")
plt.show()

plt.hist(df[df["Class"] == 0]["Length"], bins=30, alpha=0.6, label="Без задержки")
plt.hist(df[df["Class"] == 1]["Length"], bins=30, alpha=0.6, label="С задержкой")
plt.xlabel("Длительность рейса")
plt.ylabel("Количество рейсов")
plt.legend()
plt.title("Распределение длительности рейсов")
plt.show()

df.info()
print(df.isnull().sum())

# =======================
# Подготовка данных
# =======================
X = df.drop(columns=["Class"])
y = df["Class"]

categorical_features = ["Airline", "AirportFrom", "AirportTo"]
numerical_features = ["Flight", "Time", "Length", "DayOfWeek"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =======================
# Логистическая регрессия
# =======================
logreg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

logreg_model.fit(X_train, y_train)

y_pred_lr = logreg_model.predict(X_test)
y_proba_lr = logreg_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-score:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# Матрица ошибок — логистическая регрессия
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Без задержки", "С задержкой"]
)
disp_lr.plot(cmap="Blues")
plt.title("Матрица ошибок: Logistic Regression")
plt.show()

# =======================
# Случайный лес
# =======================
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# Матрица ошибок — случайный лес
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Без задержки", "С задержкой"]
)
disp_rf.plot(cmap="Greens")
plt.title("Матрица ошибок: Random Forest")
plt.show()

# =======================
# Сравнительная таблица
# =======================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf)
    ],
    "F1-score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_rf)
    ]
})

print(results)

# =======================
# ROC-кривые
# =======================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые моделей")
plt.legend()
plt.show()

# =======================
# Feature importance (Random Forest)
# =======================
ohe = rf_model.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([cat_feature_names, numerical_features])

importances = rf_model.named_steps["classifier"].feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Топ-20 признаков
plt.figure(figsize=(8, 6))
plt.barh(
    feature_importance_df["Feature"].head(20)[::-1],
    feature_importance_df["Importance"].head(20)[::-1]
)
plt.xlabel("Важность признака")
plt.title("Топ-20 наиболее значимых признаков (Random Forest)")
plt.tight_layout()
plt.show()

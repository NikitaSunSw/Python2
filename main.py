# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("processed_titanic.csv")

data = data.drop(["PassengerId", "Name", "Cabin"], axis=1)

# целевая переменная
y = data["Transported"].astype(int)

# признаки
X = data.drop("Transported", axis=1)

# тренировочкая и тестовая выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# логистическая
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# линейная
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)

# порог классификации
threshold = 0.5
y_pred_linreg_binary = (y_pred_linreg >= threshold).astype(int)

# метрики логистической
print("=== Логистическая Регрессия ===")
print(f"F1 Score: {f1_score(y_test, y_pred_logreg):.2f}")
print("\nМатрица ошибок:")
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
print(conf_matrix_logreg)

# метрики линейной
print("\n=== Линейная Регрессия ===")
print(f"F1 Score: {f1_score(y_test, y_pred_linreg_binary):.2f}")
print("\nМатрица ошибок:")
conf_matrix_linreg = confusion_matrix(y_test, y_pred_linreg_binary)
print(conf_matrix_linreg)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix_logreg, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Логистическая Регрессия")
axes[0].set_xlabel("Предсказанные")
axes[0].set_ylabel("Истинные")

sns.heatmap(conf_matrix_linreg, annot=True, fmt="d", cmap="Reds", ax=axes[1])
axes[1].set_title("Линейная Регрессия (Threshold=0.5)")
axes[1].set_xlabel("Предсказанные")
axes[1].set_ylabel("Истинные")

plt.tight_layout()
plt.show()git
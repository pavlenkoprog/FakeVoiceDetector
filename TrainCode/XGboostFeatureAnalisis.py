import pickle
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Загружаем датасет
csv_file = "../ProcessedDatasets/audio_features.csv"
df = pd.read_csv(csv_file)

# Отделяем признаки и метки классов
X = df[['mean_amp', 'var_amp', 'median_amp', 'low_amp_ratio']].values
y = df['class'].values

# Разделяем данные на обучающую и тестовую выборки (80% - обучение, 20% - тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели XGBoost
model = xgb.XGBClassifier(scale_pos_weight=len(y_train) / (2 * sum(y_train)), random_state=42)
model.fit(X_train, y_train)

model_name = "xgboost_model.pkl"
with open(f"../Models/{model_name}", "wb") as f:
    pickle.dump(model, f)
print(f"Модель сохранена {model_name}")

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Вероятности для позитивного класса (если задача бинарная)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:\n", cm)

# ROC-кривая и AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Визуализация ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

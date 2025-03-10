import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Загружаем датасет
csv_file = "../ProcessedDatasets/audio_features.csv"
df = pd.read_csv(csv_file)

# Отделяем признаки и метки классов
X = df[['mean_amp', 'var_amp', 'median_amp', 'low_amp_ratio']].values
y = df['class'].values

# Разделяем данные на обучающую и тестовую выборки (80% - обучение, 20% - тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели Random Forest
model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Используем сбалансированные веса классов для учета дисбаланса
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Вероятности для позитивного класса (если задача бинарная)

model_name = "random_forest_model.pkl"
with open(f"../Models/{model_name}", "wb") as f:
    pickle.dump(model, f)
print(f"Модель сохранена {model_name}")

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
plt.plot

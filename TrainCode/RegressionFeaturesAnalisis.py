import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загружаем датасет
csv_file = "../ProcessedDatasets/audio_features.csv"
df = pd.read_csv(csv_file)

# Отделяем признаки и метки классов
X = df[['mean_amp', 'var_amp', 'median_amp', 'low_amp_ratio']].values
y = df['class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# проверка обучения модели без понижения размерности
model = LogisticRegression(class_weight='balanced')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("Accuracy без PCA:", model.score(X_test, y_test))

# Применяем PCA для понижения размерности до 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Разделяем данные на тренировочную и тестовую выборки (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Обучаем логистическую регрессию
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность классификации: {accuracy * 100:.2f}%")

model_name = "logistic_regression_model.pkl"
with open(f"../Models/{model_name}", "wb") as f:
    pickle.dump(model, f)
print(f"Модель сохранена {model_name}")

# Сохраняем PCA в файл
with open("../Models/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print("PCA-модель сохранена в pca_model.pkl")
# Сохраняем scaler в файл
with open("../Models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("scaler тоже сохранили")

# Определяем диапазоны осей для построения сетки
x_min, x_max = X_pca[:, 0].min() - 0.05, X_pca[:, 0].max() + 0.05
y_min, y_max = X_pca[:, 1].min() - 0.05, X_pca[:, 1].max() + 0.05

# Создание сетки для отображения границы
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Прогнозируем классы для каждой точки сетки
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация
plt.figure(figsize=(8, 6))

# Отображаем линии раздела
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)

# Отображаем точки данных
colors = ['red' if label == 0 else 'green' for label in y]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, edgecolors='k')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D проекция аудиофичей после PCA с границей разделения классов")
plt.grid()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загружаем датасет
csv_file = "../ProcessedDatasets/audio_features.csv"
df = pd.read_csv(csv_file)
print("Заголовки колонко фичей", df.head())

# Отделяем признаки и метки классов
X = df[['mean_amp', 'var_amp', 'median_amp', 'low_amp_ratio']].values
y = df['class'].values

# Применяем PCA для понижения размерности до 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Визуализация
plt.figure(figsize=(8, 6))
colors = ['red' if label == 0 else 'green' for label in y]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, edgecolors='k')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D проекция аудиофичей после PCA")
plt.grid()
plt.show()

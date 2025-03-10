import os
import pickle
import soundfile as sf
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.colors import ListedColormap

# Параметры
SEGMENT_DURATION = 10  # Длительность фрагмента в секундах
DOWNSAMPLED_SIZE = 1000
train_csv_file = "ProcessedDatasets/audio_features.csv"


class AudioClassifierPipeline:
    def __init__(self):
        self.xgb_model = pickle.load(open("Models/xgboost_model.pkl", 'rb'))
        self.rf_model = pickle.load(open("Models/random_forest_model.pkl", 'rb'))
        self.lr_model = pickle.load(open("Models/logistic_regression_model.pkl", 'rb'))
        self.pca_model = pickle.load(open("Models/pca_model.pkl", 'rb'))
        self.scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
        self.nn_model = keras.models.load_model("Models/neuro_waveform_model.h5")
        print("Все модели загружены.")


    def extract_features(self, segment):
        amplitude = np.abs(segment)
        normalized_amplitude = amplitude / np.max(amplitude)

        mean_amp = np.mean(normalized_amplitude)
        var_amp = np.var(normalized_amplitude)
        median_amp = np.median(normalized_amplitude)
        threshold = 0.1
        low_amp_ratio = np.sum(normalized_amplitude < threshold) / len(normalized_amplitude)

        return [mean_amp, var_amp, median_amp, low_amp_ratio], normalized_amplitude

    def process_audio(self, data, samplerate):
        segment_samples = SEGMENT_DURATION * samplerate
        num_segments = max(1, len(data) // segment_samples)

        # Дополнение файла до нужной длины если он меньше SEGMENT_DURATION
        while len(data) < segment_samples:
            data = np.tile(data, 2)[:segment_samples]

        results = []
        total_segments = 0
        xgb_real_count = 0
        rf_real_count = 0
        lr_real_count = 0
        nn_real_count = 0
        new_data_pca = []

        for i in range(num_segments):
            segment = data[i * segment_samples: (i + 1) * segment_samples]
            features, normalized_amplitude = self.extract_features(segment)

            # Уменьшаем количество точек до DOWNSAMPLED_SIZE
            downsampled_segment = np.interp(
                np.linspace(0, len(normalized_amplitude) - 1, DOWNSAMPLED_SIZE),
                np.arange(len(normalized_amplitude)),
                normalized_amplitude
            )

            # Преобразуем в массив
            features_array = np.array(features).reshape(1, -1)
            normalized_array = downsampled_segment.reshape(1, -1)

            X_scaled = self.scaler.transform(features_array)
            X_pca = self.pca_model.transform(X_scaled)
            new_data_pca.append(X_pca)

            # Применяем модели
            xgb_pred = self.xgb_model.predict(features_array)[0]
            rf_pred = self.rf_model.predict(features_array)[0]
            lr_pred = self.lr_model.predict(X_pca)[0]
            nn_pred = self.nn_model.predict(normalized_array)[0][0]  # Для нейросети нужен float
            nn_pred = nn_pred + 0.3
            nn_pred = np.clip(nn_pred, 0, 1)

            # Считаем количество предсказаний "Real" для каждого метода
            print("xgb_pred",xgb_pred, "rf_pred", rf_pred, "lr_pred", lr_pred, "nn_class", nn_pred)
            xgb_real_count += xgb_pred
            rf_real_count += rf_pred
            lr_real_count += lr_pred
            nn_real_count += nn_pred
            total_segments += 1

        # Рассчитываем процент предсказаний "Real" для каждой модели
        xgb_real_percent = (xgb_real_count / total_segments) * 100
        rf_real_percent = (rf_real_count / total_segments) * 100
        lr_real_percent = (lr_real_count / total_segments) * 100
        nn_real_percent = (nn_real_count / total_segments) * 100

        # Вывод итоговых результатов
        print("\nИТОГОВЫЕ ПРОЦЕНТЫ 'Real':")
        print(f"  XGBoost: {xgb_real_percent:.2f}%")
        print(f"  RandomForest: {rf_real_percent:.2f}%")
        print(f"  Logistic Regression: {lr_real_percent:.2f}%")
        print(f"  Neural Network: {nn_real_percent:.2f}%")

        img_buf = plot_classification_results(self.lr_model, self.scaler, self.pca_model, train_csv_file, new_data_pca)

        results = xgb_real_percent, rf_real_percent, lr_real_percent, nn_real_percent
        return results, img_buf



# Отрисовка графика
def plot_classification_results(model, scaler, pca, csv_file, new_data_pca):
    # Загружаем данные из CSV
    df = pd.read_csv(csv_file)
    X = df[['mean_amp', 'var_amp', 'median_amp', 'low_amp_ratio']].values
    y = df['class'].values

    # Применяем сохранённые StandardScaler и PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # new_data_pca = np.vstack(new_data_pca)
    new_data_pca = np.array(new_data_pca).squeeze()
    if new_data_pca.ndim == 1:
        new_data_pca = new_data_pca.reshape(1, -1)

    # Определяем диапазоны осей для построения сетки
    x_min, x_max = X_pca[:, 0].min() - 0.05, X_pca[:, 0].max() + 0.05
    y_min, y_max = X_pca[:, 1].min() - 0.05, X_pca[:, 1].max() + 0.05

    # Создание сетки для отображения границы
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Прогнозируем классы для каждой точки сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Построение графика
    plt.figure(figsize=(8, 6))

    # Отображаем линии раздела
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)

    # Разделяем точки по классам
    class_0 = X_pca[np.array(y) == 0]  # Красные точки
    class_1 = X_pca[np.array(y) == 1]  # Зеленые точки

    # Рисуем точки для каждого класса отдельно
    plt.scatter(class_0[:, 0], class_0[:, 1], c='red', alpha=0.7, edgecolors='k', label="Класс обучения 0 (фейк)")
    plt.scatter(class_1[:, 0], class_1[:, 1], c='green', alpha=0.7, edgecolors='k', label="Класс обучения 1 (реальный)")

    # print(new_data_pca)
    # print(X_pca)
    # Новые точки
    plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='yellow', edgecolors='black', label="Сегменты вашего аудио")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Классификация данных с линией регрессии")
    plt.legend()
    plt.grid()
    # plt.show()

    # Сохранение в буфер памяти
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()

    img_buf.seek(0)
    return img_buf


if __name__ == "__main__":

    pipeline = AudioClassifierPipeline()
    #file_path = "SoudExample/rector_sample_fake_30sec 48ГГц.wav"
    file_path = "SoudExample/rector_sample_real_30sec_44.1ГГц.wav"
    print(f"Загружаем файл: {file_path}")

    data, samplerate = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    results = pipeline.process_audio(data, samplerate)
    print("\nРезультаты обработки:", results)

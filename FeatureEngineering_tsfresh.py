import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Загрузка данных
    df = pd.read_csv("ProcessedDatasets/audio_waveforms.csv")

    # Преобразуем строки в numpy массивы
    X = np.array([np.fromstring(w.strip("[]"), sep=" ") for w in df["waveform"]])
    y = df["class"].values  # Метки классов

    # Нужно подготовить данные в "long-form"
    df = pd.DataFrame(X)
    df = df.stack().reset_index()
    df.columns = ['id', 'time', 'value']

    features = extract_features(df, column_id='id', column_sort='time')

    # print("features.head()", features.head())
    # print("features.columns", features.columns)
    # print("feature_calculators.__file__", feature_calculators.__file__)

    # Сохраняю фичи
    features.to_csv('ProcessedDatasets/tsfresh_features.csv', index=False)

    # Если данные имеют пропуски, их нужно обработать
    knn_imputer = KNNImputer(n_neighbors=5)
    features_imputed = knn_imputer.fit_transform(features)

    # Понижаем размерность с помощью PCA до 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_imputed)
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

    # Выбираем цвета для каждого класса
    unique_classes = np.unique(y)
    colors = plt.cm.get_cmap('tab10', len(unique_classes))

    # Строим график
    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(unique_classes):
        indices = y == cls
        plt.scatter(pca_df['PCA1'][indices], pca_df['PCA2'][indices],
                    color=colors(i), label=f'Class {cls}', alpha=0.6)
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('2D Visualization after PCA')

    plt.show()

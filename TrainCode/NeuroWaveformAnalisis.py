import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers

# Загрузка данных
df = pd.read_csv("../ProcessedDatasets/audio_waveforms.csv")

# Преобразуем строки в numpy массивы
X = np.array([np.fromstring(w.strip("[]"), sep=" ") for w in df["waveform"]])
y = df["class"].values  # Метки классов

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),  # Dropout с вероятностью 0.3 (30% нейронов отключены)
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),  # Dropout для второго слоя
    layers.Dense(1, activation="sigmoid")
])


# Компиляция
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Обучение
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nТочность на тестовых данных: {test_acc:.4f}")

model_name = "neuro_waveform_model.h5"
model.save(f'../Models/{model_name}')
print(f"Модель сохранена {model_name}")

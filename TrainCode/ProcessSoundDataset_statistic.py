import os
import soundfile as sf
import numpy as np
import pandas as pd

# Параметры
DATASET_PATH = "../SoundDataset"
CLASSES = ["Real", "Fake"]
SEGMENT_DURATION = 5  # Длительность фрагмента в секундах
OUTPUT_CSV = "../ProcessedDatasets/audio_features.csv"


def extract_features(file_path, samplerate, segment):
    """Вычисляет статистические признаки аудиофрагмента."""
    amplitude = np.abs(segment)
    normalized_amplitude = amplitude / np.max(amplitude)
    mean_amp = np.mean(normalized_amplitude)
    var_amp = np.var(normalized_amplitude)
    median_amp = np.median(normalized_amplitude)
    threshold = 0.1
    low_amp_ratio = np.sum(normalized_amplitude < threshold) / len(normalized_amplitude)
    return [mean_amp, var_amp, median_amp, low_amp_ratio]


def process_audio_files():
    """Загружает аудиофайлы, нарезает их на 5-секундные фрагменты и сохраняет статистики в CSV."""
    dataset = []

    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        class_label = 1 if class_name == "Real" else 0

        for file_name in os.listdir(class_path):
            print(file_name)
            file_path = os.path.join(class_path, file_name)

            try:
                data, samplerate = sf.read(file_path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)  # Преобразуем стерео в моно

                segment_samples = SEGMENT_DURATION * samplerate
                num_segments = len(data) // segment_samples

                for i in range(num_segments):
                    segment = data[i * segment_samples: (i + 1) * segment_samples]
                    features = extract_features(file_path, samplerate, segment)
                    features.append(class_label)
                    dataset.append(features)
            except Exception as e:
                print(f"Ошибка при обработке {file_name}: {e}")

    # Сохраняем датасет в CSV
    columns = ["mean_amp", "var_amp", "median_amp", "low_amp_ratio", "class"]
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Датасет сохранён в {OUTPUT_CSV}")


if __name__ == "__main__":
    process_audio_files()

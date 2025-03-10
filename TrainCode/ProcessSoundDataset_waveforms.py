import os
import soundfile as sf
import numpy as np
import pandas as pd

# Параметры
DATASET_PATH = "../SoundDataset"
CLASSES = ["Real", "Fake"]
SEGMENT_DURATION = 10  # Длительность фрагмента в секундах
DOWNSAMPLED_SIZE = 1000
OUTPUT_CSV = "../ProcessedDatasets/audio_waveforms.csv"


def process_audio_files():
    """Загружает аудиофайлы, нарезает их на 5-секундные фрагменты и сохраняет нормализованные амплитуды в CSV."""
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
                    amplitude = np.abs(segment)
                    normalized_amplitude = amplitude / np.max(amplitude)
                    print(len(normalized_amplitude))

                    # Уменьшаем количество точек до DOWNSAMPLED_SIZE
                    downsampled_segment = np.interp(
                        np.linspace(0, len(normalized_amplitude) - 1, DOWNSAMPLED_SIZE),
                        np.arange(len(normalized_amplitude)),
                        normalized_amplitude
                    )

                    print(len(downsampled_segment))

                    dataset.append([downsampled_segment, class_label])
            except Exception as e:
                print(f"Ошибка при обработке {file_name}: {e}")

    # Сохраняем датасет в CSV
    df = pd.DataFrame(dataset, columns=["waveform", "class"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Датасет сохранён в {OUTPUT_CSV}")


if __name__ == "__main__":
    process_audio_files()

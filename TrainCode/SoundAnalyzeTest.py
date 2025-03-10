import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# file_path = "SoudExample/rector_sample_fake_30sec 48ГГц.wav"
# file_path = "SoudExample/rector_sample_fake_30sec_41.1Ггц.wav"
# file_path = "SoudExample/rector_sample_real_30sec_44.1ГГц.wav"
file_path = "../SoudExample/rector_sample_real_3-48ГГц.wav"

try:
    data, samplerate = sf.read(file_path)
    print(f"Частота дискретизации: {samplerate} Гц")

    # Обрезаем первые 5 секунд
    num_samples = int(5 * samplerate)
    data = data[:num_samples]

    # Воспроизведение звука
    sd.play(data, samplerate)
    sd.wait()

    # Вычисляем громкость (амплитуду) и нормализуем
    amplitude = np.abs(data)
    normalized_amplitude = amplitude / np.max(amplitude)
    time = np.linspace(0, len(data) / samplerate, num=len(data))
    print(f"Структура записи: {normalized_amplitude.shape} ")

    # Преобразуем стерео в моно, если файл многоканальный
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    amplitude = np.abs(data)
    normalized_amplitude = amplitude / np.max(amplitude)

    # Вычисляем статистические параметры
    mean_amp = np.mean(normalized_amplitude)
    var_amp = np.var(normalized_amplitude)
    median_amp = np.median(normalized_amplitude)
    threshold = 0.1  # Порог для определения "низкой громкости"
    low_amp_ratio = np.sum(normalized_amplitude < threshold) / len(normalized_amplitude)

    print(f"Средняя амплитуда: {mean_amp:.4f}")
    print(f"Дисперсия амплитуды: {var_amp:.4f}")
    print(f"Медиана амплитуды: {median_amp:.4f}")
    print(f"Доля времени ниже порога {threshold}: {low_amp_ratio:.4f}")

    # Строим график громкости
    plt.figure(figsize=(10, 4))
    plt.plot(time, normalized_amplitude, label="Нормализованная громкость", color='b')
    plt.xlabel("Время (сек)")
    plt.ylabel("Нормализованная амплитуда")
    plt.title("График громкости аудиофайла")
    plt.legend()
    plt.grid()
    plt.show()

    print(len(normalized_amplitude))

except Exception as e:
    print(f"Ошибка при обработке файла: {e}")
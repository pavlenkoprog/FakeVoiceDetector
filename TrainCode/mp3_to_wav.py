import os
from pydub import AudioSegment

# Папки для исходных и сохранённых файлов
input_folder = 'mp3_to_wav'
output_folder = 'converted_wav_files'

AudioSegment.ffmpeg = r"E:\SOFT_E\ffmpeg-2025-03-06-git-696ea1c223-full_build\bin\ffmpeg.exe"

# Создаем папку для wav файлов, если её нет
os.makedirs(output_folder, exist_ok=True)

# Получаем все mp3 файлы в папке
for filename in os.listdir(input_folder):
    if filename.endswith('.mp3'):
        # Полный путь к исходному mp3 файлу
        mp3_path = os.path.join(input_folder, filename)

        # Загружаем mp3 файл
        audio = AudioSegment.from_mp3(mp3_path)

        # Создаем новый путь для сохранения wav файла
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(output_folder, wav_filename)

        # Сохраняем в формате wav
        audio.export(wav_path, format='wav')

        print(f'Конвертирован файл: {filename}')

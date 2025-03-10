import logging
import subprocess

import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.utils import executor
import soundfile as sf
import os
from config import API_KEY

from AudioClassifierPipeline import AudioClassifierPipeline

API_TOKEN = API_KEY

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
pipeline = AudioClassifierPipeline()

# Настройка логирования в файл
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     filename="E:/python_science/FakeSoundDetection/bot_logs.log",  # Путь к файлу
#     filemode="a"
# )
# logging.info("Бот запущен и готов к работе!")


# Приветственное сообщение
@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    logging.info(f"Received message from {message.from_user.username}: {message.text}")
    await message.reply("Привет! Пожалуйста, отправьте мне голосовое сообщение или файл в формате .wav для проверки.")


async def result_message(message, results, img_buf):
    # Отправляем ответ пользователю
    response_text = (
        f"Модели машинного обучения говорят, что это реальный голос на:\n"
        f"🔹 XGBoost: {results[0]:.2f}%\n"
        f"🔹 RandomForest: {results[1]:.2f}%\n"
        f"🔹 Логистическая регрессия: {results[2]:.2f}%\n"
        f"🔹 Простая нейронная сеть: {results[3]:.2f}%\n"
        f"Вот как модели видат сегменты вашей аудио записи"

    )
    await message.reply(response_text)

    photo = InputFile(img_buf, filename="plot.png")
    await bot.send_photo(chat_id=message.chat.id, photo=photo)


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice(message: types.Message):
    logging.info(f"Received voice message from {message.from_user.username}")

    file_id = message.voice.file_id
    new_file = await bot.get_file(file_id)

    # Пути к файлам
    ogg_path = f"Downloads/voice_{message.message_id}.ogg"
    wav_path = f"Downloads/voice_{message.message_id}.wav"

    # Сохраняем голосовое сообщение
    await new_file.download(destination_file=ogg_path)
    logging.info(f"Voice message saved as {ogg_path}")

    # Конвертируем .ogg в .wav
    try:
        subprocess.run(["ffmpeg", "-i", ogg_path, "-ar", "44100", "-ac", "1", wav_path], check=True)
        logging.info(f"Converted {ogg_path} to {wav_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed: {e}")
        await message.reply("Ошибка при конвертации голосового сообщения. Попробуйте еще раз.")
        os.remove(ogg_path)
        return

    # Читаем WAV файл
    data, samplerate = sf.read(wav_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Обрабатываем файл
    results, img_buf = pipeline.process_audio(data, samplerate)

    await result_message(message, results, img_buf)

    # Удаляем файлы после обработки
    # TODO включить удаление загруженных аудио чтобы он не занимали место
    # os.remove(ogg_path)
    # os.remove(wav_path)



@dp.message_handler(content_types=[types.ContentType.DOCUMENT, types.ContentType.AUDIO])
async def handle_audio(message: types.Message):
    logging.info(f"Received audio message from {message.from_user.username}")

    # Обработка DOCUMENT
    if message.content_type == types.ContentType.DOCUMENT:
        if message.document.mime_type == 'audio/vnd.wav' or message.document.file_name.endswith('.wav'):
            # Проверяем размер файла
            file_size = message.document.file_size
            max_file_size = 20 * 1024 * 1024  # 20 MB
            if file_size > max_file_size:
                await message.reply("Файл слишком большой. Пожалуйста, отправьте файл размером до 20МБ.")
                return

            # Загрузка файла
            file_id = message.document.file_id
            new_file = await bot.get_file(file_id)
            # Дополнительные действия с файлом

    # Обработка AUDIO
    elif message.content_type == types.ContentType.AUDIO:
        if message.audio.mime_type == 'audio/wav' or message.audio.file_name.endswith('.wav'):
            # Проверяем размер файла
            file_size = message.audio.file_size
            max_file_size = 20 * 1024 * 1024  # 20 MB
            if file_size > max_file_size:
                await message.reply("Файл слишком большой. Пожалуйста, отправьте файл размером до 20МБ.")
                return

            # Загрузка файла
            file_id = message.audio.file_id
            new_file = await bot.get_file(file_id)
            # Дополнительные действия с файлом

    else:
        await message.reply("Пожалуйста, отправьте аудиофайл в формате .wav менее 20 MB")

    file_path = f"Downloads/audio_{message.message_id}.wav"
    await new_file.download(destination_file=file_path)
    logging.info(f".wav file saved as {file_path}")

    # Читаем WAV файл
    data, samplerate = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Обрабатываем файл
    results, img_buf = pipeline.process_audio(data, samplerate)

    await result_message(message, results, img_buf)

    # Удаляем файл после обработки
    # TODO включить удаление загруженных аудио чтобы он не занимали место
    # os.remove(file_path)



# Обработка других типов сообщений (например, текста)
@dp.message_handler()
async def handle_other(message: types.Message):
    logging.info(f"Received other message from {message.from_user.username}: {message.text}")
    await message.reply("Я ожидаю голосовое сообщение или файл в формате .wav менее 20 MB. Пожалуйста, отправьте что-то из этого.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

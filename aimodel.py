import telebot
import os
import config
from telebot import types

bot = telebot.TeleBot(config.TOKEN)

def save_photo(message):
    try:
        # Проверяем, что пользователь действительно отправил фото
        if message.content_type != 'photo':
            bot.send_message(message.chat.id, "Пожалуйста, отправьте фото!")
            return

        # Получаем фото с наилучшим качеством (последний элемент в массиве photo)
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Создаем папку userFiles, если ее нет
        os.makedirs('userFiles', exist_ok=True)

        # Формируем путь для сохранения
        file_extension = file_info.file_path.split('.')[-1]
        filename = f"userFiles/photo_{message.from_user.id}_{message.message_id}.{file_extension}"

        # Сохраняем файл
        with open(filename, 'wb') as new_file:
            new_file.write(downloaded_file)









        bot.send_message(message.chat.id, "Фото успешно сохранено!")










    except Exception as e:
        bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}")
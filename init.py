import telebot
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from telebot import types
import config
import getRandomModel
import glob

# Инициализация бота
bot = telebot.TeleBot(config.TOKEN)

# Словарь с марками и моделями автомобилей
CAR_MODELS = {
    'BYD': ['BYDDenzaZ9GT', 'BYDSeal05DM-i', 'BYDSeal07DM-i'],
    'Changan': ['ChanganShenlanL07'],
    'Chery': ['CheryTiggo7ProMax', 'CheryTiggo9C-DM'],
    'Jetour': ['JetourShanhaiL7']
}

# Получаем список всех возможных классов (марка + модель)
ALL_CLASSES = []
for brand, models in CAR_MODELS.items():
    for model in models:
        ALL_CLASSES.append(f"{brand}_{model}")

print(ALL_CLASSES)
# Проверяем, существует ли файл с весами модели
MODEL_PATH = 'car_model.pth'
MODEL_EXISTS = os.path.exists(MODEL_PATH)


# Подготовка модели
def load_model():
    try:
        # Загружаем предобученную ResNet18
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(ALL_CLASSES))

        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

                # Проверяем, совпадает ли число классов в модели и текущей конфигурации
                if 'fc.weight' in checkpoint and checkpoint['fc.weight'].shape[0] != len(ALL_CLASSES):
                    print(
                        f"⚠️ Количество классов в модели ({checkpoint['fc.weight'].shape[0]}) не совпадает с текущим ({len(ALL_CLASSES)}).")
                    print("❌ Модель не загружена. Нужно переобучить или проверить конфигурацию.")
                    return model  # Возвращаем модель с начальными весами

                # Если всё совпадает — загружаем веса
                model.load_state_dict(checkpoint)
                print("✅ Модель успешно загружена")
            except Exception as e:
                print(f"❌ Ошибка при загрузке модели: {str(e)}")
                print("⚠️ Используется модель с начальными весами")
        else:
            print("⚠️ Файл с весами не найден. Используется модель с начальными весами.")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Критическая ошибка при загрузке модели: {str(e)}")
        exit(1)


# Трансформации для изображения (должны совпадать с валидационными в первом коде)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_car_model(image_path, model):
    try:
        image = Image.open(image_path).convert('RGB')  # Убедимся, что изображение в RGB
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        predicted_class = ALL_CLASSES[preds.item()]
        brand, model_name = predicted_class.split('_', 1)
        return brand, model_name
    except Exception as e:
        print(f"Ошибка при предсказании: {str(e)}")
        return None, None


# Загрузка модели
model = load_model()
print(f"Модель инициализирована для {len(ALL_CLASSES)} классов")


def save_photo(message):
    try:
        if message.content_type != 'photo':
            bot.send_message(message.chat.id, "Пожалуйста, отправьте фото!")
            return

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        os.makedirs('temp_photos', exist_ok=True)
        temp_path = f"temp_photos/{message.message_id}.jpg"

        with open(temp_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        car_brand, car_model = predict_car_model(temp_path, model)

        if car_brand is None or car_model is None:
            bot.send_message(message.chat.id, "Не удалось определить модель автомобиля. Попробуйте другое фото.")
            os.remove(temp_path)
            return

        markup = types.InlineKeyboardMarkup(row_width=3)
        item1 = types.InlineKeyboardButton("Да! Это та машина.", callback_data='good')
        item2 = types.InlineKeyboardButton("Не то (повторить попытку)", callback_data='bad')
        item3 = types.InlineKeyboardButton("Вернуться назад.", callback_data='return')
        markup.add(item1, item2, item3)

        bot.send_message(message.chat.id,
                         f"🔍 На фото определена марка: {car_brand}\n"
                         f"🚘 Модель: {car_model}",
                         reply_markup=markup)
        with open(f'./ChinaCar/{car_brand}/{car_model}/character.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        bot.send_message(message.chat.id, text)

        os.remove(temp_path)

    except Exception as e:
        bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@bot.message_handler(commands=['start'])
def welcome(message):
    sti = open('AnimatedSticker.tgs', 'rb')
    bot.send_sticker(message.chat.id, sti)

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("😊 Отправить фото")
    item2 = types.KeyboardButton("🎲 Получить случайную модель")
    markup.add(item1, item2)

    bot.send_message(message.chat.id,
                     "Добрый день, <b>{0.first_name}</b>!\nЯ - <b>{1.first_name}</b>, "
                     "создан чтобы помочь узнать марку, модель "
                     "и характеристики китайского автомобиля по фотографии!".format(
                         message.from_user, bot.get_me()),
                     parse_mode='html', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def body(message):
    if message.chat.type == 'private':
        if message.text == "😊 Отправить фото":
            sent_msg = bot.send_message(message.chat.id, "Загрузите ваше фотографию:", parse_mode='html')
            bot.register_next_step_handler(sent_msg, save_photo)

        elif message.text == "🎲 Получить случайную модель":
            try:
                get_photo, get_character = getRandomModel.deep_random_path()

                with open(get_photo, 'rb') as filePhoto:
                    bot.send_photo(message.chat.id, filePhoto)

                with open(get_character, 'r', encoding='utf-8') as file:
                    text = file.read()
                bot.send_message(message.chat.id, text)
            except Exception as e:
                bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}")
        else:
            bot.send_message(message.chat.id, "Не удалось тебя понять, попробуй еще раз)")


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data == 'good':
                bot.send_message(call.message.chat.id, 'Отлично, рад был помочь!')
            elif call.data == 'bad':
                bot.send_message(call.message.chat.id, 'Попробуем еще раз! Отправьте новое фото.')
            elif call.data == 'return':
                welcome(call.message)

            # Удаляем inline-клавиатуру
            bot.edit_message_reply_markup(chat_id=call.message.chat.id,
                                          message_id=call.message.message_id,
                                          reply_markup=None)
    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    print("Бот запущен...")
    bot.polling(none_stop=True)
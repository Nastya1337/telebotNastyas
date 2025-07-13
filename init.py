import telebot

# import car
import config
import random
from telebot import types

bot = telebot.TeleBot(config.TOKEN)

@bot.message_handler(commands=['start'])
def welcome(message):
    sti = open('AnimatedSticker.tgs', 'rb')
    bot.send_sticker(message.chat.id, sti)

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("😊 Отправить фото")
    item2 = types.KeyboardButton("🎲 Получить случайную модель")
    markup.add(item1, item2)

    bot.send_message(message.chat.id, "Добрый день, <b>{0.first_name}</b>!\nЯ - <b>{1.first_name}</b>, "
                                      "создан чтобы помочь узнать марку "
                                      "и характеристики машины из Китая по фотографии!".format(message.from_user,
                                        bot.get_me()), parse_mode='html', reply_markup=markup)

@bot.message_handler(content_types=['text'])
def body(message):
    if message.chat.type == 'private':
        if message.text == "😊 Отправить фото":

            markup = types.InlineKeyboardMarkup(row_width=3)
            item1 = types.InlineKeyboardButton("Да! Это та машина.", callback_data='good')
            item2 = types.InlineKeyboardButton("Не то (повторить попытку)", callback_data='bad')
            item3 = types.InlineKeyboardButton("Вернуться назад.", callback_data='return')
            markup.add(item1, item2, item3)

            bot.send_sticker(message.chat.id, sticker=open('AnimatedSticker.tgs', 'rb'), reply_markup=markup)
        elif message.text == "🎲 Получить случайную модель":
            bot.send_message(message.chat.id, str(random.randint(1, 100)))
        else:
            bot.send_message(message.chat.id, "Не удалось тебя понять, попробуй еще раз)")

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data == 'good':
                bot.send_message(call.message.chat.id, 'Отлично, рад был помочь!')
            elif call.data == 'bad':
                bot.send_message(call.message.chat.id, 'Бывает ')
            elif call.data == 'return':
                bot.send_message(call.message.chat.id, 'Возвращаемся в меню!')

            bot.edit_message_text(chat_id=call.message.chat.id,
                                  message_id=call.message.message_id,
                                  text="Отлично, рад был помочь!",
                                  reply_markup=None)

            bot.answer_callback_query(chat_id = call.message.chat.id,
                                      show_alert=False,
                                      text="Текстовое уведомление!")
    except Exception as e:
        print(repr(e))

bot.polling(non_stop=True)



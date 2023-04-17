from flask import request, Flask, jsonify

import telebot
from telegram.ext import Updater, CommandHandler
import requests
import csv
import os
import threading
import time

import config
from messages_processor import commands, text, photo
from messages_processor.messages import clients
from train import epochs_logger

TG_TOKEN = config.BOT_TOKEN

def start_bot(bot):
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print("Опаньки! Вылетела", e.__class__)
        print(e)
        if e == "KeyboardInterrupt":
            return 0
        else:
            start_bot(bot)

def get_csv_data(save_dir):
    csv_file = os.path.join(save_dir, 'results.csv')
    if os.path.exists(csv_file) is False:
        return 'Тренировка не была запущена. Попробуйте заново.'
    with open(csv_file, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter = ",")
        for row in file_reader:
            continue
        row[0] = row[0].replace(" ", "")
        row[6] = row[6].replace(" ", "")
        row[7] = row[7].replace(" ", "")
        row[9] = row[9].replace(" ", "")
        return f'epoch = {int(row[0])+1}\nmAP_0.5 = {row[6]}, mAP_0.5:0.95 = {row[7]}, val/obj_loss = {row[9]}.'


if __name__ == '__main__':
    bot = telebot.TeleBot(TG_TOKEN)
    bot.remove_webhook()
    command_responder = commands.Commands()
    text_responder = text.Text()
    photo_responder = photo.Photo()
    
    @bot.message_handler(commands=['start', 'help', 'info', 'train'])
    def command_handler(message):
        bot.send_chat_action(message.chat.id, 'typing')
        messages_text, keyboards = command_responder.response(message)
        for n, message_text in enumerate(messages_text):
            if message_text:
                bot.send_message(
                    message.chat.id,
                    text = message_text,
                    reply_markup=keyboards[n]
                    )

    def edit_messages(messagetoedit):
        time.sleep(3)
        train_text = f'Загрузка модели' #{percentage=0}'
        n = 0
        current_epoch =  epochs_logger.current_epoch
        epochs_number = epochs_logger.epochs_number
        while current_epoch < epochs_number:
            bot.edit_message_text(
                    chat_id=messagetoedit.chat.id,
                    message_id=messagetoedit.message_id,
                    text = train_text
                    )
            time.sleep(9)
            n += 1
            current_epoch = epochs_logger.current_epoch
            epochs_number = epochs_logger.epochs_number
            train_text = f'Прошло: {n*10} сек.\n'
            train_text += f'Количество эпох {current_epoch} из {epochs_number+1}\n'
            if n > 6 and current_epoch == 0:
                clients.busy['mode'] = None
                clients.busy['model_dir'] = None
                clients.busy['epochs'] = 0
                clients.busy['iters'] = 0
                break
        time.sleep(9)
        epochs_logger.set_current_epoch(0)
        epochs_logger.set_epochs(19)
        text = get_csv_data(epochs_logger.save_dir)
        bot.edit_message_text(
                        chat_id=messagetoedit.chat.id,
                        message_id=messagetoedit.message_id,
                        text = text
                        )
        png_path = os.path.join(epochs_logger.save_dir, 'results.png')
        try:
            results_png = open(png_path, 'rb')
            bot.send_photo(messagetoedit.chat.id,
                           results_png,
                           )
        except:
            print('results.png path', png_path)
        return True

            
    
    @bot.message_handler(content_types = ['text'])
    def text_handler(message):
        bot.send_chat_action(message.chat.id, 'typing')
        messages_text, keyboards = text_responder.response(message)
        for n, message_text in enumerate(messages_text):
            if message_text:
                messagetoedit = bot.send_message(
                                            message.chat.id,
                                            text = message_text,
                                            reply_markup=keyboards[n]
                                            )
        if text_responder.bot_state.get(message.chat.id) == 'train':
            messagetoedit = bot.send_message(
                                        message.chat.id,
                                        text = 'Чтение файлов',
                                        )
            text_responder.set_state('project_x')
            t = threading.Thread(target=edit_messages, args = (messagetoedit,), daemon=True)
            t.start()
            message_text, keyboard = text_responder.train()
            bot.send_message(
                            message.chat.id,
                            text = message_text[0],
                            reply_markup=keyboard[0],
                            )
        if text_responder.bot_state.get(message.chat.id) == 'continue':
            messagetoedit = bot.send_message(
                                        message.chat.id,
                                        text = 'Чтение файлов',
                                        )
            text_responder.set_state('project_x')
            t = threading.Thread(target=edit_messages, args = (messagetoedit,), daemon=True)
            t.start()
            message_text, keyboard = text_responder.continue_train()
            bot.send_message(
                    message.chat.id,
                    text = message_text[0],
                    reply_markup=keyboard[0],
                    )
        
        if text_responder.bot_state.get(message.chat.id) == 'weights_loader_kneron':
            text_responder.set_state('models_x')
            path_to_weights, keyboard = text_responder.load_of_weights_kn()
            print('text_responder path_to_weights', path_to_weights)
            file_nef = open(path_to_weights, "rb")
            bot.send_document(message.chat.id, file_nef)
            
            message_text = 'Готово.'
            bot.send_message(
                            message.chat.id,
                            text = message_text,
                            reply_markup=keyboard
                            )
            file_nef.close()
        
        if text_responder.bot_state.get(message.chat.id) == 'weights_loader_khadas':
            text_responder.set_state('models_x')
            path_to_weights, keyboard = text_responder.load_of_weights_khds()
            print('text_responder path_to_weights', path_to_weights)
            tmfile_model = open(path_to_weights, "rb")
            bot.send_document(message.chat.id, tmfile_model)
            
            message_text = 'Готово.'
            bot.send_message(
                            message.chat.id,
                            text = message_text,
                            reply_markup=keyboard
                            )
            tmfile_model.close()
        
        if text_responder.bot_state.get(message.chat.id) == 'weights_loader_rk3588':
            text_responder.set_state('models_x')
            path_to_weights, keyboard = text_responder.load_of_weights_rk()
            print('text_responder path_to_weights', path_to_weights)
            rknn_model = open(path_to_weights, "rb")
            bot.send_document(message.chat.id, rknn_model)
            
            message_text = 'Готово.'
            bot.send_message(
                            message.chat.id,
                            text = message_text,
                            reply_markup=keyboard
                            )
            rknn_model.close()
        
        if text_responder.bot_state.get(message.chat.id) == 'weights_sender':
            text_responder.set_state('models_x')
            message_text, keyboard = text_responder.send_weights_to_device()
            bot.send_message(
                    message.chat.id,
                    text = message_text,
                    reply_markup=keyboard,
                    )

    
    @bot.message_handler(content_types = ['photo','document'])
    def photo_handler(message):
        if message.content_type == 'document':
            file_id = message.document.file_id
            file_info = bot.get_file(file_id)
            file_request = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TG_TOKEN, file_info.file_path))
            if str(file_request) == '<Response [200]>':
                downloaded_file = bot.download_file(file_info.file_path)
            else:
                downloaded_file = None
        else:
            message.file_name = message.json['photo'][-1]['file_unique_id']+'.jpg'
            file_id = message.json['photo'][-1]['file_id']
            file_info = bot.get_file(file_id)
            file_request = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TG_TOKEN, file_info.file_path))
            if str(file_request) == '<Response [200]>':
                print('response 200')
                downloaded_file = bot.download_file(file_info.file_path)
            else:
                print('response None')
                downloaded_file = None
        
        photo_responder.documents_dict[message.chat.id] = downloaded_file
        bot.send_chat_action(message.chat.id, 'typing')
        paths, keyboards = photo_responder.doc_response(message)
        for n, path in enumerate(paths):
            if path:
                if path.endswith(('bmp', 'jpg', 'png', 'jpeg', 'JPG', 'JPEG')):
                    photo = open(path, 'rb')
                    bot.send_photo(message.chat.id, 
                               photo,
                               reply_markup=keyboards[n])
                
                elif path.endswith(('json', 'zip')):
                    doc = open(path, 'rb')
                    bot.send_document(message.chat.id,
                                      doc,
                                      reply_markup=keyboards[n])
                
                else:
                    bot.send_message(
                            message.chat.id,
                            text = path,
                            reply_markup=keyboards[n]
                            )
            else:
                if n != 0:
                    clients.busy['mode'] = None 
                    bot.send_message(
                            message.chat.id,
                            text = 'Отсутствует модель.'
                            )
    
    start_bot(bot)
            

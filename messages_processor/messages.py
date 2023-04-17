import pickle
import os
import telebot

import yolov5.train as train
import yolov5.detect as detect
import yolov5.export as export


class Clients:
    DEFAULT_PARAMS = dict(
            epochs=20,
            thresh=0.5,
                )
    
    def __init__(self):
        
        self.project_name = {}
        self.project_params = {}
        self.busy = dict(mode=None,
                         model_dir=None,
                         epochs=0,
                         iters=0,
                        )
        self.address_dict = {}
        
    def get_labels(self, labels_set, dataset_dir):
        labels_list = list(labels_set)
        dataset_dir = dataset_dir[:-3] + '/custom.yaml'
        try:
            with open(dataset_dir, 'r') as config_file:
                line = config_file.readlines()[-1]
                names = line[7:]
                names = names.replace("'", " ")
                names = names.replace("[", " ")
                names = names.replace("]", " ")
                names = names.replace(",", " ")
                base_labels = names.split()
        except:
            base_labels = []
        for new_label in labels_list:
            if new_label not in base_labels:
                base_labels.append(new_label)
        return base_labels
    
    def set_project_name(self, chat_id, project_name):
        self.project_name[chat_id] = project_name
    
    def get_project_name(self, chat_id):
        return self.project_name.get(chat_id)
    
    def set_project_parameters(self, project_name, **kargs):
        # initiate project parameters storage for the session
        if project_name not in self.project_params.keys():
            self.project_params[project_name] = self.DEFAULT_PARAMS
        # update parameters values    
        for key in kargs.keys():
            self.project_params[project_name][key] = kargs[key]
    
    def get_project_parameters(self, project_name):
        if self.project_params.get(project_name):
            return self.project_params.get(project_name)
        else:
            return self.DEFAULT_PARAMS
        
    def current_addres(self, chat_id):
        return self.address_dict.get(chat_id)
    
    def set_addres(self, addres, chat_id):
        self.address_dict[chat_id] = addres
    
    def estimate_time_left(self):
        if self.busy['mode']=='train':
            _, epoch, it = read_train_log(self.busy['model_dir'])
            epochs_left = self.busy['epochs'] - epoch
            iters_left = self.busy['iters'] - it
            total_iters_left = epochs_left * self.busy['iters'] + iters_left
            return total_iters_left * 0.05 + 1
        elif self.busy['mode']=='val':
            total_iters_left = self.busy['iters']
            return total_iters_left * 0.1 + 0.5
        else:
            return -1

        

clients = Clients()

class Keyboards:
    
    def __init__(self):
        self.id = 0
    
    @staticmethod
    def inline_keyboard(buttons_data_list):
        if not buttons_data_list:
            return None
        keyboard = telebot.types.InlineKeyboardMarkup()
        for button_data in buttons_data_list:
            button = telebot.types.InlineKeyboardButton(text = button_data.get('text'),
                                                        callback_data = button_data.get('callback'),
                                                        url = button_data.get('url'))
            keyboard.row(button)
        return keyboard
    
    @staticmethod
    def reply_keyboard(buttons_data_list,
                                resize=True,
                                one_time = True):
        if not buttons_data_list:
            return None
        keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=resize,
                                                     one_time_keyboard = one_time)
        for button_data in buttons_data_list:
            b3_flag = True
            button_text = button_data.get('text')
            if type(button_text) == type(()):
                try:
                    button_text1, button_text2 = button_data.get('text')
                    b3_flag = False
                except:
                    button_text1, button_text2, button_text3 = button_data.get('text')
                button1 = telebot.types.KeyboardButton(button_text1,
                                                       request_contact = button_data.get('request_contact'))
                button2 = telebot.types.KeyboardButton(button_text2,
                                                       request_contact = button_data.get('request_contact'))
                if b3_flag:
                    button3 = telebot.types.KeyboardButton(button_text3)
                    keyboard.row(button1, button2, button3)
                else:
                    keyboard.row(button1, button2)
            else:
                button = telebot.types.KeyboardButton(button_text,
                                                      request_contact = button_data.get('request_contact'))
                keyboard.row(button)
        return keyboard
            

def init_states():
    pickle_file = 'bot_state'
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as state:
            return pickle.load(state)
    else:
        return {}

class Messages:
    '''
        bot_state = {chat_id : state}
        simple_switch = {old_state: new_state,
                         text1.lower(): new_state,
                         ...}
        complex_switch = {old_state: {text1.lower() : new_state1, text2.lower() : new_state2},
                          text1.lower(): {state1 : new_state1, state2 : new_state2},
                          ...}
        response - main method
        set_state(self, state=None) - устанавливает состояние клиенту, перед
                                    вызовом, нужно поменять current_message
        get_state(self) - возвращает состояние клиента по его id
        switch_state(self, state=None, text=None) переключает состояния исходя из условий
                                            state1 -> state2 - не важно какой текст
                                            text1 -> state2 - не важно какое состояние
                                            state1+text1 -> state2
        call_of_function(self) - Вызывает функцую из словаря функций (в зависимости от
                                                                        состояния )
    '''
    
    TEXT_DICT = {}
    BOTTOM_DICT = {}
    FUNCTION_DICT = {}
    
    bot_state = init_states()
    simple_switch = {}
    complex_switch = {}
    
    def __init__(self):
        self.current_message = ''
    
    def set_state(self, state=None):
        chat_id = self.current_message.chat.id
        self.bot_state[chat_id] = state
    
    def get_state(self):
        try:
            chat_id = self.current_message.chat.id
            return self.bot_state.get(chat_id)
        except:
            return 0
    
    def save_state(self):
        pickle_file = 'bot_state'
        with open(pickle_file, "wb") as f:
            pickle.dump(self.bot_state, f)
    
    @property
    def answer_text(self):
        state = self.get_state()
        answer_text = self.TEXT_DICT.get(state)
        return answer_text
    
    @property
    def keyboard_data(self):
        state = self.get_state()
        keyboard_data = self.BOTTOM_DICT.get(state) #В случае чего keyboard_data - None
        return keyboard_data
    
    def switcher(self):
        old_state = self.get_state()
        try:
            message_text = self.current_message.text.replace(' ', '_').lower()
        except:
            message_text = '_document_'
        self.switch_state(old_state, message_text)
        print(f'State {old_state} switcher -> {self.get_state()}')
        self.save_state()
    
    def switch_state(self, state=None, text=None):
        if not self.get_state(): 
            self.set_state(state='0')
            return True
        new_state = self.simple_switch.get(state)
        if new_state:
            self.set_state(new_state)
            return True
        new_state = self.simple_switch.get(text)
        if new_state:
            self.set_state(new_state)
            return True

        new_state = self.complex_switch.get(state)
        if new_state:
            new_state = new_state.get(text)
            if new_state:
                self.set_state(new_state)
                return True
            else:
                self.set_state('anything')
        new_state = self.complex_switch.get(text)
        if new_state:
            new_state = new_state.get(state)
            if new_state:
                self.set_state(new_state)
                return True
            else:
                self.set_state('anything')
    
    @property
    def call_of_function(self):
        state = self.get_state()
        try:
            return self.FUNCTION_DICT[state]()
        except:
            return '', None
    
    def response(self, message):
        self.current_message = message
        self.switcher()
        answer_text = [self.answer_text]
        answer_keyboard = [Keyboards.reply_keyboard(self.keyboard_data)]
        function_response = self.call_of_function # ('Text', 'keyboard_data')
        answer_text.append(function_response[0])
        answer_keyboard.append(Keyboards.reply_keyboard(function_response[1]))
        return answer_text, answer_keyboard
    
    def doc_response(self, message):
        self.current_message = message
        answer_text = [self.answer_text]
        answer_keyboard = [Keyboards.reply_keyboard(self.keyboard_data)]
        
        function_response = self.call_of_function # ('Text', 'keyboard_data')
        
        # several texts case
        if type(function_response[0]) in (list, tuple):
            for text, keyboard in zip(function_response[0], function_response[1]):
                answer_text.append(text) #text from function
                answer_keyboard.append(Keyboards.reply_keyboard(keyboard))
        # single text returned
        else:
            answer_text.append(function_response[0]) #text from function
            answer_keyboard.append(Keyboards.reply_keyboard(function_response[1]))

        return answer_text, answer_keyboard

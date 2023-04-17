from messages_processor.messages import *

PATH = '/root/yolov5_tg/projects/'

class Commands(Messages):
    
    def __init__(self):
        Messages.__init__(self)
        self.set_functions()
    
    TEXT_DICT = {'0' : 'Здравствуйте! Здесь вы можете обучать нейросети без программирования. ' +\
                       'Для получения информации по работе с ботом напишите команду /info.',
                 'menu' : 'Для получения информации по работе с ботом напишите команду /info.',
                 'info' : 'В Projects отображаются ваши проекты, там же вы можете создать новый '+\
                          'проект - кнопка "Создать" \n' +\
                          'В DataSet можно скинуть аннотации, которые будут использоваться ' +\
                          'для обучения.\n' +\
                          'Для  запуска тренировки:\n' +\
                          '1) Выберите проект. \n' +\
                          '2) Задайте параметры \n' +\
                          '3) Нажмите "Обучение" \n',
                 'help' : 'Если у вас появились вопросы свяжитесь с нами.',
                 'mistake' : 'Сначала выполните проверку',
                 'anything' : 'Для получения информации по работе с ботом напишите команду /info',
                 'user_menu' :  'Для получения информации по работе с ботом напишите команду /info.'
                }
    
    BOTTOM_DICT = {
                   'menu' : [{'text':'Разметка'}, 
                             {'text':'Проекты'}, 
                             {'text':'Модели'},
                             {'text':'Демо'},
                            ],
                   'help' : [{'text':'Связаться с разработчиками',
                              'url':'t.me/nanoparticles_nsk'}],
                   'user_menu' : [{'text':('Подписка', 'Инференс')}],
                   'state_ok' : [{'text':'Menu'}]
                  }
    simple_switch = {'/start' : 'menu',
                     '/info' : 'info',
                     'mistake' : 'menu',
                     '/help' : 'help',
                     '/roboflow_link' : 'roboflow',
                    }
    
    
    def set_functions(self):
        self.FUNCTION_DICT = {'0' : self.create_user_directory
                             }
    
    def create_user_directory(self):
        chat_id = self.current_message.chat.id
        start_text = self.current_message.text
        try:
            referal_link = start_text.split()[1]
        except:
            referal_link = ''
        if referal_link == '1a2b3c4d5e6f':
            self.set_state('user_menu')
            project = '"my_project"'
            text = f'Проект {project}\nНажмите инференс для работы'
            keyboard = [{'text':('Подписка','Инференс')}
                       ]
            return text, keyboard
        else:
            self.set_state('menu')
            text = 'Меню: '
            keyboard = self.BOTTOM_DICT['menu']
            path = PATH + str(chat_id)+'/projects'
            if os.path.exists(path):
                return text, keyboard
            else:
                os.mkdir(PATH+str(chat_id))
                os.mkdir(PATH+str(chat_id)+'/projects')
                return text, keyboard

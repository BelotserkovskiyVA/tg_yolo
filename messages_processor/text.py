import os
import requests
import subprocess
import onnx
import numpy as np
import zipfile
import shutil

from yolov5.train import epochs_logger

from messages_processor.messages import *

PATH = '/root/yolov5_tg/projects'

class Text(Messages):
    
    def __init__(self):
        Messages.__init__(self)
        self.set_functions()
    
    PARAMS_DICT = dict(
            thresh={'Все предположения':0.01,
                    'Без шума':0.1,
                    'Достоверные':0.3,
                    'Проверенные':0.7,
                   },
            cont={'Начать заново':False,
                   'Продолжить':True,
                  }
        )
        
    TEXT_DICT = {'0' : 'Для начала работы напишите команду /start ',
                 'menu' : 'Меню: ',
                 'demo' : 'Загрузите любую картинку или сделайте снимок камерой для этого чата и протестируйте, как работает YOLOv5.',
                 'projectes' : 'Выберите проект или создайте новый.',
                 'train_annots' : 'Загрузите аннотации для обучения в формате JSON (LabelMe) '+\
                                     'или в формате YOLO (Изображения, Лейблы и config.yaml)',
                 'test_annots' : 'Загрузите аннотации для проверки в формате JSON (LabelMe) '+\
                                 'или в формате YOLO (Изображения, Лейблы и config.yaml)',
                 'roboflow' : 'Скиньте ссылку на public.roboflow.com датасет',
                 'list_train_annots' : 'Список загруженных аннотаций для обучения:',
                 'list_test_annots' : 'Список загруженных аннотаций для проверки',
                 'create_project' : 'Введите название проекта и затем нажмите "Ok".',
                 'name_setted' : 'Имя проекта установлено:',
                 'project_x' : 'Для запуска тренировки загрузите данные, \n' +\
                               'установите уровень обучения \n' +\
                               'и нажмите "Начать обучение".',
                 'delete_project_state': 'Вы уверены?',
                 'config': 'Текущие параметры обучения сети:',
                 'set_epochs' : 'Выберите уровень обучения сети',
                 'set_thresh' : 'Какие предположения показывать?',
                 'inf_thresh' : 'Какие предположения показывать?',
                 'continue' : 'Продолжаем тренировку. Чтобы не нарушить работу бота, дождитесь сообщения об окончании.',
                 'check_train' : 'Запускается обучение сети. Чтобы не нарушить работу бота, дождитесь сообщения с результатами тренировки',
                 'weights_sender' : 'Подготавливаются веса для отправки. Это может занять несколько минут.',
                 'weights_loader' : 'Подготавливаются веса для загрузки. Это может занять несколько минут.',
                 'weights_loader_kneron' : 'Подготавливаются веса для загрузки. Это может занять несколько минут.',
                 'weights_loader_khadas' : 'Подготавливаются веса для загрузки. Это может занять несколько минут.',
                 'weights_loader_rk3588' : 'Подготавливаются веса для загрузки. Это может занять несколько минут.',
                 'models_x' : 'Обученая модель:',
                 'failed' : 'Что-то пошло не так.',
                 'anything' : 'Я вас не понимаю. =( Для получения информации команда /start',
                 'test_data' : "Загрузите изображение для распознавания.",
                 
                 'user_menu' : 'Меню: ',
                 'subscription_state' : 'Подписка действует до',
                 'inference_state' : 'Для получения результатов инференса загрузите изображение '+\
                                     'в формате .jpg или .png.'
                 
                }
    
    BOTTOM_DICT = {'menu' : [{'text':'Разметка'}, 
                             {'text':'Проекты'}, 
                             {'text':'Модели'},
                             {'text':'Демо'},
                            ],
                   'demo' : [{'text':'Назад'},
                            ],
                   'dataset' : [{'text':'Для обучения'},
                                {'text':'Для проверки'},
                                {'text':'Назад'},
                               ],
                   'train_annots' : [{'text':('Мои аннотации','Отмена')}
                                      ],
                   
                   'test_annots' : [{'text':('Мои аннотации','Отмена')}
                                  ],
                   'roboflow' : [{'text':'Отмена'}
                                ],
                   'list_train_annots' : [{'text':('Очистить список','Назад')}
                                  ],
                   'list_test_annots' : [{'text':('Очистить список','Назад')}
                                  ],

                   'delete_images': [{'text':'Назад'}
                                  ],
                   'create_project' : [{'text':'Отмена'}
                                      ],
                   'name_setted' : [{'text':'Ok'}
                                   ],
                   'project_x' : [{'text':'Данные'},
                                  {'text':'Уровень обучения'}, 
                                  {'text':('Начать обучение', 'Продолжить обучение')},
                                  {'text':('Удалить проект','Назад')},
                                 ],
                   'delete_project_state': [{'text':('Ok','Отмена')}
                                           ],
                   'config' : [{'text':('Уровень обучения')}, 
                               {'text':'Продолжение тренировки'}, 
                               {'text':'Назад'},
                              ],
                   'state_ok' : [{'text':'Назад'}
                                ],
                   'set_epochs' : [{'text':'Начальный'},
                                   {'text':'Средний'},
                                   {'text':'Опытный'},
                                  ],
                   'set_thresh' : [{'text':'Все предположения'},
                                   {'text':'Без шума'},
                                   {'text':'Достоверные'},
                                   {'text':'Проверенные'},
                                  ],
                   'check_train' : [{'text' : 'Назад'}
                             ],
                   'continue' : [{'text' : 'Назад'}
                                 ],
                   'models_x' : [{'text':'Параметры'}, 
                                 {'text':'Распознавание'},
                                 #{'text':('Веса Kneron', 'Веса Khadas', 'Отправить веса')},
                                 {'text':('Веса Khadas', 'Веса RK3588', 'Отправить веса')},
                                 {'text':'Назад'},
                                ],
                   'model_config' : [{'text':'Порог распознавания'}, 
                                     {'text':'Назад'},
                                    ],
                   'weights_loader' : [{'text' : 'Назад'}
                                      ],
                   'weights_loader_kneron': [{'text' : 'Назад'}
                                              ],
                   'weights_loader_khadas': [{'text' : 'Назад'}
                                              ],
                   'weights_loader_rk3588': [{'text' : 'Назад'}
                                              ],
                   'weights_sender' : [{'text' : 'Назад'}
                                      ],
                   'inf_thresh' : [{'text':'Все предположения'},
                                   {'text':'Без шума'},
                                   {'text':'Достоверные'},
                                   {'text':'Проверенные'},
                                  ],
                   'test_data' : [{'text':'Назад'},
                                 ],
                   
                   'user_menu' : [{'text':('Подписка', 'Инференс')}
                                 ],
                   'subscription_state' : [{'text':'Отмена'}
                                          ],
                   'inference_state' : [{'text':'Назад'}
                                       ]
                  }
    
    simple_switch = {'projectes' : 'choose_project',
                     'models' : 'choose_model',
                     'delete_project_state' : 'delete_project',
                     'create_project' : 'set_project_name',
                     'delete_train_annots' : 'train_annots',
                     'delete_test_annots' : 'test_annots',
                     'annotation' : 'state_ok',
                     'test_data' : 'models_x',
                     'state_ok' : 'menu',
                     'demo' : 'menu',
                    }
    complex_switch = {'0' : {'разметка':'annotation',
                             'проекты':'projectes',
                             'модели':'models',
                             'демо':'demo',
                                },
                      'menu' : {'разметка':'annotation',
                                'проекты':'projectes',
                                'модели':'models',
                                'демо':'demo',
                                },
                      'roboflow' : {'отмена' : 'dataset'},
                      'train_annots' : {'отмена':'dataset',
                                        'мои_аннотации' : 'list_train_annots'
                                       },
                      'list_train_annots' : {'назад':'train_annots',
                                             'очистить_список' : 'delete_train_annots'
                                            },
                      'test_annots' : {'отмена':'dataset',
                                       'мои_аннотации' : 'list_test_annots'
                                       },
                      'list_test_annots' : {'назад':'test_annots',
                                            'очистить_список' : 'delete_test_annots'
                                            },
                      'name_setted' : {'ok':'project_x',
                                     },
                      'project_x' : {'данные':'dataset',
                                     'уровень_обучения':'set_epochs',
                                     'начать_обучение':'check_train',
                                     'продолжить_обучение':'continue',
                                     'удалить_проект':'delete_project_state',
                                     'назад':'projectes',
                                 },
                      'set_epochs' : {'начальный':'set_epochs',
                                      'средний':'set_epochs',
                                      'опытный':'set_epochs'
                                    },
                      'set_thresh' : {'все_предположения':'config',
                                      'без_шума':'config',
                                      'достоверные':'config',
                                      'проверенные':'config',
                                    },
                      'state_ok' : {'назад':'menu' 
                                   },
                      'models_x' : {'параметры':'model_config',
                                    'распознавание':'test_data',
                                    'скачать_веса':'weights_loader',
                                    #'веса_kneron':'weights_loader_kneron',
                                    'веса_khadas':'weights_loader_khadas',
                                    'веса_rk3588':'weights_loader_rk3588',
                                    'отправить_веса':'check_addres',
                                    'назад':'models',
                                   },
                      'model_config' : {'порог_распознавания':'inf_thresh',
                                        'назад':'models_x', 
                                       },
                      'inf_thresh' : {'все_предположения':'model_config',
                                      'без_шума':'model_config',
                                      'достоверные':'model_config',
                                      'проверенные':'model_config',
                                    },
                      
                      'user_menu' : {'подписка':'subscription_state',
                                      'инференс':'inference_state'
                                     },
                      'subscription_state' : {'отмена':'user_menu',
                                          },
                      'inference_state' : {'назад':'user_menu',
                                          },
                        }
    
    def set_functions(self):
        self.FUNCTION_DICT = {'annotation' : self.instruct_annotator,
                              
                              'projectes' : self.get_projects_list,
                              'choose_project' : self.set_project,
                              'set_project_name' : self.set_project_name,
                              'delete_project' : self.delete_project,
                              'dataset' : self.choose_dataset,
                              'set_epochs' : self.set_config,
                              'check_train' : self.check_busy,
                              'check_addres' : self.check_addres,
                              'addres_entry' : self.set_addres,
                              'roboflow' : self.download_roboflow_data,
                              'list_train_annots' : self.get_train_list,
                              'list_test_annots' : self.get_test_list,
                              'delete_train_annots' : self.delete_train_list,
                              'delete_test_annots' : self.delete_test_list,
                              
                              'models' : self.get_models_list,
                              'model_config' : self.set_inf_config,
                              'choose_model' : self.set_model,
                             }
    
    def get_models_list(self):
        text = 'Мои обученные модели:'
        keyboard = []
        chat_id = self.current_message.chat.id
        proj_dir = os.path.join(str(chat_id), 'projects')
        base_path = os.path.join('/root', 'yolov5_bot')
        if os.path.isdir(proj_dir):
            projects_list = [f for f in os.listdir(proj_dir) if not f.startswith('.')]
            keyboard = [{'text':proj_name} for proj_name in projects_list]
        else:
            os.chdir(base_path)
            projects_list = [f for f in os.listdir(proj_dir) if not f.startswith('.')]
            keyboard = [{'text':proj_name} for proj_name in projects_list]
        keyboard.append({'text':'Назад'})
        return text, keyboard
    
    def set_model(self):
        message_text = self.current_message.text
        chat_id = self.current_message.chat.id
        proj_dir = os.path.join(str(chat_id), 'projects')
        projects_list = [f for f in os.listdir(proj_dir) if not f.startswith('.')]
        if message_text.lower() == 'назад':
            self.set_state('menu')
            text = 'Меню: '
            keyboard = self.BOTTOM_DICT['menu']
        elif message_text in projects_list:
            self.set_state('models_x')
            clients.set_project_name(chat_id, message_text)
            project_name = clients.get_project_name(chat_id)
            clients.set_project_parameters(project_name, **{})
            text = f'Выбрана модель {message_text}\n'
            text += f'Текущие параметры {clients.get_project_parameters(project_name)}'

            keyboard = self.BOTTOM_DICT['models_x']
        else:
            self.set_state('menu')
            text = 'Такой модели не существует.'
            keyboard = self.BOTTOM_DICT['menu']
        return text, keyboard
    
    def instruct_annotator(self):
        text = 'Как размечать данные?\n \
 1. Перед тренировкой необходимо разметить данные, т.е. обвести контурами интересующие объекты.\n \
 2. Размечать необходимо в программе labelme для Windows, которую можно скачать по ссылке\n \
 https://disk.yandex.ru/d/-lmmiAIZUIrLYA\n \
 3. В программе labelme нужно открыть изображение нажав кнопку open, выбрать инструмент Polygon \
 и обвести каждый объект контуром, стараясь точно следовать границе объекта.\n \
 4. Сохранить размеченное изображение, нажав кнопку save. Должен появиться файл с раширением json \
 в той же папке, где было исходное изображение.\n \
 5. Для обучения нейросети файл нужно загрузить в этот чат телеграм, используя кнопку Данные'
        
        self.set_state('state_ok')
        keyboard = self.BOTTOM_DICT['state_ok']
        return text, keyboard

    def set_inf_config(self):
        message_text = self.current_message.text
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)

        if 'назад' == message_text.lower():
            self.set_state('model_x')
            text = self.TEXT_DICT['model_x']
            keyboard = self.BOTTOM_DICT['model_x']
            return text, keyboard
        new_params = {}
        for key in self.PARAMS_DICT.keys():
            if message_text in self.PARAMS_DICT[key].keys():
                new_params[key] = self.PARAMS_DICT[key][message_text]
                
        clients.set_project_parameters(project_name, **new_params)
        updated_params =  clients.get_project_parameters(project_name)
        
        text = ', '.join([f'{key}: {updated_params[key]}' for key in updated_params.keys()])
        keyboard = self.BOTTOM_DICT['model_config']
        return text, keyboard

    def set_config(self):
        text = 'Текущие параметры:\n'
        keyboard = self.BOTTOM_DICT['set_epochs']
        message_text = self.current_message.text
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)

        new_params = {}
        if message_text in ['Начальный', 'Средний', 'Опытный']:
            new_params['epochs'] = self.set_epoch_n(message_text)
            keyboard = self.BOTTOM_DICT['project_x']
            self.set_state('project_x')
        for key in self.PARAMS_DICT.keys():
            if message_text in self.PARAMS_DICT[key].keys():
                new_params[key] = self.PARAMS_DICT[key][message_text]
                
        clients.set_project_parameters(project_name, **new_params)
        updated_params =  clients.get_project_parameters(project_name)
        
        text += ', '.join([f'{key}: {updated_params[key]}' for key in updated_params.keys()])
        
        
        return text, keyboard
    
    def set_epoch_n(self, message_text):
        iters = {'Начальный' : 100,
                 'Средний' : 500,
                 'Опытный' : 1000
                }
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        train_data_dir = os.path.join(str(chat_id), 'projects', str(project_name), 'dataset', 'train', 'images')
        imgs_number = len(os.listdir(train_data_dir))
        batch_size = min(imgs_number, 32)
        
        epochs_n = iters.get(message_text)*batch_size/imgs_number
        epochs_n = max(2, int(epochs_n))
        epochs_logger.set_epochs(epochs_n - 1)
        return epochs_n
    
    def get_projects_list(self):
        text = 'Мои проекты:'
        keyboard = []
        chat_id = self.current_message.chat.id
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects')
        projects_list = [f for f in os.listdir(proj_dir) if not f.startswith('.')]
        for proj_name in projects_list:
            keyboard.append({'text':proj_name}) 
        keyboard.append({'text':('Создать', 'Отмена')})
        return text, keyboard
    
    def set_project(self):
        message_text = self.current_message.text
        chat_id = self.current_message.chat.id
        path = os.path.join(PATH, str(chat_id))
        projects_list = os.listdir(path+'/projects/')
        
        if message_text.lower() == 'создать':
            self.set_state('create_project')
            text = 'Отправьте название проекта и затем нажмите "Ok".'
            keyboard = [{'text':'Отмена'}]
        elif message_text.lower() == 'отмена':
            self.set_state('menu')
            text = 'Меню: '
            keyboard = self.BOTTOM_DICT['menu']
        elif message_text in projects_list:
            self.set_state('project_x')
            clients.set_project_name(chat_id, message_text)
            project_name = clients.get_project_name(chat_id)
            clients.set_project_parameters(project_name, **{})
            
            text = f'Выбран проект {message_text}.\n'
            text += self.TEXT_DICT['project_x']
            keyboard = self.BOTTOM_DICT['project_x']
            
        else:
            text = 'Такого проекта не существует.'
            keyboard = None
        return text, keyboard
    
    def set_project_name(self):
        text = ''
        keyboard = None

        chat_id = self.current_message.chat.id
        path = os.path.join(PATH, str(chat_id))
        user_dir = os.path.join(path, 'projects')
        num_projects = 0
        for obj in os.listdir(user_dir):
            path = os.path.join(user_dir, obj)
            if os.path.isdir(path) and not obj.startswith('.'):
                num_projects +=1

        a = [88888888, 555555555, 777777777]
        if not chat_id in a:
            if num_projects>=2:
                text = 'Лимит проектов - не более 2 на пользователя. Чтобы снять ограничение обратитесь к администраторам'
                self.set_state('menu')
                keyboard = self.BOTTOM_DICT['menu']
                return text, keyboard
        
        message_text = self.current_message.text
        if message_text.lower() == 'ok':
            self.set_state('project_x')
        elif message_text.lower() == 'отмена':
            self.set_state('menu')
            text = 'Меню: '
            keyboard = self.BOTTOM_DICT['menu']
        else:
            chat_id = self.current_message.chat.id
            message_text = message_text.replace(' ', '_')
            path = os.path.join(PATH, str(chat_id))
            proj_dir = os.path.join(path, 'projects', message_text)
            os.mkdir(proj_dir)
            for user_folder in ['jsons_list', 'data', 'dataset', 'train', 'models']:
                data_dir = os.path.join(proj_dir, user_folder)
                os.mkdir(data_dir)
            data_dir = os.path.join(proj_dir, 'jsons_list/train_data')
            os.mkdir(data_dir)
            data_dir = os.path.join(proj_dir, 'jsons_list/test_data')
            os.mkdir(data_dir)
            for dataset in ['dataset/train', 'dataset/valid']:
                data_dir = os.path.join(proj_dir, dataset)
                os.mkdir(data_dir)
                os.mkdir(data_dir+'/images')
                os.mkdir(data_dir+'/labels')

            clients.set_project_name(chat_id, message_text)
            self.set_state('name_setted')
            text = f'"{message_text}"'
            keyboard = [{'text':'Ok'}]
        return text, keyboard
    
    def delete_project(self):
        message_text = self.current_message.text
        text = ''
        keyboard = None
        if message_text.lower() == 'отмена':
            self.set_state('project_x')
            text = 'Для получения результатов инференса загрузите изображение. \n' +\
                   'Для запуска тренировки установите параметры обучения \n' +\
                   'и нажмите "Начать обучение".'
            keyboard = self.BOTTOM_DICT['project_x']
        elif message_text.lower() == 'ok':
            chat_id = self.current_message.chat.id
            project_name = clients.get_project_name(chat_id)
            path = os.path.join(PATH, str(chat_id))
            path = path+'/projects/' + project_name
            shutil.rmtree(path)
            self.set_state('projectes')
            text, keyboard = self.get_projects_list()
        else:
            self.set_state('project_x')
            text = 'Проект не был удален'
            keyboard = self.BOTTOM_DICT['project_x']
        return text, keyboard
    
    def choose_dataset(self):
        url = self.current_message.text
        message_text = self.current_message.text.replace(' ', '_').lower()
        new_state = {'для_обучения' : 'train_annots',
                     'для_проверки' : 'test_annots',
                     'назад':'project_x',
                    }
        new_state = new_state.get(message_text)
        if new_state is not None:
            self.set_state(new_state)
            text = self.TEXT_DICT[new_state]
            keyboard = self.BOTTOM_DICT[new_state]
            
            return text, keyboard
        else:
            text = 'Здесь можно добавить данные для обучения нейронной сети и для проверки результатов.\n'+\
                   'Можно добавить данные вручную, а можно скинуть ссылку на датасет в app.roboflow.com \n'+\
                   '\nВыберите назначение данных:'
                   
            keyboard = self.BOTTOM_DICT['dataset']
            if 'https' in message_text:
                text, keyboard = self.download_roboflow_data(url)
            return text, keyboard
            
    
    def download_roboflow_data(self, message_text):
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        dataset_dir = os.path.join(proj_dir, 'dataset')
        file_name = os.path.join(dataset_dir,'roboflow.zip')
        f = open(file_name,"wb")
        url = message_text
        try:
            text = 'Скачивание успешно завершено. Можете тренировать модель.'
            keyboard = self.BOTTOM_DICT['dataset']
            r = requests.get(url)
            f.write(r.content)
            f.close
            dataset_zip = zipfile.ZipFile(file_name)
            dataset_zip.extractall(dataset_dir)
            dataset_zip.close()
            self.cp_roboflow_yaml(proj_dir)
        except:
            text = 'Не удалось скачать датасет. Ссылка должна быть в виде https://app.roboflow.com/ds/.....'
        return text, keyboard
    
    def cp_roboflow_yaml(self, proj_dir):
        dataset_dir = os.path.join(proj_dir, 'dataset')
        rbflw_yaml_path = os.path.join(dataset_dir, 'data.yaml')
        yaml_path = os.path.join(proj_dir, 'data', 'custom.yaml')
        labels_list = clients.get_labels(set(), dataset_dir) 
        
        shutil.copyfile(rbflw_yaml_path, yaml_path)
        
        labels_list = clients.get_labels(set(labels_list), dataset_dir)
        nc = len(labels_list)
        dataset_path = f"path: {dataset_dir}\n"
        train_path = "train: train/images\n"
        val_path = "val: valid/images\n"
        class_number = f"nc: {nc}\n"
        class_name = f"names: {labels_list}"
        head = dataset_path+\
                train_path+\
                val_path+\
                class_number+\
                class_name
        path_to_config = dataset_dir[:-3] + '/custom.yaml'
        with open(path_to_config, 'w') as config_file:
            config_file.write(head)
    
    def check_busy(self):
        text = 'Шкала прогресса:'
        keyboard = None
    
        if clients.busy['mode']:
            self.set_state('project_x')
            ETA = epochs_logger.epochs_number - epochs_logger.current_epoch
            ETA = ETA/59
            text = f'В данный момент нельзя запустить тренировку.\nБот освободится примерно через {ETA:0.1f} минут'
            return text, keyboard
        else:
            self.set_state('train')
            return text, None
    
    def train(self):
        # train
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        result_dir = proj_dir + '/train'
        data_dir = proj_dir +'/data/custom.yaml'
        model_dir = 'models/yolov5m.yaml'
        weights = 'yolov5m_leaky.pt'
        freeze = '10'
        project_parameters = clients.get_project_parameters(project_name)
        batch = '16'
        num_epochs = project_parameters.get('epochs')
        clients.busy['mode'] = 'train' 
        result_dir = proj_dir + '/train'
        val_data = os.path.join(proj_dir, 'dataset/valid/labels')
        if len(os.listdir(val_data)) <= 1:
            self.set_state('project_x')
            text = ['Добавьте данные для проверки. Тренировка не запущена']
            keyboard = self.BOTTOM_DICT['project_x']
            keyboard = [Keyboards.reply_keyboard(keyboard)]
            epochs_logger.set_current_epoch(22)
            epochs_logger.set_epochs(20)
            return text, keyboard
#         opt = train.parse_opt()
#         opt.data = data_dir
#         opt.noautoanchor = True
#         opt.cfg = model_dir
#         opt.weights = weights
#         opt.freeze = freeze
#         opt.batch = batch
#         opt.epochs = num_epochs
#         opt.project = result_dir
#         opt.name = 'exp'
        clients.busy['mode'] = 'train'
        savedPath = os.getcwd()
        os.chdir('/root/yolov5_tg/tg_yolo/yolov5/')
        command = '/root/yolov5_tg/tg_yolo/yolov5/train.py'
        params = f'--weights {weights} --data {data_dir} --cfg {model_dir} --freeze {freeze} --batch {batch} --epochs {num_epochs} --project {result_dir}'
        popen = subprocess.Popen('python3 '+ command +' ' +  params, executable='/bin/bash', shell=True)
        popen.wait()
        #train.main(opt)
        
        os.chdir(savedPath)
        clients.busy['mode'] = None
        clients.busy['model_dir'] = None
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 0

        print('training completed')
        self.set_state('project_x')
        text = ['Обучение закончено:']
        keyboard = self.BOTTOM_DICT['check_train']
        keyboard = [Keyboards.reply_keyboard(keyboard)]
        return text, keyboard

    def continue_train(self):
        text = ''
        keyboard = None
        if clients.busy['mode']:
            self.set_state('project_x')
            ETA = epochs_logger.epochs_number - epochs_logger.current_epoch
            ETA = ETA/60 + 1
            text = f'В данный момент нельзя запустить тренировку.\nБот освободится примерно через {ETA:0.1f} минут'
            return text, keyboard
        
        # train
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        result_dir = proj_dir + '/train'
        data_dir = proj_dir +'/data/custom.yaml'
        model_dir = 'models/yolov5m.yaml'
        exp = os.listdir(result_dir)
        if len(exp) == 0:
            self.set_state('project_x')
            text = 'training folder is empty'
            return text, keyboard
        exp.sort()
        exp.sort(key=len)
        if '.ipynb' in exp[-1]:
            _ = exp.pop()
        exp = exp[-1]
        weights = os.path.join(result_dir, f'{exp}', 'weights', 'best.pt')
        freeze = '10'
        project_parameters = clients.get_project_parameters(project_name)
        batch = '16'
        num_epochs = project_parameters.get('epochs')
        clients.busy['mode'] = 'train' 
        opt = train.parse_opt()
        opt.data = data_dir
        opt.noautoanchor = True
        opt.cfg = model_dir
        opt.weights = weights
        opt.freeze = freeze
        opt.batch = batch
        opt.epochs = num_epochs
        opt.project = result_dir
        opt.name = 'exp'
        clients.busy['mode'] = 'train'
        print('clients train')
        train.main(opt)
        clients.busy['mode'] = None
        clients.busy['model_dir'] = None
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 0

        self.set_state('project_x')
        text = ['Обучение закончено:']
        keyboard = self.BOTTOM_DICT['check_train']
        keyboard = [Keyboards.reply_keyboard(keyboard)]
        return text, keyboard
    
    def load_of_weights_kn(self):
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        weights_path = os.path.join(proj_dir, 'train')
        exp = os.listdir(weights_path)
        exp.sort()
        exp.sort(key=len)
        if '.ipynb' in exp[-1]:
            _ = exp.pop()
        weights_path = os.path.join(weights_path, exp[-1], 'weights', 'best.pt')
        keyboard = self.BOTTOM_DICT['models_x']
        keyboard = Keyboards.reply_keyboard(keyboard)
        path_to_onnx_weights = self.export_weights_to_onnx(chat_id, kn=True)
        nef_model_path = self.compile_nef_model(path_to_onnx_weights)
        base_path = os.path.join('/root', 'yolov5_bot')
        os.chdir(base_path)
        return nef_model_path, keyboard
    
    def load_of_weights_khds(self):
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        weights_path = os.path.join(proj_dir, 'train')
        exp = os.listdir(weights_path)
        exp.sort()
        exp.sort(key=len)
        if '.ipynb' in exp[-1]:
            _ = exp.pop()
        weights_path = os.path.join(weights_path, exp[-1], 'weights', 'best.pt')
        keyboard = self.BOTTOM_DICT['models_x']
        keyboard = Keyboards.reply_keyboard(keyboard)
        path_to_onnx_weights = self.export_weights_to_onnx(chat_id, kn=False) # kn=False means khds=True
        path_to_onnx_weights = path_to_onnx_weights[:-3] + '.onnx'
        path_to_dataset = os.path.join(proj_dir, 'dataset', 'train', 'images')
        if os.path.isdir(path_to_dataset):
            input_img = path_to_dataset
        else:
            input_img = '/root/coco_calib'

        tmfile_model_dir = '/root/yolov5_bot/tm_model/'
        tmfile_model_path = os.path.join(tmfile_model_dir, 'yolov5m.tmfile')
        for tmfile in os.listdir(tmfile_model_dir):
            path_to_tmfile = os.path.join(tmfile_model_dir, tmfile)
            try:
                os.remove(path_to_tmfile)
            except:
                pass
        args = ('/root/yolov5_bot/convert_tool/convert_tool',
                '-f', 'onnx',
                '-m', path_to_onnx_weights,
                '-o', tmfile_model_path)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        print('process', popen.returncode)

        uint8_tmfile_model_path = '/root/yolov5_bot/tm_model/yolov5m_uint8.tmfile'

        args_2 = ('/root/yolov5_bot/quant_tool/quant_tool_uint8',
                  '-m', tmfile_model_path,
                  '-i', input_img,
                  '-o', uint8_tmfile_model_path,
                  '-g', '3,352,352',
                  '-a', '0',
                  '-w', '0,0,0',
                  '-s', '0.003922,0.003922,0.003922',
                  '-c', '0',
                  '-t', '4',
                  '-b', '1',
                  '-y', '352,352'
                 )
        popen2 = subprocess.Popen(args_2, stdout=subprocess.PIPE)
        popen2.wait()
        output = popen2.stdout.read()
        base_path = os.path.join('/root', 'yolov5_bot')
        os.chdir(base_path)
        keyboard = self.BOTTOM_DICT['models_x']
        keyboard = Keyboards.reply_keyboard(keyboard)
        return uint8_tmfile_model_path, keyboard
    
    def load_of_weights_rk(self):
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        weights_path = os.path.join(proj_dir, 'train')
        exp = os.listdir(weights_path)
        exp.sort()
        exp.sort(key=len)
        if '.ipynb' in exp[-1]:
            _ = exp.pop()
        weights_path = os.path.join(weights_path, exp[-1], 'weights', 'best.pt')
        path_to_onnx_weights = self.export_weights_to_onnx(chat_id, kn=False) # kn=False means rk3588=True
        path_to_onnx_weights = path_to_onnx_weights[:-3] + '.onnx'
        path_to_dataset = os.path.join(proj_dir, 'dataset', 'train', 'images')
        if os.path.isdir(path_to_dataset):
            input_img = path_to_dataset
        else:
            input_img = '/root/coco_calib'

        #path_to_onnx_weights =  path_to_onnx_weights
        print('path_to_onnx_weights', path_to_onnx_weights)
        print('input_img', input_img)
        command = '/root/yolov5_tg/tg_yolo/rknn_converter/yolo_convert.py '
        popen = subprocess.Popen('python3 '+ command + path_to_onnx_weights +' '+ input_img, executable='/bin/bash', shell=True)
        popen.wait()

        keyboard = self.BOTTOM_DICT['models_x']
        keyboard = Keyboards.reply_keyboard(keyboard)
        if popen.returncode == 0:
            return '/root/yolov5_tg/tg_yolo/rknn_converter/yolov5_quant.rknn', keyboard
        else:
            return '', keyboard

    def check_addres(self):
        self.set_state('addres_entry')
        text = 'Введите адрес устройства (IP).'
        chat_id = self.current_message.chat.id
        addres = clients.current_addres(chat_id)
        if addres is None:
            keyboard = [{'text':'Отмена'}]
            return text, keyboard
        else:
            keyboard = [{'text':f'{addres}'}]
            return text, keyboard
    
    def set_addres(self):
        chat_id = self.current_message.chat.id
        message_text = self.current_message.text
        if message_text.lower() == 'отмена':
            self.set_state('models_x')
            text = 'Обученая модель:'
            keyboard = self.BOTTOM_DICT['models_x']
            return text, keyboard
        else:
            self.set_state('weights_sender')
            text = 'Подготавливаются веса для отправки. Это может занять несколько минут.'
            keyboard = [{'text':'Назад'}]
            clients.set_addres(message_text, chat_id)
            return text, keyboard
    
    def send_weights_to_device(self):
        text = 'Готово'
        keyboard = self.BOTTOM_DICT['models_x']
        keyboard = Keyboards.reply_keyboard(keyboard)
        self.set_state('models_x')
        chat_id = self.current_message.chat.id
        ip = clients.current_addres(chat_id)
        if ip == '0.0.0.0': #Kneron
            model_path, _ = self.load_of_weights_kn()
        else : #Khadas
            model_path, _ = self.load_of_weights_khds()
        port = '8080'
        url = f'http://{ip}:{port}/model'
        files={'file': open(model_path,'rb')}
        try:
            r = requests.request('POST', url, files=files)
        except Exception as e:
            text = 'Не получилось. Проверьте адрес.'
            print('exception', e)
        return text, keyboard
    
    def export_weights_to_onnx(self, chat_id, kn):
        # export
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        result_dir = proj_dir + '/train'
        exp = os.listdir(result_dir)
        exp.sort()
        exp.sort(key=len)
        if '.ipynb' in exp[-1]:
            _ = exp.pop()
        weights = os.path.join(result_dir, exp[-1], 'weights', 'best.pt')
        weights_onnx = weights[:-3] + '.onnx'
        if os.path.isfile(weights_onnx):
            return weights
        #opt = export.parse_opt()
        #opt.weights = weights
        #opt.opset = 11
        savedPath = os.getcwd()
        os.chdir('/root/yolov5_tg/tg_yolo/yolov5/')
        command = '/root/yolov5_tg/tg_yolo/yolov5/export.py'
        #export.main(opt)
        params = f'--weights {weights} --include onnx --opset 11'
        print()
        print('python3 '+ command +' ' +  params)
        popen = subprocess.Popen('python3 '+ command +' ' +  params, executable='/bin/bash', shell=True)
        popen.wait()
        os.chdir(savedPath)
        
        return weights
    
    def compile_nef_model(self, path_to_weights):
        NAME = "best"
        IMG_SIZE = 640
        INPUT = [1, IMG_SIZE, IMG_SIZE, 3]

        MODEL_PATH_ONNX = path_to_weights[:-3] + '.onnx'
        SAVING_PATH_ONNX = path_to_weights[:-3] + '.opt.onnx'

        m = onnx.load(MODEL_PATH_ONNX)
        onnx.checker.check_model(m)
        print('\nonnx.helper.printable_graph(m.graph)')
        print(onnx.helper.printable_graph(m.graph))

        m = ktc.onnx_optimizer.torch_exported_onnx_flow(m, disable_fuse_bn=False)
        onnx.save(m,SAVING_PATH_ONNX)
        onnx.checker.check_model(m)
        print('\nonnx.helper.printable_graph(m.graph)')
        print(onnx.helper.printable_graph(m.graph))

        m = ktc.onnx_optimizer.onnx2onnx_flow(m, norm = True)
        onnx.save(m,SAVING_PATH_ONNX)
        onnx.checker.check_model(m)
        print(onnx.helper.printable_graph(m.graph))

        km = ktc.ModelConfig(211, "0001", "720", onnx_model=m)
        eval_result = km.evaluate()
        print("\nNpu performance evaluation result:\n" + str(eval_result))
        def letterbox_image(image, size):
            '''resize image with unchanged aspect ratio using padding'''
            iw, ih = image.size
            w, h = size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            return new_image

        def preprocess(pil_img):
            model_input_size = (IMG_SIZE, IMG_SIZE)  # to match our model input size when converting
            boxed_image = letterbox_image(pil_img, model_input_size)
            np_data = np.array(boxed_image, dtype='float32')

            np_data = np_data/255 - 0.5
            return np_data

        # load and normalize all image data from folder
        img_list = []
        n = 0
        model_path = MODEL_PATH_ONNX.split('/')[:-4]
        path_to_dataset = ''
        for el in model_path+['dataset', 'train', 'images']:
            path_to_dataset = os.path.join(path_to_dataset, el)
        if os.path.isdir(path_to_dataset):
            file_names = os.listdir(path_to_dataset)
        else:
            path_to_dataset = "/root/coco_calib"
            file_names = os.listdir(path_to_dataset)
        for f_n in file_names[:20]:
            fullpath = os.path.join(path_to_dataset, f_n)
            image = Image.open(fullpath)
            img_data = preprocess(image)
            img_list.append(img_data)

        # fix point analysis
        bie_model_path = km.analysis({"origin_input": img_list}, threads = 40, quantize_mode = "post_sigmoid" )
        print("\nFix point analysis done. Save bie model to '" + str(bie_model_path) + "'")
        # compile
        nef_model_path = ktc.compile([km])
        print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
        return nef_model_path
    
    def _get_list(self, data_name):
        text = 'Изображения: \n'
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        data_dir = os.path.join(path, 'projects', str(project_name), data_name)
        imgs = os.listdir(data_dir)
        labels = os.listdir(data_dir[:-6]+'labels')
        if len(imgs) == 0:
            text = 'Список пуст.'
        elif len(imgs) > 30:
            text += f'Всего загружено {len(imgs)} изображений\n'
            text += f'и {len(labels)} лэйблов\n'
        else:
            for img in imgs:
                text += img +', '
            text += '\nЛейблы: \n'
            for lbl in labels:
                text += lbl +', '
        return text, None
    
    def get_train_list(self):
        return self._get_list('dataset/train/images')

    def get_test_list(self):
        return self._get_list('dataset/valid/images')
        
    def _delete_list(self, data_name):
        text = ''
        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        data_dir = os.path.join(path, 'projects', str(project_name), data_name)
        for annot in os.listdir(data_dir):
            path_to_ann = os.path.join(data_dir, annot)
            try:
                os.remove(path_to_ann)
                text += annot + ',\n'
            except:
                print(f'failed remove {path_to_ann}')
        config_dir = os.path.join(path, 'projects', str(project_name), 'data')
        config_file = os.path.join(config_dir, 'custom.yaml')
        os.remove(config_file)
        text = 'Удалены ' + text
        keyboard = self.BOTTOM_DICT['delete_images']
        return text, keyboard
    
    def delete_train_list(self):
        return self._delete_list('dataset/train/images')
    
    def delete_test_list(self):
        return self._delete_list('dataset/valid/images')

import json
import PIL
import base64
import io

from messages_processor.messages import *


PATH = '/root/yolov5_tg/projects'


class Photo(Messages):
    def __init__(self):
        Messages.__init__(self)
        self.set_functions()
    
    documents_dict = {} 
    
    TEXT_DICT = {}
    
    BOTTOM_DICT = {}
    
    FUNCTION_DICT = {}
    
    DEMO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
    def set_functions(self):
        self.FUNCTION_DICT = {
                              'demo' : self.demo,
                              'train_annots' : self.save_train_annot,
                              'test_annots' : self.save_test_annot, 

                              'test_data' : self.custom,
                             }
    def save_train_annot(self):
        return self._save_annot('jsons_list/train_data')

    def save_test_annot(self):
        return self._save_annot('jsons_list/test_data')
    

    def _save_annot(self, data_name):
        text = 'Изображение загружено.'
        keyboard = [{'text':('Мои аннотации','Отмена')}]

        chat_id = self.current_message.chat.id
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        data_dir = os.path.join(path, 'projects', str(project_name), data_name)
        dataset_dir = os.path.join(path, 'projects', str(project_name), 'dataset')
        if self.current_message.content_type == 'photo':
            text = 'Файл не был загружен. Не нужно сжимать изображение.'
            keyboard = [{'text':('Мои аннотации','Отмена')}]
            return text, keyboard
        file_name = self.current_message.document.file_name
        path_to_save = os.path.join(data_dir, file_name)

        downloaded_file = self.documents_dict[chat_id]
        with open(path_to_save, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        if data_name == 'jsons_list/train_data':
            path_to_save_img = os.path.join(dataset_dir, 'train/images', file_name)
            path_to_labels = os.path.join(dataset_dir, 'train/labels', file_name)
        else:
            path_to_save_img = os.path.join(dataset_dir, 'valid/images', file_name)
            path_to_labels = os.path.join(dataset_dir, 'valid/labels', file_name)
        
        if 'json' in file_name:
            self.json_to_img(path_to_save, data_name, dataset_dir, file_name)
        elif 'txt' in file_name:
            with open(path_to_labels, 'wb') as new_file:
                new_file.write(downloaded_file)
        elif 'yaml' in file_name:
            with open(path_to_save, 'r') as downloaded_file:
                lines = downloaded_file.readlines()
            line = lines[-1]
            names = line[7:]
            names = names.replace("'", " ")
            names = names.replace("[", " ")
            names = names.replace("]", " ")
            names = names.replace(",", " ")
            labels_list = names.split()
            self.yaml_config(dataset_dir, labels_list)
        else:
            with open(path_to_save_img, 'wb') as new_file:
                new_file.write(downloaded_file)
        text = f'Сохранено {file_name}.'
        return text, keyboard
    
    def json_to_img(self, path_to_load, data_name, dataset_dir, file_name):
        img_file_name = file_name[:-5] + '.png'
        lbl_file_name = file_name[:-5] + '.txt'
        with open(path_to_load , "r") as read_file:
            load_json = json.load(read_file)
        if data_name == 'jsons_list/train_data':
            path_to_save = os.path.join(dataset_dir, 'train/images', img_file_name)
            path_to_labels = os.path.join(dataset_dir, 'train/labels', lbl_file_name)
        else:
            path_to_save = os.path.join(dataset_dir, 'valid/images', img_file_name)
            path_to_labels = os.path.join(dataset_dir, 'valid/labels', lbl_file_name)
        
        img_b64 = load_json['imageData']
        img_data = base64.b64decode(img_b64) 
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        try:
            img_pil.save(path_to_save)
        except:
            print('no save')
        self.json_to_labels(load_json, path_to_labels, dataset_dir)
    
    def json_to_labels(self, load_json, path_to_labels, dataset_dir):
        labels_set = set()
        for shape in load_json['shapes']:
            labels_set.add(shape['label'])
        labels_list = clients.get_labels(labels_set, dataset_dir)
        if load_json['shapes'][0].get('points'):
            labels_data = self.poligon_to_box(load_json, labels_list)
        with open(path_to_labels, 'w') as label_file:
            label_file.write(labels_data)
        self.yaml_config(dataset_dir, labels_list)
    
    def poligon_to_box(self, json, labels_list):
        labels_data = ''
        for shape in json['shapes']:
            min_x = min_y = 10000
            max_x = max_y = 0
            label_name = shape['label']
            label = labels_list.index(label_name)
            for point in shape['points']:
                min_x = min(min_x, point[0])
                max_x = max(max_x, point[0])
                min_y = min(min_y, point[1])
                max_y = max(max_y, point[1])
            width = max_x - min_x
            height = max_y - min_y
            x_c = (min_x + width/2)/json['imageWidth']
            y_c = (min_y + height/2)/json['imageHeight']
            labels_data += f'{label} {x_c} {y_c} {width/json["imageWidth"]} {height/json["imageHeight"]}\n'
        return labels_data
    
    def yaml_config(self, dataset_dir, labels_list):
        nc = len(labels_list)
        # Yaml creation
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
    
    def demo(self):
        texts = []
        keyboards = []
        if clients.busy['mode']:
            ETA = clients.estimate_time_left()    
            text = f'В данный момент нельзя запустить распознавание.\nБот освободится примерно через {ETA:0.1f} минут'
            return text, None
        clients.busy['mode'] = 'val' 
        clients.busy['model_dir'] = None 
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 1
        chat_id = self.current_message.chat.id
        downloaded_file = self.documents_dict[chat_id]
        
        stock_data_dir = os.path.join(str(chat_id), 'stock')
        if not os.path.isdir(stock_data_dir):
            os.makedirs(stock_data_dir)
        
        try:
            file_name = self.current_message.document.file_name
        except:
            file_name = self.current_message.file_name
            
        src_img_path = os.path.join(stock_data_dir, file_name)

        with open(src_img_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        opt = detect.parse_opt()
        opt.source = './'+src_img_path
        detect.main(opt)
        
        result_dir = 'runs/detect'
        exp = os.listdir(result_dir)
        exp.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)))
        
        dst_img_path = f'{result_dir}/{exp[-1]}/{file_name}'
        clients.busy['mode'] = None 
        clients.busy['model_dir'] = None 
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 0
        
        text = dst_img_path
        texts.append(text)
        keyboards.append([{'text':'Назад'}])
        return texts, keyboards
    
    def custom(self):
        texts = []
        keyboards = []
        if clients.busy['mode']:
            ETA = clients.estimate_time_left()    
            text = f'В данный момент нельзя запустить распознавание.\nБот освободится примерно через {ETA:0.1f} минут'
            return text, None
        
        clients.busy['mode'] = 'val' 
        clients.busy['model_dir'] = None 
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 1

        chat_id = self.current_message.chat.id
        downloaded_file = self.documents_dict[chat_id]
        
        stock_data_dir = os.path.join(str(chat_id), 'stock')
        if not os.path.isdir(stock_data_dir):
            os.makedirs(stock_data_dir)
        
        try:
            file_name = self.current_message.document.file_name
        except:
            file_name = self.current_message.file_name
            
        src_img_path = os.path.join(stock_data_dir, file_name)
            
        with open(src_img_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        project_name = clients.get_project_name(chat_id)
        path = os.path.join(PATH, str(chat_id))
        proj_dir = os.path.join(path, 'projects', project_name)
        result_dir = proj_dir + '/detect'
        data_dir = os.path.join(proj_dir, 'data', 'custom.yaml')
        opt = detect.parse_opt()
        weights_path = os.path.join(proj_dir, 'train')
        exp = os.listdir(weights_path)
        exp.sort()
        exp.sort(key=len)
        weights_path = os.path.join(weights_path, exp[-1], 'weights', 'best.pt')

        opt.weights = weights_path
        opt.data = data_dir
        opt.iou_thres=0.25
        opt.source = './'+src_img_path
        opt.project = result_dir
        opt.name = 'exp'
        detect.main(opt)

        exp = os.listdir(result_dir)
        exp.sort()
        exp.sort(key=len)
        dst_img_path = os.path.join(result_dir, exp[-1], file_name)
        
        clients.busy['mode'] = None 
        clients.busy['model_dir'] = None 
        clients.busy['epochs'] = 0
        clients.busy['iters'] = 0
        text = dst_img_path
        texts.append(text)
        keyboards.append([{'text':'Назад'}])
        return texts, keyboards

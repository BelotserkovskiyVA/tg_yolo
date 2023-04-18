# Yolov5 telegram bot
# Install instruction
## 1. Install the Telegram-YOLO
```
cd /root/yolov5_tg/
git clone https://github.com/BelotserkovskiyVA/tg_yolo.git
```
## 2. Install ultralytics/yolov5
```
cd tg_yolo/
pip install r requirements.txt
git clone https://github.com/ultralytics/yolov5.git
```
## 3. set a BOT_TOKEN into config.py

```
BOT_TOKEN = '12345:abcde' #Your_token
```
## 4. make changes to programs:  
train_replace.py -> train.py;
export_replace.py -> export.py;
yolo_replace.py -> yolo.py;
```
cd yolov5/
pip install r requirements.txt
```
## 5. Start bot
```
cd ../
python3 main.py
```


# Yolov5 telegram bot
# Install instruction
## 1. Install the Telegram-YOLO
```
cd /root/yolov5_tg/
git clone https://github.com/BelotserkovskiyVA/tg_yolo.git
cd tg_yolo/
pip install -r requirements.txt
```
## 2. set a BOT_TOKEN into config.py

```
BOT_TOKEN = '12345:abcde' #Your_token
```
## 3. Install ultralytics/yolov5
```
git clone --depth 1 --branch v6.0 https://github.com/ultralytics/yolov5.git
```
## 4. make changes to programs:  
```
cp yolov5m_leaky.pt yolov5/yolov5m_leaky.pt
cd yolov5/
pip install -r requirements.txt
```
train_replace.py -> train.py;
export_replace.py -> export.py;
yolo_replace.py -> yolo.py;
## 5. Start bot
```
cd ../
python3 main.py
```


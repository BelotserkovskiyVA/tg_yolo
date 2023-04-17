import os
import sys
libs = ['/root/rknn_converter', 
        '/usr/lib/python38.zip', 
        '/usr/lib/python3.8', 
        '/usr/lib/python3.8/lib-dynload', 
        '/rknpu2_env/lib/python3.8/site-packages', 
        '/base']
sys.path.extend(libs)
import argparse
from rknn.api import RKNN

#print(sys.path)

RKNN_MODEL = '/root/yolov5_tg/tg_yolo/rknn_converter/yolov5_quant.rknn'
#DATASET = './yolo_dataset/dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

def main(ONNX_MODEL='../yolov5_bot/best.onnx', DATASET='/root/yolov5_tg/coco_calib/'):
    if os.path.isdir(DATASET):
        imgs_list = [f for f in os.listdir(DATASET) if os.path.isfile(os.path.join(DATASET, f))]
        file_name = '/root/yolov5_tg/tg_yolo/rknn_converter/dataset.txt'
        with open(file_name, 'w') as f:
            for line in imgs_list:
                img_path = os.path.join(DATASET, line)
                f.writelines(img_path+'\n')
        DATASET = '/root/yolov5_tg/tg_yolo/rknn_converter/dataset.txt'
    else:
        print('dataset not found; try by coco_calib')
        DATASET = '/root/yolov5_tg/tg_yolo/rknn_converter/coco_dataset.txt'
    
    if os.path.isfile(ONNX_MODEL) is False:
        print('onnx model not found', ONNX_MODEL)
        return 0
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    rknn.release()
    #getcwd = os.path.abspath(os.getcwd())
    #rknn_path = os.path.join(getcwd, RKNN_MODEL)
    return RKNN_MODEL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path to ONNX model')
    parser.add_argument('ONNX_MODEL', nargs='?', type=str, default='../yolov5_bot/best.onnx',
                    help='path to ONNX model')
    parser.add_argument('DATASET', nargs='?', type=str, default='../coco_calib/',
                    help='path to dataset')
    opt = parser.parse_args()
    main(**vars(opt))


import cv2
import time
import sys
import numpy as np

def build_model(is_cuda):
    net = cv2.dnn.readNet("best.onnx") # OpenCV 로 yolo 모델을 쓰기 위해서 best.pt 파일을 export.py를 통해서 onnx 로 변환 후 함수를 이용하여 적용
    if is_cuda: # CPU or GPU 실행 확인
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# 기본 셋팅 설정, Yolo Input Image 크기 640 x 640
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.8

def detect(image, net):
    """이미지를 모델에 넣어 예측하는 함수"""
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False) # 모델에 집어넣기 위해 이미지 전처리
    net.setInput(blob) # 모델에 이미지 전달
    preds = net.forward() # 입력 x 로부터 예측된 y를 얻는 forward 연산 진행 후 저장
    return preds

def load_classes():
    """클래스를 불러오는 함수"""
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

def wrap_detection(input_image, output_data):
    """예측 전 이미지와 예측 후 이미지를 가지고 결과 정리하는 함수"""
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    # 입력 이미지와 Yolo 모델의 이미지 비율 계산
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            # 화면에 파악된 클래스 구하기
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                # 바운딩 박스를 위해서 좌표 계산
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    # NMS 알고리즘으로 정확한 바운딩 박스만 걸러내기
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    """이미지를 Yolo에 넣기 위해 정사각형 이미지로 만들어주는 함수"""

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result
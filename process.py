import cv2
import time
import sys
from model import *
from multiprocessing import Process, Value, Array, Lock
from voice import *
import openai
import time, os
import speech_recognition as sr
from gtts import gTTS
import playsound


def func_camera(classids, flag, lock):
    """영상처리 담당하는 함수"""

    # 저장된 클래스 불러오기
    class_list = load_classes()
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    
    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    
    # 모델 적용
    net = build_model(is_cuda)
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        
        if ret is False:
            print("End of stream")
            break
        
        # 이미지 전처리를 거치고 예측 후 식별된 클래스와 신뢰도, 박스 받아오기
        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)    
        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])    
        
        # 바운딩 박스를 그리고 flag 값과 classids 값을 수정하기
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            
            
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
            # shared memory 로 인한 메모리 선점 문제를 위해 Lock
            lock.acquire()
            flag.value = 1
            classids[classid] = 1
            lock.release()
            
        cv2.imshow("output", frame)
            
        
        if cv2.waitKey(1) == ord('q'):
            break
            
            
    cap.release()
    cv2.destroyAllWindows()
    
def func_voice(classids, flag, lock):
    """음성 담당하는 함수"""
    
    r = sr.Recognizer()
    m = sr.Microphone()
    
    # 절전 상태로 듣다가 들리면 반응
    r.listen_in_background(m, speech_to_text)
    
    # 클래스 불러오기
    class_list = load_classes()
    
    while True:
        if flag.value == 1:
            for i in range(5):
                if classids[i] == 1:
                    text_to_speech1(f"앞에 {class_list[i]}가 있습니다")
            
            # 마찬가지로 메모리 선점 문제 해결
            lock.acquire()
            flag.value = 0
            lock.release()
        time.sleep(0.5)


if __name__ == '__main__':

    # shared memory 사용을 위해 변수 설정
    lock = Lock()
    shared_classid = Array("i", [0, 0, 0, 0, 0])
    shared_flag = Value("i", 0)

    # 병렬처리로 영상처리와 음성인식 동시 실행
    p1 = Process(target=func_camera, args=(shared_classid, shared_flag, lock))
    p2 = Process(target=func_voice, args=(shared_classid, shared_flag, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.terminate()
    p2.join()
print("webcam_infer.py 실행행")

import cv2
import torch
import time
from model import SimpleCNN  # 최적화된 모델 import
import serial  # 시리얼 통신 라이브러리 추가

print("기본 라이브러리 import 완료")

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# 모델 불러오기
print("모델 로딩 중...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("모델 로딩 완료")

# 모델 최적화 (추론 속도 향상)
print("모델 최적화 중...")
if device.type == 'cpu':
    model = torch.jit.script(model)  # TorchScript로 최적화
else:
    model = torch.jit.trace(model, torch.randn(1, 3, 160, 160).to(device))
print("모델 최적화 완료")

# 클래스 이름 정의
class_names = ['non_wheelchair', 'wheelchair']

# 이미지 전처리 (더 작은 크기로 조정)
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),  # 224x224에서 160x160으로 축소
    transforms.ToTensor(),
])

print("웹캠 초기화 중...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('카메라를 열 수 없습니다.')
    exit()

# 카메라 해상도 설정 (더 낮은 해상도로 설정)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 성능 측정 변수
frame_count = 0
fps_start_time = time.time()
inference_time = 0
last_connection_time = time.time()  # 마지막 연결 상태 출력 시간
fps = 0.0  # fps 초기화

# 아두이노 시리얼 포트 설정
try:
    arduino = serial.Serial('COM6', 9600, timeout=1) 
    print('아두이노 연결 성공')
except Exception as e:
    print('아두이노 연결 실패:', e)
    arduino = None

print('웹캠 추론을 시작합니다. ESC 키를 누르면 종료됩니다.')

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임을 읽을 수 없습니다.')
        break
    
    # 웹캠 연결 상태 출력
    current_time = time.time()
    if current_time - last_connection_time >= 4.0:
        print('웹캠이 연결되어 있습니다.')
        last_connection_time = current_time
    
    # 매 3프레임마다 추론 실행
    if frame_count % 3 == 0:
        # 프레임 크기 조정
        small_frame = cv2.resize(frame, (160, 160))
        img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 추론 시간 측정
        start_time = time.time()
        
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(1).item()
        
        inference_time = time.time() - start_time
        label = class_names[pred]
        confidence = torch.softmax(output, dim=1).max().item()
        #아두이노에 신호 전송
        if label == 'wheelchair' and arduino:
            try:
                arduino.write(b'1')  # 모터 ON
                print('아두이노: 모터 ON')
            except Exception as e:
                print('아두이노 신호 전송 오류:', e)
        elif label == 'non_wheelchair' and arduino:
            try:
                arduino.write(b'0')  # 모터 OFF
                print('아두이노: 모터 OFF')
            except Exception as e:
                print('아두이노 신호 전송 오류:', e)
    
    # FPS 계산
    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()
    
    # 화면에 정보 표시
    cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Probability: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Inference Time: {inference_time*1000:.1f}ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
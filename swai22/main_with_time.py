import numpy as np
import torch
import torch.nn as nn
from sensor import get_sensor, set_motor, display_info
from camera import capture_and_classify
from datetime import datetime
import time

class frostClassifier(nn.Module):
    def __init__(self, input_size):
        super(frostClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def load_model(model_path, input_size):
    model = frostClassifier(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_frost(input_data, model):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data)
        prediction = model(input_tensor)
        result = (prediction >= 0.65).float().item()
        return bool(result)

def is_night_time():
    """
    현재 시간이 새벽 시간(0시~5시)인지 확인하는 함수
    """
    current_hour = datetime.now().hour
    return 0 <= current_hour <= 5

def get_current_time_info():
    """
    현재 시간 정보를 반환하는 함수
    """
    now = datetime.now()
    return {
        'hour': now.hour,
        'minute': now.minute,
        'second': now.second,
        'time_string': now.strftime("%H:%M:%S"),
        'date_string': now.strftime("%Y-%m-%d")
    }

def display_time_status():
    """
    현재 시간 상태를 표시하는 함수
    """
    time_info = get_current_time_info()
    print(f"현재 시간: {time_info['time_string']} ({time_info['date_string']})")
    
    if is_night_time():
        print("🌙 새벽 시간 - 서리 감지 시스템 활성화")
        return True
    else:
        print("☀️ 주간 시간 - 서리 감지 시스템 비활성화")
        return False

input_size = 3  
model = load_model('frost_classifier_model.pth', input_size)

print("=== 서리 감지 시스템 (시간 제한 버전) ===")
print("시스템이 새벽 시간(0시~5시)에만 작동합니다.")
print("=" * 50)

while True:
    try:
        # 현재 시간 상태 확인
        is_active = display_time_status()
        
        if is_active:
            # 새벽 시간일 때만 서리 감지 수행
            print("\n🔍 서리 감지 중...")
            
            # 카메라로 첫 번째 게이트 확인
            first_gate = capture_and_classify()
            print(f"카메라 감지 결과: {first_gate}")
            
            # 센서 데이터 가져오기
            sensor_data = get_sensor()
            print(f"센서 데이터: 온도={sensor_data[0]:.1f}°C, 습도={sensor_data[1]:.1f}%, 풍속={sensor_data[2]:.1f}m/s")
            
            # 서리 예측
            frost_detected = predict_frost(sensor_data, model)
            print(f"서리 예측 결과: {'발생' if frost_detected else '미발생'}")
            
            # 조건 확인 및 동작 수행
            if first_gate >= 3 and frost_detected:
                print("❄️ 서리 발생 감지! 스프링클러 가동...")
                set_motor()
            elif not frost_detected or first_gate >= 3:
                print("✅ 정상 상태 - 정보 표시")
                display_info()
            else:
                print("⏸️ 대기 상태")
        else:
            # 주간 시간일 때는 대기 메시지 표시
            print("💤 주간 시간 - 시스템 대기 중...")
            print("새벽 시간(0시~5시)까지 대기합니다.")
            
            # 다음 새벽 시간까지 남은 시간 계산
            current_hour = get_current_time_info()['hour']
            if current_hour > 5:
                hours_until_night = 24 - current_hour
            else:
                hours_until_night = 0 - current_hour
                
            if hours_until_night > 0:
                print(f"⏰ 다음 활성화까지 약 {hours_until_night}시간 남음")
        
        print("-" * 50)
        
        # 10초 대기 (실제 시스템에서는 더 짧게 설정 가능)
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n🛑 시스템 종료 요청됨")
        break
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("5초 후 재시도...")
        time.sleep(5)

print("시스템이 종료되었습니다.") 
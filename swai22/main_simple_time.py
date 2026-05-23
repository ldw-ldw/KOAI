import numpy as np
import torch
import torch.nn as nn
from sensor import get_sensor, set_motor, display_info
from camera import capture_and_classify
from datetime import datetime

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

input_size = 3  
model = load_model('frost_classifier_model.pth', input_size)

while True:
    # 새벽 시간(0시~5시)에만 작동
    current_hour = datetime.now().hour
    if 0 <= current_hour <= 5:
        first_gate = capture_and_classify()
        
        if first_gate >= 3 and predict_frost(get_sensor(), model):
            set_motor()
        elif not predict_frost(get_sensor(), model) or first_gate >= 3:
            display_info() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('frost_dataset.csv')
print("데이터셋 정보:")
print(df.info())
print("\n데이터셋 통계:")
print(df.describe())

# 데이터 분리
X = df.iloc[:, :-1].values  # 최저 기온, 평균 풍속, 최소 상대 습도
y = df.iloc[:, -1].values   # 서리 발생 여부 (0: 없음, 1: 있음)

# 모델 클래스 정의 (main.py와 동일)
class frostClassifier(nn.Module):
    def __init__(self, input_size):
        super(frostClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# 저장된 모델 로드
def load_model(model_path, input_size):
    model = frostClassifier(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 1. 3D 산점도로 데이터 분포 시각화
def plot_3d_scatter():
    fig = plt.figure(figsize=(15, 5))
    
    # 서리 발생 데이터 (label=1)
    frost_data = X[y == 1]
    # 서리 미발생 데이터 (label=0)
    no_frost_data = X[y == 0]
    
    # 3D 산점도
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(frost_data[:, 0], frost_data[:, 1], frost_data[:, 2], 
                c='red', marker='o', label='서리 발생', alpha=0.7, s=50)
    ax1.scatter(no_frost_data[:, 0], no_frost_data[:, 1], no_frost_data[:, 2], 
                c='blue', marker='^', label='서리 미발생', alpha=0.7, s=50)
    ax1.set_xlabel('최저 기온 (°C)')
    ax1.set_ylabel('평균 풍속 (m/s)')
    ax1.set_zlabel('최소 상대 습도 (%)')
    ax1.set_title('3D 데이터 분포')
    ax1.legend()
    
    # 2D 투영 (기온 vs 풍속)
    ax2 = fig.add_subplot(132)
    ax2.scatter(frost_data[:, 0], frost_data[:, 1], c='red', marker='o', 
                label='서리 발생', alpha=0.7, s=50)
    ax2.scatter(no_frost_data[:, 0], no_frost_data[:, 1], c='blue', marker='^', 
                label='서리 미발생', alpha=0.7, s=50)
    ax2.set_xlabel('최저 기온 (°C)')
    ax2.set_ylabel('평균 풍속 (m/s)')
    ax2.set_title('기온 vs 풍속')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2D 투영 (기온 vs 습도)
    ax3 = fig.add_subplot(133)
    ax3.scatter(frost_data[:, 0], frost_data[:, 2], c='red', marker='o', 
                label='서리 발생', alpha=0.7, s=50)
    ax3.scatter(no_frost_data[:, 0], no_frost_data[:, 2], c='blue', marker='^', 
                label='서리 미발생', alpha=0.7, s=50)
    ax3.set_xlabel('최저 기온 (°C)')
    ax3.set_ylabel('최소 상대 습도 (%)')
    ax3.set_title('기온 vs 습도')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3d_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. 각 특성별 분포 히스토그램
def plot_feature_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    feature_names = ['최저 기온 (°C)', '평균 풍속 (m/s)', '최소 상대 습도 (%)']
    
    for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
        # 서리 발생 데이터
        ax.hist(X[y == 1, i], bins=15, alpha=0.7, color='red', 
                label='서리 발생', density=True)
        # 서리 미발생 데이터
        ax.hist(X[y == 0, i], bins=15, alpha=0.7, color='blue', 
                label='서리 미발생', density=True)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('밀도')
        ax.set_title(f'{feature_name} 분포')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 모델 예측 결과 시각화
def plot_model_predictions():
    # 모델 로드
    model = load_model('frost_classifier_model.pth', X.shape[1])
    
    # 예측 수행
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy().flatten()
        predicted_labels = (predictions >= 0.65).astype(int)  # main.py와 동일한 임계값
    
    fig = plt.figure(figsize=(15, 5))
    
    # 실제 vs 예측 비교
    ax1 = fig.add_subplot(131)
    correct = y == predicted_labels
    incorrect = y != predicted_labels
    
    # 정확한 예측
    ax1.scatter(X[correct, 0], X[correct, 1], c='green', marker='o', 
                label='정확한 예측', alpha=0.7, s=50)
    # 잘못된 예측
    ax1.scatter(X[incorrect, 0], X[incorrect, 1], c='red', marker='x', 
                label='잘못된 예측', alpha=0.7, s=100)
    
    ax1.set_xlabel('최저 기온 (°C)')
    ax1.set_ylabel('평균 풍속 (m/s)')
    ax1.set_title('모델 예측 결과')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 예측 확률 분포
    ax2 = fig.add_subplot(132)
    ax2.hist(predictions[y == 1], bins=20, alpha=0.7, color='red', 
             label='실제 서리 발생', density=True)
    ax2.hist(predictions[y == 0], bins=20, alpha=0.7, color='blue', 
             label='실제 서리 미발생', density=True)
    ax2.axvline(x=0.65, color='black', linestyle='--', label='임계값 (0.65)')
    ax2.set_xlabel('예측 확률')
    ax2.set_ylabel('밀도')
    ax2.set_title('예측 확률 분포')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 혼동 행렬
    ax3 = fig.add_subplot(133)
    cm = confusion_matrix(y, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['서리 미발생', '서리 발생'],
                yticklabels=['서리 미발생', '서리 발생'], ax=ax3)
    ax3.set_title('혼동 행렬')
    ax3.set_xlabel('예측')
    ax3.set_ylabel('실제')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 분류 리포트 출력
    print("\n분류 리포트:")
    print(classification_report(y, predicted_labels, 
                               target_names=['서리 미발생', '서리 발생']))

# 4. 결정 경계 시각화 (2D 투영)
def plot_decision_boundary():
    # 모델 로드
    model = load_model('frost_classifier_model.pth', X.shape[1])
    
    # 기온과 풍속으로 결정 경계 그리기
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 습도는 평균값으로 고정
    mean_humidity = np.mean(X[:, 2])
    
    # 격자점에서 예측
    grid_points = np.c_[xx.ravel(), yy.ravel(), 
                       np.full(xx.ravel().shape, mean_humidity)]
    
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_points)
        predictions = model(grid_tensor).numpy().flatten()
        predictions = predictions.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    # 결정 경계
    plt.contourf(xx, yy, predictions, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='서리 발생 확률')
    
    # 데이터 포인트
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o', 
                label='서리 발생', s=50, alpha=0.8)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='^', 
                label='서리 미발생', s=50, alpha=0.8)
    
    # 임계값 경계선
    plt.contour(xx, yy, predictions, levels=[0.65], colors='black', 
                linewidths=2, linestyles='--', label='임계값 (0.65)')
    
    plt.xlabel('최저 기온 (°C)')
    plt.ylabel('평균 풍속 (m/s)')
    plt.title('결정 경계 (습도 = 평균값)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. 모델 성능 지표
def print_model_performance():
    model = load_model('frost_classifier_model.pth', X.shape[1])
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy().flatten()
        predicted_labels = (predictions >= 0.65).astype(int)
    
    accuracy = np.mean(y == predicted_labels)
    precision = np.sum((y == 1) & (predicted_labels == 1)) / np.sum(predicted_labels == 1)
    recall = np.sum((y == 1) & (predicted_labels == 1)) / np.sum(y == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print("\n=== 모델 성능 지표 ===")
    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1 점수: {f1_score:.4f}")
    
    # 클래스별 데이터 수
    print(f"\n=== 데이터 분포 ===")
    print(f"서리 발생 데이터: {np.sum(y == 1)}개")
    print(f"서리 미발생 데이터: {np.sum(y == 0)}개")
    print(f"총 데이터: {len(y)}개")

if __name__ == "__main__":
    print("=== 서리 분류 모델 시각화 ===\n")
    
    # 1. 3D 데이터 분포 시각화
    print("1. 3D 데이터 분포 시각화 중...")
    plot_3d_scatter()
    
    # 2. 특성별 분포 시각화
    print("\n2. 특성별 분포 시각화 중...")
    plot_feature_distributions()
    
    # 3. 모델 예측 결과 시각화
    print("\n3. 모델 예측 결과 시각화 중...")
    plot_model_predictions()
    
    # 4. 결정 경계 시각화
    print("\n4. 결정 경계 시각화 중...")
    plot_decision_boundary()
    
    # 5. 모델 성능 지표 출력
    print("\n5. 모델 성능 지표 계산 중...")
    print_model_performance()
    
    print("\n=== 시각화 완료 ===")
    print("생성된 이미지 파일들:")
    print("- 3d_data_distribution.png")
    print("- feature_distributions.png") 
    print("- model_predictions.png")
    print("- decision_boundary.png") 
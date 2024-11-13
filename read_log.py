import os
import json
import matplotlib.pyplot as plt

# log 파일 경로
# log_dir = r'F:\YOON\research\detr_pose_jaxa\output\random_30000_epcho_300_noanytransform\20241029-184111'
# log_dir = r'F:\YOON\research\detr_pose_jaxa\output\old\random_30000_350epochs_posecoefficient5'
log_dir = r'F:\YOON\research\detr_pose_jaxa\output\20241101-234202'
log_path = os.path.join(log_dir, 'log.txt')

# log 파일 읽기
with open(log_path, 'r') as f:
    log_data = f.readlines()

# 각 줄마다 json 형식의 로그를 파싱하여 리스트로 저장
logs = [json.loads(line) for line in log_data]

# 관심 있는 키들만 남기기 위한 필터 설정
relevant_keys = [
                #  'train_loss', 
                 'train_loss_bbox', 
                #  'train_loss_ce', 
                 'train_loss_poses',
                #  'test_loss',
                 'test_loss_bbox', 
                #  'test_loss_ce', 
                 'test_loss_poses'
                 ]

# train과 test 관련 키를 저장할 딕셔너리 생성
train_logs = {}
test_logs = {}

# 로그 데이터에서 필터링된 키들 추출
for log_dict in logs:
    for key, value in log_dict.items():
        if key in relevant_keys:
            if 'train' in key:
                if key not in train_logs:
                    train_logs[key] = []
                train_logs[key].append(value)
            elif 'test' in key:
                if key not in test_logs:
                    test_logs[key] = []
                test_logs[key].append(value)

# 그래프 그리기
plt.figure(figsize=(15, 10))

# Train losses and metrics
plt.subplot(2, 1, 1)
for key, values in train_logs.items():
    plt.plot(values, label=key, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Train Losses and Metrics per Epoch')
plt.legend()
plt.grid(True)

# Test losses and metrics
plt.subplot(2, 1, 2)
for key, values in test_logs.items():
    plt.plot(values, label=key, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Test Losses and Metrics per Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


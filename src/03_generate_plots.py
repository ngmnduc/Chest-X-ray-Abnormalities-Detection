import os
import yaml
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(PROJECT_ROOT, 'data', 'yolo_dataset')
YAML_PATH = os.path.join(YOLO_DIR, 'dataset.yaml')

# 1. Tự động tạo file dataset.yaml nếu nó chưa tồn tại trên máy tính
if not os.path.exists(YAML_PATH):
    dataset_info = {
        'path': YOLO_DIR,          # Trỏ đường dẫn tuyệt đối tới thư mục data trên máy bạn
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['Abnormality']
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    print(f"Create file at: {YAML_PATH}")

# 2. Load model
MODEL_PATH = os.path.join(PROJECT_ROOT, 'weights', 'best.pt')
model = YOLO(MODEL_PATH)

# 3. Chạy validation để sinh ảnh plots
print("Start Evaluation...")
metrics = model.val(
    data=YAML_PATH, 
    split='val',
    plots=True,          
    save_json=False,
    project=os.path.join(PROJECT_ROOT, 'runs', 'detect'),
    name='v3_reconstructed_metrics'
)
print("Done! Go to runs/v3_reconstructed_metrics to take the picture.")
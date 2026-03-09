import os
import pandas as pd
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMG_DIR = os.path.join(PROJECT_ROOT, 'data', 'original_images')
YOLO_DIR = os.path.join(PROJECT_ROOT, 'data', 'yolo_dataset')

if os.path.exists(YOLO_DIR):
    shutil.rmtree(YOLO_DIR) 

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

print("Reading and merging CSV data...")
train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
meta_df = pd.read_csv(os.path.join(RAW_DIR, 'train_meta.csv'))
df = train_df.merge(meta_df, on='image_id', how='left')

print("Filtering consensus and fusing bounding boxes...")

# Liệt kê logic sửa đổi:
# 1. Thêm Center Distance để xử lý đặc thù ranh giới mờ của X-quang.
# 2. Bổ sung Validation để chặn các box có width hoặc height < 5px.
def consensus_fusion(boxes, iou_thresh=0.3, min_voters=2):
    if not boxes: return []
    keep_boxes = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        cluster = [boxes[i]]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]: continue
            b1, b2 = boxes[i], boxes[j]
            
            # Tính IoU (Intersection over Union)
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            
            # Tính Center Distance
            cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
            cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
            center_dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
            avg_size = ((b1[2] - b1[0]) + (b2[2] - b2[0]) + (b1[3] - b1[1]) + (b2[3] - b2[1])) / 4
            
            # Điều kiện gom nhóm
            if iou > iou_thresh or center_dist < avg_size * 0.2:
                cluster.append(boxes[j])
                used[j] = True
                
        if len(cluster) >= min_voters:
            avg_box = [sum(c)/len(cluster) for c in zip(*cluster)]
            # Validation: Chặn box dị dạng
            if (avg_box[2] - avg_box[0] >= 5) and (avg_box[3] - avg_box[1] >= 5):
                keep_boxes.append(avg_box)
    return keep_boxes

abnormal_raw = df[df['class_id'] != 14].copy()
fused_data = []

# Liệt kê logic sửa đổi: Groupby theo class_id trước khi fusion
for image_id, group in tqdm(abnormal_raw.groupby('image_id')):
    dim0 = group['dim0'].iloc[0]
    dim1 = group['dim1'].iloc[0]
    
    for class_id, class_group in group.groupby('class_id'):
        boxes = class_group[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        fused_boxes = consensus_fusion(boxes, iou_thresh=0.3, min_voters=2)
        
        for fb in fused_boxes:
            fused_data.append({
                'image_id': image_id,
                'x_min': fb[0], 'y_min': fb[1],
                'x_max': fb[2], 'y_max': fb[3],
                'dim0': dim0, 'dim1': dim1,
                'yolo_class': 0  # Đưa về bài toán 1 class (Abnormality)
            })

abnormal_df = pd.DataFrame(fused_data)

valid_abnormal_imgs = abnormal_df['image_id'].unique().tolist() if not abnormal_df.empty else []
normal_imgs = [img for img in df['image_id'].unique() if img not in valid_abnormal_imgs]

print(f"\nOriginal - Normal: {len(normal_imgs)}, Abnormal: {len(valid_abnormal_imgs)}")
random.seed(42)
normal_sampled = random.sample(normal_imgs, min(len(valid_abnormal_imgs), len(normal_imgs)))
unique_images = valid_abnormal_imgs + normal_sampled
print(f"Balanced - Normal: {len(normal_sampled)}, Abnormal: {len(valid_abnormal_imgs)}, Total: {len(unique_images)}")

train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

def get_split(img_id):
    if img_id in train_imgs: return 'train'
    elif img_id in val_imgs: return 'val'
    else: return 'test'

print("Normalizing YOLO coordinates...")
if not abnormal_df.empty:
    abnormal_df['box_w'] = abnormal_df['x_max'] - abnormal_df['x_min']
    abnormal_df['box_h'] = abnormal_df['y_max'] - abnormal_df['y_min']
    abnormal_df['x_center'] = ((abnormal_df['x_min'] + (abnormal_df['box_w'] / 2)) / abnormal_df['dim1']).clip(0, 1)
    abnormal_df['y_center'] = ((abnormal_df['y_min'] + (abnormal_df['box_h'] / 2)) / abnormal_df['dim0']).clip(0, 1)
    abnormal_df['w_norm'] = (abnormal_df['box_w'] / abnormal_df['dim1']).clip(0, 1)
    abnormal_df['h_norm'] = (abnormal_df['box_h'] / abnormal_df['dim0']).clip(0, 1)
    grouped = abnormal_df.groupby('image_id')
else:
    grouped = {}

print("Copying Images and Labels (This will take a few minutes)...")
for image_id in tqdm(unique_images):
    split = get_split(image_id)
    img_src = os.path.join(IMG_DIR, f"{image_id}.png")
    img_dst = os.path.join(YOLO_DIR, 'images', split, f"{image_id}.png")
    txt_dst = os.path.join(YOLO_DIR, 'labels', split, f"{image_id}.txt")
    
    if os.path.exists(img_src):
        shutil.copy(img_src, img_dst) 
        
        if image_id in grouped.groups if isinstance(grouped, pd.core.groupby.DataFrameGroupBy) else False:
            boxes = grouped.get_group(image_id)[['yolo_class', 'x_center', 'y_center', 'w_norm', 'h_norm']]
            boxes.to_csv(txt_dst, sep=' ', header=False, index=False)
        else:
            open(txt_dst, 'w').close()

print("DONE PREPARING DATA!")
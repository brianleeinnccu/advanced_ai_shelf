import os
import shutil
import yaml
from ultralytics import YOLO

# 1. 原始檔案與模型路徑 (保持不變)
source_image_dir = "D:\\USER\\Desktop\\clear_cropped_shelfs"
model_path = "D:\\USER\\Desktop\\product_train_last3085\\runs\\detect\\shelf_finetune_v26_v1_add_2000_yolo11n_600_advanced_finetune_with_100\\weights\\best.pt"

# 2. 設定匯出資料夾與新的目錄結構
cvat_export_dir = "D:\\USER\\Desktop\\CVAT_YOLO_Dataset"

# 建立 images/train 和 labels/train 資料夾
images_train_dir = os.path.join(cvat_export_dir, "images", "train")
labels_train_dir = os.path.join(cvat_export_dir, "labels", "train")

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)

# 3. 載入模型並獲取類別名稱
print("正在載入 YOLO 模型...")
model = YOLO(model_path)
class_names = model.names # YOLO 模型內建的類別字典，例如 {0: '類別A', 1: '類別B'}

# 4. 生成 data.yaml (使用你提供的格式)
data_yaml_path = os.path.join(cvat_export_dir, "data.yaml")
data_yaml = {
    "path": ".",
    "train": "images/train",
    "val": "images/train",  # CVAT 匯入通常沒差，可以指向同一個
    "names": class_names
}

# 將字典寫入 yaml 檔案
with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
print(f"已生成設定檔: {data_yaml_path}")

# 5. 找出原始資料夾中的所有 JPG 檔案
image_files = [f for f in os.listdir(source_image_dir) if f.lower().endswith('.jpg')]
print(f"共找到 {len(image_files)} 張圖片，開始進行推理與建立資料集...")

# 6. 開始推理、複製圖片並寫入標註
for img_name in image_files:
    source_img_path = os.path.join(source_image_dir, img_name)
    target_img_path = os.path.join(images_train_dir, img_name)
    
    # 複製圖片到 images/train
    shutil.copy2(source_img_path, target_img_path)
    
    # 進行預測 (信心度 conf=0.25)
    results = model.predict(source=source_img_path, conf=0.7, verbose=False)
    
    # 在 labels/train 中寫入 YOLO 格式的 .txt 標註檔
    target_txt_path = os.path.join(labels_train_dir, os.path.splitext(img_name)[0] + '.txt')
    
    with open(target_txt_path, 'w', encoding='utf-8') as f_txt:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            x, y, w, h = box.xywhn[0].tolist()
            f_txt.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print(f"\n🎉 大功告成！請前往查看: {cvat_export_dir}")
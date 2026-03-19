import os
import random
import shutil
from pathlib import Path
import yaml

# --- 1. 路徑與範圍設定 ---
# 你的原始資料集路徑 (此資料夾將被視為唯讀，不寫入任何新檔案)
base_dir = Path("E:\\task_29_dataset_2026_03_17_09_50_22_ultralytics yolo detection 1.0")
img_dir = base_dir / "images" / "train"
label_dir = base_dir / "labels" / "train"

# 指定 CVAT 順序標註的起訖 ID (例如 0 ~ 200)
START_ID = 0
END_ID = 600

# 設定輸出目錄：在當前目錄下建立一個名為 "yolo_dataset" 的標準 YOLO 資料夾
output_dir = Path.cwd() / "yolo_dataset"

# 建立標準 YOLO 資料夾結構 (images 和 labels 下分別建立 train, val, test)
for split in ["train", "val", "test"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# --- 2. 取得並排序圖片 ---
# valid_extensions = (".jpg", ".png", ".jpeg")
# all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

valid_extensions = (".jpg", ".png", ".jpeg")
all_images = [
    f for f in os.listdir(img_dir) 
    if f.lower().endswith(valid_extensions) and not f.startswith("._")
]
# 【關鍵】按照字母/數字順序排列，確保與 CVAT 輸出的順序一致
all_images.sort()
print(all_images[:10])  # 印出前 10 張圖片名稱確認順序
# 只擷取我們有標註的範圍 (0 ~ 400，共 401 張)
labeled_images = all_images[START_ID : END_ID + 1]

# --- 3. 隨機打亂並切分比例 ---
random.seed(42)
random.shuffle(labeled_images)

total_labeled = len(labeled_images)
train_idx = int(total_labeled * 0.8)
val_idx = int(total_labeled * 0.9) # 80%:10%:10%

train_files = labeled_images[:train_idx]
val_files = labeled_images[train_idx:val_idx]
test_files = labeled_images[val_idx:]

print(f"Extraction Range: {START_ID} to {END_ID}")
print(f"Total labeled images used: {total_labeled}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")
print(f"Testing images: {len(test_files)}")

# --- 4. 複製圖片與標籤到對應的資料夾 ---
def copy_files(file_list, split_name):
    for file_name in file_list:
        img_src = img_dir / file_name
        label_src = label_dir / (Path(file_name).stem + ".txt")
        
        # 確保標籤檔存在，避免複製到未標註的空圖
        if label_src.exists():
            img_dst = output_dir / "images" / split_name / file_name
            label_dst = output_dir / "labels" / split_name / label_src.name
            
            # 複製檔案
            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)
        else:
            print(f"Warning: Label not found for {file_name}, skipping.")

print("Copying train files...")
copy_files(train_files, "train")
print("Copying validation files...")
copy_files(val_files, "val")
print("Copying test files...")
copy_files(test_files, "test")

# --- 5. 建立專屬的 data.yaml ---
def create_local_data_yaml():
    local_yaml_path = output_dir / "data.yaml"
    original_yaml_path = base_dir / "data.yaml"
    
    # 讀取原本的分類名稱 (如果原資料夾有的話)
    existing_names = {0: 'shelf'} 
    if original_yaml_path.exists():
        with open(original_yaml_path, "r", encoding="utf-8") as f:
            try:
                content = yaml.safe_load(f)
                if content and "names" in content:
                    existing_names = content["names"]
            except Exception as e:
                print(f"Reading original data.yaml failed: {e}")

    # 建立新的 yaml，這一次指定為新建立的 images 資料夾結構
    data_config = {
        "path": str(output_dir.resolve()), # 將根目錄指向新生成的資料夾
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": existing_names
    }

    with open(local_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    print(f"Created local data.yaml at {local_yaml_path}")

create_local_data_yaml()

print(f"✅ 成功！完整 YOLO 資料集已建立於：{output_dir}")
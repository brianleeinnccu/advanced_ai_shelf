from ultralytics import YOLO

def train_yolo():
    # 載入預訓練模型
    model = YOLO("D:\\USER\\Desktop\\product_train_last3085\\yolo11n.pt")

    # 開始訓練
    results = model.train(
        data = "D:\\USER\\Desktop\\product_train_last3085\\yolo_dataset\\data.yaml",  # 資料集配置文件
        epochs=100,            # 訓練輪數，微調通常 50-100 即可
        imgsz=640,             # 圖片輸入大小
        batch=16,              # Batch size
        name="shelf_finetune_v26_v1_add_2000_yolo11n_finetune_600_advanced", # 輸出結果資料夾名
        
        # 資料增強參數 (Augmentation)
        mosaic=1.0,           # Mosaic 增強 (1.0 代表開啟)
        mixup=0.1,            # Mixup 增強 (0.1 小幅度增加多樣性)
        fliplr=0.5,           # 左右翻轉 (50% 機率)
        degrees=10.0,         # 旋轉角度 (+/- 10度)
        hsv_h=0.015,          # HSV 色調調整
        hsv_s=0.7,            # HSV 飽和度調整
        hsv_v=0.4,            # HSV 亮度調整

        # # for transfer learning
        # lr0=0.001,             # 初始學習率
        # lrf=0.01,              # 最終學習率比例
        # freeze=10,            # 凍結前 10 層
        
        # 硬體設定
        device=0,             # 使用 GPU (若只有 CPU 請設為 'cpu')
    )

if __name__ == "__main__":
    train_yolo()

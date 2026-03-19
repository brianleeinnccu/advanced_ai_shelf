# from ultralytics import YOLO

# def test_yolo():
#     # 載入預訓練模型
#     model = YOLO("D:\\USER\\Desktop\\product_train_last3085\\runs\\shelf_finetune_v26_v1_add_2000_yolo11n_finetune_600\\weights\\best.pt")

#     # 開始訓練
#     results = model.val(
#         split="test",   
#     )
# if __name__ == "__main__":
#     test_yolo()

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

def test_yolo():
    # 載入預訓練模型
    model = YOLO("D:\\USER\Desktop\\product_train_last3085\\runs\\detect\\shelf_finetune_v26_v1_add_2000_yolo11n_600_advanced_finetune_with_100\\weights\\best.pt")

    # 開始測試/驗證
    results = model.val(split="test",name="val_600_advanced_finetune_with_100",)

    # --- 自訂繪製字體放大的 Confusion Matrix ---
    # 1. 取得混淆矩陣原始數據與類別名稱
    cm = results.confusion_matrix.matrix
    
    # YOLO 的混淆矩陣預設會在最後多加一個 'background' (背景) 類別
    classes = list(model.names.values()) + ['background'] 

    # 2. 設定畫布尺寸與整體字體比例
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # 👈 放大座標軸與標題的字體

    # 3. 繪製熱力圖
    # annot_kws={"size": 16} 控制矩陣內部數字的大小
    sns.heatmap(cm, annot=True, fmt='.2g', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16}) # 👈 在這裡改矩陣內部數字的字體大小

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 4. 顯示與儲存
    plt.tight_layout()
    plt.savefig('custom_confusion_matrix.png', dpi=300)
    print("已成功儲存自訂字體大小的 Confusion Matrix！")

if __name__ == "__main__":
    test_yolo()
import matplotlib.pyplot as plt

# --- 1. 準備數據 ---
experiments = ['Exp 1(2000 + clear_half_blur)', 'Exp 2(2000 + clear_all_blur)', 'Exp 3(2718 photos)']

# 比例型指標 (Scores: 0~1)
precision = [0.908, 0.904, 0.973]
recall = [0.875, 0.927, 0.959]
map50 = [0.929, 0.959, 0.987]
map50_95 = [0.740, 0.768, 0.846]

# 數量型指標 (Counts)
fn = [585, 126, 121]  # 漏抓
fp = [83, 290, 351]   # 多抓

# --- 2. 建立畫布與子圖 (1列2欄) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 3. 繪製第一張圖：各項效能指標成長趨勢 ---
ax1.plot(experiments, precision, marker='o', linestyle='-', linewidth=2, label='Precision')
ax1.plot(experiments, recall, marker='s', linestyle='-', linewidth=2, label='Recall')
ax1.plot(experiments, map50, marker='^', linestyle='-', linewidth=2, label='mAP@50')
ax1.plot(experiments, map50_95, marker='d', linestyle='-', linewidth=2, label='mAP@50-95')

ax1.set_title('Model Performance Metrics Trend', fontsize=14)
ax1.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
ax1.set_ylim(0.7, 1.02) # 讓 Y 軸範圍更聚焦在數據變化區間
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- 4. 繪製第二張圖：FP (多抓) 與 FN (漏抓) 趨勢 ---
ax2.plot(experiments, fn, marker='o', linestyle='-', color='red', linewidth=2, label='FN (Missed)')
ax2.plot(experiments, fp, marker='s', linestyle='-', color='orange', linewidth=2, label='FP (Over-predicted)')

ax2.set_title('FP & FN Errors Trend', fontsize=14)
ax2.set_ylabel('Count', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# --- 5. 顯示圖表 ---
plt.tight_layout()
plt.show()
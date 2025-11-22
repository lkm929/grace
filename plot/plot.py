import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. 定義平滑函數 (Exponential Moving Average)
# ==========================================
def smooth(scalars, weight=0.9):  # Weight 越接近 1，曲線越平滑
    last = scalars[0]  # 第一個值
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # 計算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# ==========================================
# 2. 讀取數據
# ==========================================
# 請確保 csv 檔案在同一目錄下，或修改為正確路徑
df_loss = pd.read_csv('TrainingLoss.csv')
df_dice = pd.read_csv('DiceScore.csv')

# ==========================================
# 3. 設定學術圖表樣式 (Publication Quality)
# ==========================================
# 設定字體為襯線體 (類似 Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14          # 字體大小
plt.rcParams['axes.linewidth'] = 1.5    # 座標軸線條加粗
plt.rcParams['xtick.major.width'] = 1.5 # X軸刻度加粗
plt.rcParams['ytick.major.width'] = 1.5 # Y軸刻度加粗

# 創建畫布：1 行 2 列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# 4. 繪製 Training Loss (左圖)
# ==========================================
# 繪製原始數據 (淺色背景，alpha=0.3)
ax1.plot(df_loss['Step'], df_loss['Value'], 
         color='gray', alpha=0.3, linewidth=1, label='Raw')

# 繪製平滑數據 (深色實線)
loss_smooth = smooth(df_loss['Value'], weight=0.9) # 調整 weight 來改變平滑度
ax1.plot(df_loss['Step'], loss_smooth, 
         color='#d62728', linewidth=2.5, label='Smoothed') # 紅色

ax1.set_xlabel('Step', fontweight='bold')
ax1.set_ylabel('Training Loss', fontweight='bold')
ax1.set_title('Training Loss Curve', fontweight='bold')
ax1.legend(frameon=False) # 去除圖例邊框
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xlim(left=0)

# ==========================================
# 5. 繪製 Dice Score (右圖)
# ==========================================
# Dice Score 數據點通常較少，可以用點+線表示
# 原始數據 (淺色點)
ax2.scatter(df_dice['Step'], df_dice['Value'], 
            color='gray', alpha=0.4, s=10, label='Raw')

# 平滑數據 (深色實線)
dice_smooth = smooth(df_dice['Value'], weight=0.6) 
ax2.plot(df_dice['Step'], dice_smooth, 
         color='#1f77b4', linewidth=2.5, label='Smoothed') # 藍色

ax2.set_xlabel('Step', fontweight='bold')
ax2.set_ylabel('Dice Score', fontweight='bold')
ax2.set_title('Validation Dice Score', fontweight='bold')
ax2.legend(frameon=False, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlim(left=0)
ax2.set_ylim(0, 1.0) # Dice 範圍通常在 0~1 之間

# ==========================================
# 6. 調整佈局與保存
# ==========================================
plt.tight_layout()

# 保存為 PDF (矢量圖，適合插入 LaTeX/Word，放大不失真)
plt.savefig('training_results.pdf', dpi=300, bbox_inches='tight')

# 保存為 PNG (預覽用)
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')

print("圖表已保存為 training_results.pdf 和 training_results.png")
plt.show()
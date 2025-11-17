import os
import json
import random
import csv  # 匯入 CSV 模組

# --- 您需要設定的參數 ---
# 您的資料根目錄 (相對於 tojson.py)
DATA_ROOT = r"C:\Users\iris\Desktop\GRACE\Data" 
# 您想輸出的 JSON 檔案名稱
OUTPUT_JSON = r"C:\Users\iris\Desktop\GRACE\Data\dataset.json"
# 驗證集所佔的比例 (例如 0.2 = 20%)
VALIDATION_SPLIT = 0.2
# --------------------------

def check_dir_names(base_path, name_with_underscore):
    """
    檢查資料夾名稱，支援有底線或空格。
    例如：傳入 "Full-Head_Segmentation"，
    它會檢查 "Full-Head_Segmentation" 和 "Full-Head Segmentation"
    """
    name_with_space = name_with_underscore.replace("_", " ")
    path_underscore = os.path.join(base_path, name_with_underscore)
    path_space = os.path.join(base_path, name_with_space)

    if os.path.isdir(path_underscore):
        return path_underscore, name_with_underscore
    elif os.path.isdir(path_space):
        return path_space, name_with_space
    else:
        # 如果兩個都找不到，返回 None
        return None, None

def create_datalist_from_csv(data_root, output_json, validation_split):
    """
    從 Metadata.csv 讀取 ID 列表，並產生 dataset.json
    """
    metadata_path = os.path.join(data_root, "Metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"錯誤：找不到 {metadata_path}！")
        return

    # 1. 從 CSV 讀取所有有效的 Subject IDs
    subject_ids = []
    try:
        # 'utf-8-sig' 用來處理 CSV 檔案開頭可能存在的 BOM (Byte Order Mark)
        with open(metadata_path, 'r', encoding='utf-8-sig') as f: 
            reader = csv.reader(f)
            for row in reader:
                if row: # 確保不是空行
                    subject_id = row[0].strip()
                    if subject_id: # 確保 ID 不是空字串
                        subject_ids.append(subject_id) 
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")
        return

    if not subject_ids:
        print("錯誤：CSV 是空的或無法讀取。")
        return

    print(f"從 {metadata_path} 讀取了 {len(subject_ids)} 個 Subject IDs。")

    all_file_pairs = []
    folders_to_check = ["Anonymized_Subjects", "Control_Subjects"]

    # 2. 根據 ID 列表去尋找檔案
    for subject_id in subject_ids:
        found_pair_for_id = False
        for folder in folders_to_check:
            subject_folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(subject_folder_path):
                continue

            # --- 偵測標籤和影像資料夾 ---
            image_dir_path, image_dir_name = check_dir_names(subject_folder_path, "T1-Weighted_MRI")
            label_dir_path, label_dir_name = check_dir_names(subject_folder_path, "Full-Head_Segmentation")
            # 如果這個 subject folder (Anonymized/Control) 缺少 T1 或 Label 資料夾，就跳過
            if not image_dir_path or not label_dir_path:
                continue

            # --- 根據 ID 尋找檔案 ---
            # 影像檔案 (X)
            image_filename = f"{subject_id}_deface.nii"
            image_path = os.path.join(image_dir_path, image_filename)

            found_label_path = None
            found_label_filename = None

            # 標籤檔案 (Y) - 嘗試兩種可能的命名
            # 模式 1: [ID].label.nii (例如: subj1.label.nii)
            label_filename_1 = f"{subject_id}_label_deface.nii"
            label_path_1 = os.path.join(label_dir_path, label_filename_1)
            
            # 模式 2: [ID].nii (例如: GU039.nii)
            label_filename_2 = f"{subject_id}.nii"
            label_path_2 = os.path.join(label_dir_path, label_filename_2)

            # 必須影像 (X) 存在，才能進行配對
            if os.path.exists(image_path):
                # 檢查標籤 (Y) 是否存在
                if os.path.exists(label_path_1):
                    found_label_path = label_path_1
                    found_label_filename = label_filename_1
                elif os.path.exists(label_path_2):
                    found_label_path = label_path_2
                    found_label_filename = label_filename_2

            # 如果影像 (X) 和 標籤 (Y) 都找到了
            if found_label_path:
                # 建立 JSON 內部需要的相對路徑
                rel_image_path = os.path.join(data_root, folder, image_dir_name, image_filename).replace("\\", "/")
                rel_label_path = os.path.join(data_root, folder, label_dir_name, found_label_filename).replace("\\", "/")
                
                all_file_pairs.append({
                    "image": rel_image_path,
                    "label": rel_label_path
                })
                found_pair_for_id = True
                break # 找到了，跳出內層迴圈 (不再檢查另一個資料夾)
        
        if not found_pair_for_id:
            print(f"  -> 警告：CSV中的ID '{subject_id}' 找不到對應的 影像/標籤 檔案對。")

    # 3. 建立 JSON
    if not all_file_pairs:
        print("\n錯誤：根據 CSV 列表，找不到任何影像/標籤對。")
        return

    print(f"\n總共找到 {len(all_file_pairs)} 筆有效的資料對。")

    # 隨機打亂並分割
    random.shuffle(all_file_pairs)
    split_index = int(len(all_file_pairs) * (1 - validation_split))
    
    train_files = all_file_pairs[:split_index]
    val_files = all_file_pairs[split_index:]

    final_json_data = {
        "description": "Auto-generated dataset for MONAI from Metadata.csv",
        "numTraining": len(train_files),
        "numValidation": len(val_files),
        "training": train_files,
        "validation": val_files
    }

    # 確保 JSON 檔案寫在 tojson.py 旁邊
    output_path = os.path.join(os.getcwd(), output_json)
    if data_root in output_path: 
        output_path = output_json
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_data, f, indent=4)

    print(f"\n成功！已建立 {output_path}：")
    print(f" - {len(train_files)} 筆訓練資料")
    print(f" - {len(val_files)} 筆驗證資料")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 確保 Data 資料夾在 tojson.py 旁邊
    data_root_path = os.path.join(script_dir, DATA_ROOT)
    
    if not os.path.isdir(data_root_path):
        print(f"錯誤：找不到 '{DATA_ROOT}' 資料夾。")
        print(f"請確保 '{DATA_ROOT}' 資料夾與 {os.path.basename(__file__)} 在同一目錄下。")
    else:
        # 我們傳遞 'Data' (相對路徑)，因為 JSON 內的_路徑_需要相對於 MONAI 程式碼的 data_dir
        create_datalist_from_csv(DATA_ROOT, OUTPUT_JSON, VALIDATION_SPLIT)
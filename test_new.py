# coding=utf-8

"""
python .\test_new.py --model_load_name unetr_20251122-182209.pth
python .\test_new.py --model_load_name aftunet_20251122-184618.pth
"""



#load packages:
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt  # 新增：用於繪圖

#load monai functions - 
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Spacingd,
    LoadImaged,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    EnsureChannelFirstd
)

from monai.config import print_config

from monai.data import (
    Dataset,
    DataLoader,
    pad_list_data_collate,
    load_decathlon_datalist
)

#-----------------------------------
# Pick device to run on
#-----------------------------------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print(f"DEBUG Using device: {device}")

if device.type == "cuda":
    print('DEBUG GPUs:', end='')
    print(torch.cuda.device_count())

print_config()

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--spatial_size", type=int, default=64, help="one patch dimension")
parser.add_argument("--a_min_value", type=int, default=0, help="minimum image intensity")
parser.add_argument("--N_classes", type=int, default=7, help="number of tissues classes")
parser.add_argument("--a_max_value", type=int, default=255, help="maximum image intensity")
parser.add_argument("--batch_size_test", type=int, default=1, help="batch size testing data")
parser.add_argument("--model_load_name", type=str, default="unetr_v5_cos.pth", help="model to load")
parser.add_argument("--dataparallel", type=str, default="False", help="did your model use multi-gpu")
parser.add_argument("--json_name", type=str, default=r"dataset.json", help="name of the file used to map data splits")
# parser.add_argument("--data_dir", type=str, default=r"C:\Users\51236\Documents\CV\grace\Data", help="directory the dataset is in")
parser.add_argument("--data_dir", type=str, default=r"C:\Users\irisc\Documents\CV\grace\Data", help="directory the dataset is in")
# parser.add_argument("--data_dir", type=str, default=r"C:\Users\iris\Desktop\GRACE\Data", help="directory the dataset is in")
args = parser.parse_args()

split_JSON = args.json_name
datasets = os.path.join(args.data_dir, split_JSON)
print(f"Using dataset file: {datasets}")

#-----------------------------------
# Data transformations
# 修改點：加入 label 的讀取與處理，以確保能畫出 Ground Truth
#-----------------------------------
test_transforms = Compose(
    [
        # 嘗試讀取 image 和 label。如果 label 不存在 (例如真正的測試集)，allow_missing_keys=True 會防止報錯
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"), # 影像用線性插值，標籤用最近鄰插值
            allow_missing_keys=True
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None, allow_missing_keys=True),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min_value, a_max=args.a_max_value, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
    ]
)

#-----------------------------------
# Set up data loaders
# 注意：這取決於 dataset.json 的 "test" 區塊是否有 "label" 欄位。
# 如果 "test" 只有 "image"，則 Ground Truth 無法繪製。
#-----------------------------------
test_files = load_decathlon_datalist(datasets, True, "test")

test_ds = Dataset(
    data=test_files, transform=test_transforms, 
)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size_test, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_list_data_collate,
)

#-----------------------------------
# Set up gpu device and unetr model
#----------------------------------- 
from monai.networks.nets import UNETR
# model = UNETR(
#     in_channels=1,
#     out_channels=args.N_classes,
#     img_size=(args.spatial_size, args.spatial_size, args.spatial_size),
#     feature_size=16, 
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)

from model.aftunet import AFTUNET
model = AFTUNET(
    in_channels=1,
    out_channels=args.N_classes,
    img_size=(args.spatial_size, args.spatial_size, args.spatial_size),
    feature_size=16,     # 保持與 UNETR 一致
    hidden_size=768,     # 保持與 UNETR 一致
    mlp_dim=3072,        # 保持與 UNETR 一致
    num_heads=12,        # 保持與 UNETR 一致
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
    spatial_dims=3,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

#-----------------------------------
# Load Weights
#-----------------------------------
model.load_state_dict(torch.load(os.path.join(args.data_dir, args.model_load_name), map_location=device))
model.eval()

#-----------------------------------
# Directories
#-----------------------------------
ModelName = os.path.splitext(os.path.basename(args.model_load_name))[0]
save_dir = os.path.join(args.data_dir, "TestResults", ModelName)
os.makedirs(save_dir, exist_ok=True)

#-----------------------------------
# Inference Loop with Plotting
#-----------------------------------
print(f"Starting inference on {len(test_ds)} cases...")

# 為了避免重複 transform，建議直接遍歷 test_ds 
# (但在大量數據時建議用 DataLoader，這裡維持您的寫法以便理解)
case_num = len(test_ds)
for i in range(case_num):
    data_item = test_ds[i] # 這會觸發 transform
    img_name = data_item["image"].meta["filename_or_obj"].split("/")[-1]
    img = data_item["image"]
    
    # 嘗試取得 Label (Ground Truth)
    label = data_item.get("label") # 如果沒有 label 則為 None

    print(f"Processing {img_name}...")

    with torch.no_grad():
        test_inputs = torch.unsqueeze(img, 1).to(device)  # add channel dim
        test_outputs = sliding_window_inference(
            test_inputs,
            (args.spatial_size, args.spatial_size, args.spatial_size),
            4,
            model,
            overlap=0.8
        )

    # 取得預測結果 (移除 batch 維度)
    testimage_tensor = torch.argmax(test_outputs, dim=1)[0]
    testimage = testimage_tensor.detach().cpu().numpy()

    # --- 1. 儲存 NIfTI 檔案 (原本的功能) ---
    ref_nii = nib.load(img.meta["filename_or_obj"])
    ref_header = ref_nii.header.copy()
    ref_affine = ref_nii.affine

    new_img = nib.Nifti1Image(testimage.astype(np.uint8), affine=ref_affine, header=ref_header)
    
    # Windows path fix
    filename_base = os.path.splitext(img_name)[0]
    if "\\" in filename_base:
        filename_base = filename_base.split("\\")[-1]
    
    savepath = os.path.join(save_dir, filename_base + ".nii.gz")
    nib.save(new_img, savepath)

    # --- 2. 繪製並儲存對比圖 (新增功能) ---
    
    # 取出中間切片 (Z軸)
    slice_idx = testimage.shape[2] // 2
    
    # 準備 Ground Truth 切片
    if label is not None:
        gt_slice = label[0, :, :, slice_idx].cpu().numpy() # label 也有 channel dim
    else:
        gt_slice = None

    # 準備 Prediction 切片
    pred_slice = testimage[:, :, slice_idx]

    # 建立畫布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左圖：Ground Truth
    if gt_slice is not None:
        axes[0].imshow(gt_slice, cmap='tab10', interpolation='nearest')
        axes[0].set_title(f"Ground Truth (Slice {slice_idx})")
    else:
        axes[0].text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center')
        axes[0].set_title("Ground Truth")
    axes[0].axis('off') # 關閉座標軸

    # 右圖：Prediction
    axes[1].imshow(pred_slice, cmap='tab10', interpolation='nearest')
    axes[1].set_title(f"Prediction (Slice {slice_idx})")
    axes[1].axis('off')

    # 標題與儲存
    plt.suptitle(f"Case: {filename_base}")
    plt.tight_layout()
    
    plot_savepath = os.path.join(save_dir, filename_base + "_viz.png")
    plt.savefig(plot_savepath, dpi=150)
    plt.close() # 關閉圖表釋放記憶體

    print(f"Saved NIfTI to {savepath}")
    print(f"Saved Plot to {plot_savepath}")

print("Done!")
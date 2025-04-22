import os
import shutil

# 定义目标目录
target_directory = "E:/usd_aaset/ExportedUSD/5"

# 创建目标目录（如果不存在）
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 文件路径列表
file_paths = [
    "E:/usd_aaset/ExportedUSD/Countertop_L/Countertop_L_10x8.usdz",
    "E:/usd_aaset/ExportedUSD/Fridge/Fridge_29.usdz",
    "E:/usd_aaset/ExportedUSD/Dining_Table/Dining_Table_218_1.usdz",
    "E:/usd_aaset/ExportedUSD/RoboTHOR/RoboTHOR_chair_jokkmokk_v.usdz",
    "E:/usd_aaset/ExportedUSD/RoboTHOR/RoboTHOR_chair_jokkmokk_v.usdz",
    "E:/usd_aaset/ExportedUSD/RoboTHOR/RoboTHOR_chair_jokkmokk_v.usdz",
    "E:/usd_aaset/ExportedUSD/RoboTHOR/RoboTHOR_chair_jokkmokk_v.usdz",
    "E:/usd_aaset/ExportedUSD/bin/bin_22.usdz",
    "E:/usd_aaset/ExportedUSD/Shelving_Unit/Shelving_Unit_303_1.usdz",
    "E:/usd_aaset/ExportedUSD/Stool/Stool_4_1.usdz",
    "E:/usd_aaset/ExportedUSD/GarbageBag/GarbageBag_21_1.usdz",
    "E:/usd_aaset/ExportedUSD/TV_Stand/TV_Stand_206_3.usdz",
    "E:/usd_aaset/ExportedUSD/Dining_Table/Dining_Table_205_1.usdz",
    "E:/usd_aaset/ExportedUSD/Sofa/Sofa_214_1.usdz",
    "E:/usd_aaset/ExportedUSD/Armchair/Armchair_207_4.usdz",
    "E:/usd_aaset/ExportedUSD/Side_Table/Side_Table_302_1_6.usdz",
    "E:/usd_aaset/ExportedUSD/Wall_Decor_Photo/Wall_Decor_Photo_3V.usdz",
    "E:/usd_aaset/ExportedUSD/Wall_Decor_Photo/Wall_Decor_Photo_10.usdz",
    "E:/usd_aaset/ExportedUSD/Wall_Decor_Painting/Wall_Decor_Painting_10.usdz"
]

# 复制文件
for file_path in file_paths:
    if os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_directory, file_name)

        # 如果目标文件已存在，跳过复制
        if os.path.exists(target_path):
            print(f"File already exists: {file_name}")
        else:
            shutil.copy(file_path, target_path)
            print(f"Copied {file_name} to {target_directory}")
    else:
        print(f"File not found: {file_path}")

print("All files have been processed.")
import os
import shutil
import random

# Images文件夹绝对路径
images_dir = r"D:\machine_vision\pythonProject2\class\yolo\datasets\mydata\Images"
# 划分比例（列表切片的核心依赖，总和为1）
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
# 支持的图片格式
img_formats = ['.jpg', '.jpeg', '.png', '.bmp']


# 1. 创建train/val/test文件夹（若不存在）
train_dir = os.path.join(images_dir, "train")
val_dir = os.path.join(images_dir, "val")
test_dir = os.path.join(images_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 2. 获取原始图片列表（过滤掉train/val/test文件夹，仅保留图片文件）
img_files = []
for file_name in os.listdir(images_dir):
    file_path = os.path.join(images_dir, file_name)
    # 条件1：是文件（不是文件夹）；条件2：是支持的图片格式
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in img_formats:
        img_files.append(file_name)

# 3. 随机打乱图片列表（为切片划分提供随机分布，避免类别偏置）
random.shuffle(img_files)
total_img_num = len(img_files)
print(f"原始图片列表长度：{total_img_num} 张")
print(f"打乱后的图片列表前5个：{img_files[:5]}") # 可查看打乱效果

# 4. 列表切片核心：按比例计算切片索引，实现无重叠划分
train_slice_end = int(total_img_num * train_ratio)  # 训练集切片结束索引
val_slice_end = train_slice_end + int(total_img_num * val_ratio)  # 验证集切片结束索引

# 严格列表切片（无重叠，核心满足你的要求）
train_files = img_files[:train_slice_end]  # 切片1：训练集（从开头到train_slice_end）
val_files = img_files[train_slice_end:val_slice_end]  # 切片2：验证集（中间段）
test_files = img_files[val_slice_end:]  # 切片3：测试集（剩余所有，避免小数取整偏差）

# 校验：三个列表无重叠，且总数等于原始数量
assert len(set(train_files) & set(val_files)) == 0, "训练集和验证集有重叠！"
assert len(set(train_files) & set(test_files)) == 0, "训练集和测试集有重叠！"
assert len(set(val_files) & set(test_files)) == 0, "验证集和测试集有重叠！"
assert len(train_files) + len(val_files) + len(test_files) == total_img_num, "总数不匹配！"

# 5. 移动图片（真正的划分，非复制）
def move_files(file_list, dst_dir):
    for file_name in file_list:
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        # 移动文件（彻底从原目录转移到子文件夹，无副本）
        if os.path.exists(src_path): # 避免重复移动导致报错
            shutil.move(src_path, dst_path)
        else:
            print(f"跳过不存在的文件：{file_name}")

# 执行移动
print(f"\n开始列表切片移动划分：")
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

# 输出结果
print(f"划分完成！")
print(f"训练集（train）：{len(train_files)} 张（列表切片：[:{train_slice_end}]）")
print(f"验证集（val）：{len(val_files)} 张（列表切片：[{train_slice_end}:{val_slice_end}]）")
print(f"测试集（test）：{len(test_files)} 张（列表切片：[{val_slice_end}:]）")
print(f"原始Images目录下仅保留 train/val/test 文件夹，无原始图片（已全部移动）")
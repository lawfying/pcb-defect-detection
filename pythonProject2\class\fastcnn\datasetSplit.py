import os
import json
import shutil
import random

def split_dataset(json_dir, image_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    划分数据集为训练集、验证集和测试集
    """
    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, 'annotations', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    print(os.listdir(json_dir))
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(len(json_files))
    random.shuffle(json_files)

    # 划分数据集
    n_total = len(json_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = json_files[:n_train]
    val_files = json_files[n_train:n_train + n_val]
    test_files = json_files[n_train + n_val:]

    print(f"数据集划分: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")

    # 复制文件到相应目录
    for split, files in zip(splits, [train_files, val_files, test_files]):
        for json_file in files:
            # 复制JSON文件
            src_json = os.path.join(json_dir, json_file)
            dst_json = os.path.join(output_dir, 'annotations', split, json_file)

            # 读取JSON获取图像文件名
            with open(src_json, 'r') as f:
                data = json.load(f)

            image_name = data["filename"]
            src_image = os.path.join(image_dir, image_name)
            # 如果文件不存在，尝试添加扩展名
            if not os.path.exists(src_image):
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_path = os.path.join(src_image + ext)
                    if os.path.exists(potential_path):
                        src_image = potential_path
                        break

            if os.path.exists(src_image):
                dst_image = os.path.join(output_dir, 'images', split, os.path.basename(src_image))
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_json, dst_json)
            else:
                print(f"警告: 找不到图像文件 {src_image}")

    return os.path.join(output_dir, 'annotations'), os.path.join(output_dir, 'images')


# 使用示例
if __name__ == "__main__":
    original_json_dir = (".\datasets\labels_json")
    original_image_dir = ".\datasets\images"
    output_directory = "./new_datasets"

    train_anno_dir, train_img_dir = split_dataset(
        original_json_dir,
        original_image_dir,
        output_directory
    )
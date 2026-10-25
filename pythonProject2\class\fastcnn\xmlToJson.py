import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path


def xml_to_json(xml_file_path):
    """将XML标注文件转换为指定的JSON格式"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        result = {
            "filename": filename,
            "defect_info": []
        }

        class_mapping = {
            "missing_hole": 0,
            "mouse_bite": 1,
            "open_circuit": 2,
            "short": 3,
            "spur": 4,
            "spurious_copper": 5
        }

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_mapping:
                bndbox = obj.find('bndbox')
                defect_info = {
                    "class_id": class_mapping[name],
                    "class_name": name,
                    "x_topleft": int(bndbox.find('xmin').text),
                    "y_topleft": int(bndbox.find('ymin').text),
                    "x_bottomright": int(bndbox.find('xmax').text),
                    "y_bottomright": int(bndbox.find('ymax').text)
                }
                result["defect_info"].append(defect_info)

        return result

    except Exception as e:
        print(f"处理文件 {xml_file_path} 时出错: {str(e)}")
        return None


def convert_xml_directory(xml_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    converted_count = 0
    error_count = 0

    # 1.修改点：用rglob递归找所有子文件夹里的.xml文件（替换glob）
    # 2.修改点：匹配后缀为.xml的文件（去掉_readdataset）
    for xml_file in Path(xml_dir).rglob("*.xml"):
        print(f"正在转换: {xml_file.name}")
        json_data = xml_to_json(xml_file)

        if json_data is not None:
            # 保持原文件的子文件夹结构（避免所有JSON挤在一起）
            relative_path = xml_file.relative_to(xml_dir)
            output_subdir = os.path.join(output_dir, relative_path.parent)
            os.makedirs(output_subdir, exist_ok=True)

            output_filename = relative_path.stem + ".json"
            output_path = os.path.join(output_subdir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            converted_count += 1
            print(f"成功转换: {xml_file.name} -> {output_path}")
        else:
            error_count += 1
            print(f"转换失败: {xml_file.name}")

    print(f"\n转换完成!")
    print(f"成功转换: {converted_count} 个文件")
    print(f"转换失败: {error_count} 个文件")


if __name__ == "__main__":
    print("\n=== 批量转换目录 ===")
    # 3.修改路径：脚本在fastcnn文件夹下，所以换成./datasets/Annotations
    xml_directory = "./datasets/Annotations"
    output_directory = "./datasets/labels_json"
    convert_xml_directory(xml_directory, output_directory)
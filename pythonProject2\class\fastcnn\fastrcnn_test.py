import torch
from torch.utils.data import Dataset, DataLoader
import json, os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm


class PCBDefectDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.json_files[idx])

        with open(json_path, 'r') as f:
            data = json.load(f)

        image_path = os.path.join(self.image_dir, data["filename"] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        boxes, labels = [], []
        for defect in data["defect_info"]:
            x1 = defect["x_topleft"] / original_width
            y1 = defect["y_topleft"] / original_height
            x2 = defect["x_bottomright"] / original_width
            y2 = defect["y_bottomright"] / original_height

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(defect["class_id"] + 1)  # 注意：背景是0，所以类别从1开始

        # 确保至少有一个框
        if len(boxes) == 0:
            # 如果没有框，创建一个全图的假框
            boxes.append([0.1, 0.1, 0.9, 0.9])
            labels.append(1)  # 使用第一个类别

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes=7):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def compute_val_loss(model, images, targets, device='cpu'):
    """在验证阶段计算损失"""
    # 将模型设置为训练模式来计算损失
    model.train()

    # 前向传播计算损失
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    # 设置回评估模式
    model.eval()

    return losses.item()


def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    model.to(device)

    # 优化器参数调整
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # 训练阶段
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}')

        for images, targets in train_pbar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            train_pbar.set_postfix({'loss': losses.item()})

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}')
            for images, targets in val_pbar:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # 在验证阶段直接计算损失
                loss = compute_val_loss(model, images, targets, device)
                val_loss += loss
                val_pbar.set_postfix({'loss': loss})

        # 更新学习率
        lr_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 保存模型
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
            print(f'Model saved: model_epoch_{epoch + 1}.pth')

    torch.save(model.state_dict(), 'model_final.pth')
    print('Final model saved: fastrcnn.pt')
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 数据增强
    transforms_list = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_root = "new_datasets"

    # 创建数据集
    train_dataset = PCBDefectDataset(
        json_dir=os.path.join(dataset_root, 'annotations/train'),
        image_dir=os.path.join(dataset_root, 'images/train'),
        transform=transforms_list
    )

    val_dataset = PCBDefectDataset(
        json_dir=os.path.join(dataset_root, 'annotations/val'),
        image_dir=os.path.join(dataset_root, 'images/val'),
        transform=transforms_list
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 创建模型
    model = create_model()

    # 训练模型
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
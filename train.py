import os
import argparse
import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import swanlab


# ============== Dataset ==============
class FlowerDataset(Dataset):
    def __init__(self, image_dir, labels, indices, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = os.path.join(self.image_dir, f"image_{img_idx:05d}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[img_idx - 1]) - 1  # 标签从1开始，转为0-indexed

        if self.transform:
            image = self.transform(image)
        return image, label


# ============== SE Block ==============
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ============== SE-ResNet-18 ==============
def make_se_resnet18(num_classes=102, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # 在每个layer的每个block后插入SE模块
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, name)
        for i, block in enumerate(layer):
            channels = block.conv2.out_channels
            block.se = SEBlock(channels)
            original_forward = block.forward

            def make_forward(blk):
                def forward_with_se(x):
                    identity = x
                    out = blk.conv1(x)
                    out = blk.bn1(out)
                    out = blk.relu(out)
                    out = blk.conv2(out)
                    out = blk.bn2(out)
                    out = blk.se(out)
                    if blk.downsample is not None:
                        identity = blk.downsample(x)
                    out += identity
                    out = blk.relu(out)
                    return out
                return forward_with_se

            block.forward = make_forward(block)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============== Model Factory ==============
def create_model(experiment, num_classes=102):
    if experiment == "baseline":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif experiment == "scratch":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif experiment == "se_resnet":
        return make_se_resnet18(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


# ============== Training ==============
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="102 Flower Classification")
    parser.add_argument("--experiment", type=str, default="baseline",
                        choices=["baseline", "scratch", "se_resnet",
                                 "baseline_lr1", "baseline_lr2", "baseline_lr3"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fc_lr", type=float, default=0.01)
    parser.add_argument("--backbone_lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数实验的lr配置
    lr_configs = {
        "baseline_lr1": (0.005, 0.0005),
        "baseline_lr2": (0.01, 0.001),
        "baseline_lr3": (0.02, 0.002),
    }

    if args.experiment in lr_configs:
        args.fc_lr, args.backbone_lr = lr_configs[args.experiment]
        experiment_type = "baseline"
    else:
        experiment_type = args.experiment

    # 加载数据
    labels_mat = sio.loadmat(os.path.join(args.data_dir, "imagelabels.mat"))
    setid_mat = sio.loadmat(os.path.join(args.data_dir, "setid.mat"))

    all_labels = labels_mat["labels"][0]
    train_ids = setid_mat["trnid"][0]
    val_ids = setid_mat["valid"][0]
    test_ids = setid_mat["tstid"][0]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dir = os.path.join(args.data_dir, "102flowers")

    train_dataset = FlowerDataset(image_dir, all_labels, train_ids, train_transform)
    val_dataset = FlowerDataset(image_dir, all_labels, val_ids, val_transform)
    test_dataset = FlowerDataset(image_dir, all_labels, test_ids, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 创建模型
    model = create_model(experiment_type, num_classes=102)
    model = model.to(device)

    # 优化器：差分学习率
    if experiment_type == "scratch":
        optimizer = optim.SGD(model.parameters(), lr=args.fc_lr, momentum=0.9, weight_decay=1e-4)
    else:
        fc_params = list(model.fc.parameters())
        fc_param_ids = [id(p) for p in fc_params]
        backbone_params = [p for p in model.parameters() if id(p) not in fc_param_ids]
        optimizer = optim.SGD([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": fc_params, "lr": args.fc_lr}
        ], momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 初始化 swanlab
    run_name = f"{args.experiment}_fc{args.fc_lr}_bb{args.backbone_lr}_ep{args.epochs}"
    swanlab.init(
        project="102-flower-classification",
        experiment_name=run_name,
        mode="local",
        config={
            "experiment": args.experiment,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "fc_lr": args.fc_lr,
            "backbone_lr": args.backbone_lr,
            "optimizer": "SGD",
            "scheduler": "CosineAnnealingLR",
            "model": experiment_type,
        }
    )

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        swanlab.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": current_lr,
            "epoch": epoch + 1
        })

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(),
                       f"checkpoints/best_{args.experiment}.pth")

    # 测试集评估
    model.load_state_dict(torch.load(f"checkpoints/best_{args.experiment}.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"Experiment: {args.experiment}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*50}")

    swanlab.log({
        "test/accuracy": test_acc,
        "test/loss": test_loss,
        "best_val_accuracy": best_val_acc,
    })

    swanlab.finish()


if __name__ == "__main__":
    main()

"""
模型训练脚本。

用于基于 ResNet18 / ResNet50 迁移学习训练水果蔬菜分类模型。
运行示例：
python src/train.py --model resnet18 --epochs 20 --batch-size 32
"""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

# 在受限环境中避免 Matplotlib / fontconfig 尝试写入用户主目录缓存。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_cache"))
if "TORCH_HOME" not in os.environ and not os.access(Path.home() / ".cache", os.W_OK):
    os.environ["TORCH_HOME"] = str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_torch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="基于 ResNet 迁移学习的水果蔬菜图像分类训练脚本")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18", help="选择 ResNet18 或 ResNet50")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="数据集根目录，内部应包含 train/val/test")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小；ResNet50 显存或内存不足时可改为 16 或 8")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减系数")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 读取数据的进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--models-dir", type=Path, default=Path("outputs/models"), help="模型权重保存目录")
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"), help="训练曲线保存目录")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"), help="类别映射等报告保存目录")
    parser.add_argument("--model-path", type=Path, default=None, help="最佳模型保存路径，默认按模型名自动生成")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--fine-tune-all", action="store_true", help="训练全部网络参数；默认只训练最后的全连接层")
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子，尽量保证实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """自动选择可用设备：优先 CUDA，其次 Apple MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path):
    """确保输出目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def build_transforms():
    """定义训练、验证和测试阶段的图像预处理。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
            ),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(data_dir, batch_size, num_workers, device):
    """使用 ImageFolder 读取 train、val、test 三个数据集。"""
    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    # ImageFolder 会按类别文件夹名称排序生成 class_to_idx，三个划分必须保持一致。
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("data/val 的类别目录与 data/train 不一致，请检查数据集划分。")
    if test_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("data/test 的类别目录与 data/train 不一致，请检查数据集划分。")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_model(model_name, num_classes, pretrained=True, freeze_backbone=True):
    """构建迁移学习模型。"""
    # 兼容新版 torchvision 的 weights API 和旧版 torchvision 的 pretrained 参数。
    try:
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        else:
            raise ValueError("model_name 只能是 resnet18 或 resnet50")
    except AttributeError:
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("model_name 只能是 resnet18 或 resnet50")

    if freeze_backbone:
        # 迁移学习常见做法：先冻结预训练特征提取层，只训练最后的分类层。
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_class_mapping(class_to_idx, output_path):
    """保存类别名称到索引的映射，供评估、预测和 Streamlit 应用复用。"""
    idx_to_class = {str(index): class_name for class_name, index in class_to_idx.items()}
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_one_epoch(model, dataloader, criterion, device, optimizer=None):
    """执行一个 epoch；传入 optimizer 时为训练模式，否则为验证模式。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def plot_training_curves(history, output_path):
    """绘制并保存训练/验证 loss 和 accuracy 曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="train acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train():
    """执行模型训练流程。"""
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.models_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.reports_dir)

    model_path = args.model_path or args.models_dir / f"best_{args.model}.pth"
    curve_path = args.figures_dir / f"training_curve_{args.model}.png"
    class_mapping_path = args.reports_dir / "class_to_idx.json"

    device = get_device()
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, device
    )
    num_classes = len(class_to_idx)
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    save_class_mapping(class_to_idx, class_mapping_path)
    print(f"类别数量: {num_classes}")
    print(f"训练集: {len(train_loader.dataset)} 张，验证集: {len(val_loader.dataset)} 张，测试集: {len(test_loader.dataset)} 张")
    print(f"类别映射已保存到: {class_mapping_path}")

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.fine_tune_all,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
        print(f"val   loss: {val_loss:.4f}, val   acc: {val_acc:.4f}")

        # 只保存验证集准确率最高的模型，作为后续 evaluate.py 的默认输入。
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_name": args.model,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(index): class_name for index, class_name in idx_to_class.items()},
                "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
                "args": vars(args),
            }
            torch.save(checkpoint, model_path)
            print(f"已保存当前最佳模型: {model_path}，best val acc: {best_val_acc:.4f}")

    plot_training_curves(history, curve_path)
    print(f"\n训练完成，最佳验证集准确率: {best_val_acc:.4f}")
    print(f"最佳模型保存路径: {model_path}")
    print(f"训练曲线保存路径: {curve_path}")


if __name__ == "__main__":
    train()

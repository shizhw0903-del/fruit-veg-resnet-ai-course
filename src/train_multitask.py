"""
水果/蔬菜一级分组 + 50 类细分类的多任务 ResNet 训练脚本。

运行示例：
python src/train_multitask.py --model resnet50 --epochs 20 --batch-size 16 --data-dir data
python src/train_multitask.py --model resnet18 --epochs 20 --batch-size 32 --data-dir data
"""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

# 避免 Matplotlib/fontconfig 在受限环境中写入用户目录。
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_cache"),
)
if "TORCH_HOME" not in os.environ and not os.access(
    Path.home() / ".cache",
    os.W_OK,
):
    os.environ["TORCH_HOME"] = str(
        Path(tempfile.gettempdir()) / "fruit_veg_resnet_torch"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FRUIT_CLASSES = {
    "apple",
    "apricot",
    "avocado",
    "banana",
    "dragonfruit",
    "durian",
    "fig",
    "grapes",
    "grapefruit",
    "guava",
    "jack fruit",
    "kiwi",
    "lemon",
    "mandarine",
    "mango",
    "mangosteen",
    "orange",
    "passion_fruit",
    "pear",
    "pineapple",
    "plum",
    "pomegranate",
    "rambutan",
    "raspberry",
    "watermelon",
}

VEGETABLE_CLASSES = {
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "jalepeno",
    "lettuce",
    "onion",
    "paprika",
    "peas",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
}

GROUP_TO_IDX = {"fruit": 0, "vegetable": 1}
EXPECTED_CLASSES = FRUIT_CLASSES | VEGETABLE_CLASSES


def validate_group_definitions() -> None:
    """检查代码中的类别分组规则自身是否完整且互斥。"""
    overlap = sorted(FRUIT_CLASSES & VEGETABLE_CLASSES)
    if overlap:
        raise ValueError(
            "水果和蔬菜分组存在重复类别：" + ", ".join(overlap)
        )
    if len(FRUIT_CLASSES) != 25 or len(VEGETABLE_CLASSES) != 25:
        raise ValueError(
            "分组规则必须包含 25 个水果和 25 个蔬菜；"
            f"当前为 {len(FRUIT_CLASSES)} 和 {len(VEGETABLE_CLASSES)}"
        )
    if len(EXPECTED_CLASSES) != 50:
        raise ValueError(
            f"分组规则合计应为 50 类，当前为 {len(EXPECTED_CLASSES)} 类"
        )


def validate_dataset_classes(classes, split_name: str) -> None:
    """要求一个 split 的文件夹名与预定义的 50 类完全一致。"""
    actual_classes = set(classes)
    ungrouped = sorted(actual_classes - EXPECTED_CLASSES)
    missing = sorted(EXPECTED_CLASSES - actual_classes)

    errors = []
    if ungrouped:
        errors.append("未分组/多余类别：" + ", ".join(ungrouped))
    if missing:
        errors.append("缺少类别：" + ", ".join(missing))
    if errors:
        raise ValueError(
            f"{split_name} 类别文件夹与多任务分组规则不一致；"
            + "；".join(errors)
            + "。请检查类别名和数据路径，不会自动重命名文件夹。"
        )


def class_name_to_group_index(class_name: str) -> int:
    """根据类别名返回 fruit=0 或 vegetable=1。"""
    if class_name in FRUIT_CLASSES:
        return GROUP_TO_IDX["fruit"]
    if class_name in VEGETABLE_CLASSES:
        return GROUP_TO_IDX["vegetable"]
    raise ValueError(f"类别未分组：{class_name}")


class MultiTaskImageFolder(datasets.ImageFolder):
    """在 ImageFolder 的细分类标签之外增加水果/蔬菜分组标签。"""

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        validate_dataset_classes(self.classes, str(Path(root)))
        self.class_idx_to_group_idx = {
            class_index: class_name_to_group_index(class_name)
            for class_name, class_index in self.class_to_idx.items()
        }

    def __getitem__(self, index):
        image, class_label = super().__getitem__(index)
        group_label = self.class_idx_to_group_idx[class_label]
        return image, class_label, group_label


class MultiTaskResNet(nn.Module):
    """共享 ResNet 特征，并分别输出一级分组与 50 类细分类 logits。"""

    def __init__(
        self,
        model_name,
        num_classes=50,
        num_groups=2,
        pretrained=True,
    ):
        super().__init__()
        if num_classes <= 0 or num_groups <= 0:
            raise ValueError("num_classes 和 num_groups 必须大于 0")

        try:
            if model_name == "resnet18":
                weights = (
                    models.ResNet18_Weights.DEFAULT if pretrained else None
                )
                backbone = models.resnet18(weights=weights)
            elif model_name == "resnet50":
                weights = (
                    models.ResNet50_Weights.DEFAULT if pretrained else None
                )
                backbone = models.resnet50(weights=weights)
            else:
                raise ValueError("model_name 只能是 resnet18 或 resnet50")
        except AttributeError:
            if model_name == "resnet18":
                backbone = models.resnet18(pretrained=pretrained)
            elif model_name == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
            else:
                raise ValueError("model_name 只能是 resnet18 或 resnet50")

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.model_name = model_name
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.shared_feature = backbone
        self.group_head = nn.Linear(feature_dim, num_groups)
        self.class_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.shared_feature(x)
        group_logits = self.group_head(features)
        class_logits = self.class_head(features)
        return group_logits, class_logits


def parse_args():
    """解析训练参数。"""
    parser = argparse.ArgumentParser(
        description="水果/蔬菜分组与 50 类细分类多任务 ResNet 训练脚本"
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50"],
        default="resnet50",
        help="选择 ResNet18 或 ResNet50",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="数据集根目录，内部应包含 train/val/test",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批大小",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="权重衰减系数",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数",
    )
    parser.add_argument(
        "--group-loss-weight",
        type=float,
        default=0.3,
        help="总损失中 group_loss 的权重",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="模型、报告和图表的输出根目录",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="验证集细分类准确率不提升时的早停耐心轮数",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """将路径转换为绝对路径。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def validate_args(args) -> None:
    """检查命令行数值参数。"""
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.lr <= 0:
        raise ValueError("--lr 必须大于 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay 不能为负数")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")
    if args.group_loss_weight < 0:
        raise ValueError("--group-loss-weight 不能为负数")
    if args.patience <= 0:
        raise ValueError("--patience 必须大于 0")


def set_seed(seed):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """优先 CUDA，其次 Apple MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms():
    """定义多任务训练增强和标准验证/测试预处理。"""
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
    """构建 train、val、test 三个多任务 DataLoader。"""
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"数据目录不存在：{split_dir}。请检查 --data-dir。"
            )

    train_transform, eval_transform = build_transforms()
    train_dataset = MultiTaskImageFolder(
        data_dir / "train",
        transform=train_transform,
    )
    val_dataset = MultiTaskImageFolder(
        data_dir / "val",
        transform=eval_transform,
    )
    test_dataset = MultiTaskImageFolder(
        data_dir / "test",
        transform=eval_transform,
    )

    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError(
            "data/val 的类别索引与 data/train 不一致，请检查类别文件夹名。"
        )
    if test_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError(
            "data/test 的类别索引与 data/train 不一致，请检查类别文件夹名。"
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def run_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    group_loss_weight,
    optimizer=None,
):
    """训练或评估一个 epoch，并返回双任务损失和准确率。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_class_correct = 0
    running_group_correct = 0
    total_samples = 0

    for images, class_labels, group_labels in dataloader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        group_labels = group_labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            group_logits, class_logits = model(images)
            class_loss = criterion(class_logits, class_labels)
            group_loss = criterion(group_logits, group_labels)
            total_loss = class_loss + group_loss_weight * group_loss

            if is_train:
                total_loss.backward()
                optimizer.step()

        batch_size = class_labels.size(0)
        class_predictions = class_logits.argmax(dim=1)
        group_predictions = group_logits.argmax(dim=1)
        running_loss += total_loss.item() * batch_size
        running_class_correct += int(
            (class_predictions == class_labels).sum().item()
        )
        running_group_correct += int(
            (group_predictions == group_labels).sum().item()
        )
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader 中没有可训练或评估的样本")

    return {
        "loss": running_loss / total_samples,
        "class_acc": running_class_correct / total_samples,
        "group_acc": running_group_correct / total_samples,
        "total_samples": total_samples,
    }


def save_json(payload, output_path):
    """保存 UTF-8 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def plot_training_curves(history, output_path):
    """绘制双任务 loss 和 accuracy 训练曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="train loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="val loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Multi-task Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["train_class_acc"],
        marker="o",
        label="train class acc",
    )
    axes[1].plot(
        epochs,
        history["val_class_acc"],
        marker="o",
        label="val class acc",
    )
    axes[1].plot(
        epochs,
        history["train_group_acc"],
        marker="s",
        label="train group acc",
    )
    axes[1].plot(
        epochs,
        history["val_group_acc"],
        marker="s",
        label="val group acc",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Class and Group Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def safe_torch_load(path, device):
    """兼容不同 PyTorch 版本的 checkpoint 加载。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train():
    """执行多任务模型训练、早停和最佳模型测试。"""
    args = parse_args()
    validate_args(args)
    validate_group_definitions()
    set_seed(args.seed)

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    for directory in (models_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = (
        models_dir / f"best_multitask_{args.model}.pth"
    )
    curve_path = (
        figures_dir / f"multitask_training_curves_{args.model}.png"
    )
    class_mapping_path = reports_dir / "class_to_idx_multitask.json"
    group_mapping_path = reports_dir / "group_to_idx.json"

    device = get_device()
    print(f"使用设备：{device}")
    print(f"数据目录：{data_dir}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir,
        args.batch_size,
        args.num_workers,
        device,
    )
    class_to_idx = train_loader.dataset.class_to_idx
    num_classes = len(class_to_idx)
    num_groups = len(GROUP_TO_IDX)

    save_json(
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {
                str(index): class_name
                for class_name, index in class_to_idx.items()
            },
        },
        class_mapping_path,
    )
    save_json(GROUP_TO_IDX, group_mapping_path)

    print(f"类别数量：{num_classes}，分组数量：{num_groups}")
    print(
        f"训练集：{len(train_loader.dataset)} 张，"
        f"验证集：{len(val_loader.dataset)} 张，"
        f"测试集：{len(test_loader.dataset)} 张"
    )

    model = MultiTaskResNet(
        model_name=args.model,
        num_classes=num_classes,
        num_groups=num_groups,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_class_acc": [],
        "train_group_acc": [],
        "val_loss": [],
        "val_class_acc": [],
        "val_group_acc": [],
    }
    best_val_class_acc = -1.0
    best_val_group_acc = -1.0
    val_group_acc_at_best_class_epoch = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.group_loss_weight,
            optimizer=optimizer,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.group_loss_weight,
        )

        for prefix, metrics in (
            ("train", train_metrics),
            ("val", val_metrics),
        ):
            history[f"{prefix}_loss"].append(metrics["loss"])
            history[f"{prefix}_class_acc"].append(metrics["class_acc"])
            history[f"{prefix}_group_acc"].append(metrics["group_acc"])

        print(
            f"train loss: {train_metrics['loss']:.4f}, "
            f"class acc: {train_metrics['class_acc']:.4f}, "
            f"group acc: {train_metrics['group_acc']:.4f}"
        )
        print(
            f"val   loss: {val_metrics['loss']:.4f}, "
            f"class acc: {val_metrics['class_acc']:.4f}, "
            f"group acc: {val_metrics['group_acc']:.4f}"
        )

        best_val_group_acc = max(
            best_val_group_acc,
            val_metrics["group_acc"],
        )
        if val_metrics["class_acc"] > best_val_class_acc:
            best_val_class_acc = val_metrics["class_acc"]
            val_group_acc_at_best_class_epoch = val_metrics["group_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": args.model,
                "num_classes": num_classes,
                "num_groups": num_groups,
                "class_to_idx": class_to_idx,
                "group_to_idx": GROUP_TO_IDX,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "best_val_class_acc": best_val_class_acc,
                "best_val_group_acc": best_val_group_acc,
                "val_group_acc_at_best_class_epoch": (
                    val_group_acc_at_best_class_epoch
                ),
                "epoch": best_epoch,
                "group_loss_weight": args.group_loss_weight,
                "args": vars(args),
            }
            torch.save(checkpoint, model_path)
            print(
                f"已保存最佳多任务模型：{model_path}，"
                f"val class acc: {best_val_class_acc:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                "验证集细分类准确率未提升："
                f"{epochs_without_improvement}/{args.patience}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"触发 early stopping，停止于第 {epoch} 轮。")
                break

    plot_training_curves(history, curve_path)

    checkpoint = safe_torch_load(model_path, torch.device("cpu"))
    checkpoint["best_val_group_acc"] = best_val_group_acc
    checkpoint["val_group_acc_at_best_class_epoch"] = (
        val_group_acc_at_best_class_epoch
    )
    torch.save(checkpoint, model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_metrics = run_one_epoch(
        model,
        test_loader,
        criterion,
        device,
        args.group_loss_weight,
    )

    print("\n多任务训练总结：")
    print(
        f"  最佳 val_class_acc：{best_val_class_acc:.4f} "
        f"(epoch {best_epoch})"
    )
    print(f"  最佳 val_group_acc：{best_val_group_acc:.4f}")
    print(f"  test class_accuracy：{test_metrics['class_acc']:.4f}")
    print(f"  test group_accuracy：{test_metrics['group_acc']:.4f}")
    print(f"  模型保存路径：{model_path}")
    print(
        f"  报告保存路径：{class_mapping_path}；{group_mapping_path}"
    )
    print(f"  图表保存路径：{curve_path}")


if __name__ == "__main__":
    train()

"""
模型评估脚本。

用于在测试集上评估已训练模型，并生成准确率、分类报告和混淆矩阵等结果。
运行示例：
python src/evaluate.py --model resnet18 --model-path outputs/models/best_resnet18.pth
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

# 在受限环境中避免 Matplotlib / fontconfig 尝试写入用户主目录缓存。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="水果蔬菜图像分类模型评估脚本")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18", help="模型结构，checkpoint 中存在时优先使用 checkpoint")
    parser.add_argument("--model-path", type=Path, default=None, help="训练得到的最佳模型路径；默认使用 outputs/models/best_<model>.pth")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="数据集根目录，内部应包含 test")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小；ResNet50 显存或内存不足时可改为 16 或 8")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 读取数据的进程数")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"), help="评估报告保存目录")
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"), help="混淆矩阵图片保存目录")
    parser.add_argument("--class-to-idx", type=Path, default=Path("outputs/reports/class_to_idx.json"), help="类别映射 JSON 路径")
    return parser.parse_args()


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


def build_eval_transform():
    """定义测试阶段图像预处理，需与训练时验证集预处理保持一致。"""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name, num_classes):
    """构建与训练阶段一致的 ResNet 模型结构。"""
    # 评估只需要模型结构，不需要再次下载 ImageNet 预训练权重。
    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
        else:
            raise ValueError("model_name 只能是 resnet18 或 resnet50")
    except AttributeError:
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False)
        else:
            raise ValueError("model_name 只能是 resnet18 或 resnet50")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def safe_torch_load(path, device):
    """兼容不同 PyTorch 版本的 torch.load 参数。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_class_mapping(json_path):
    """从 JSON 文件读取类别映射。"""
    if not json_path.exists():
        raise FileNotFoundError(f"类别映射文件不存在: {json_path.resolve()}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # 兼容两种格式：直接保存 class_to_idx，或保存 {"class_to_idx": ...}。
    if "class_to_idx" in payload:
        return payload["class_to_idx"]
    return payload


def get_class_names(class_to_idx):
    """根据 class_to_idx 还原按索引排序的类别名称列表。"""
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    return [idx_to_class[index] for index in range(len(idx_to_class))]


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制并保存混淆矩阵图片。"""
    class_count = len(class_names)
    figure_size = max(10, min(24, class_count * 0.45))

    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate():
    """执行测试集评估。"""
    args = parse_args()
    ensure_dir(args.reports_dir)
    ensure_dir(args.figures_dir)
    if args.model_path is None:
        args.model_path = Path("outputs/models") / f"best_{args.model}.pth"
    args.model_path = args.model_path.expanduser()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {args.model_path.resolve()}\n"
            f"请检查 --model-path 参数，或先训练对应模型，例如: python src/train.py --model {args.model}"
        )

    device = get_device()
    print(f"使用设备: {device}")
    print(f"正在加载模型文件: {args.model_path.resolve()}")

    checkpoint = safe_torch_load(args.model_path, device)

    checkpoint_model_name = checkpoint.get("model_name") if isinstance(checkpoint, dict) else None
    if checkpoint_model_name is not None and checkpoint_model_name != args.model:
        raise ValueError(
            f"命令行参数 --model={args.model} 与 checkpoint 中的模型结构 {checkpoint_model_name} 不一致。\n"
            f"请使用正确命令，例如: python src/evaluate.py --model {checkpoint_model_name} --model-path {args.model_path}"
        )

    # 优先使用训练时保存到 checkpoint 中的类别映射；如果没有，则读取 class_to_idx.json。
    if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
    else:
        class_to_idx = load_class_mapping(args.class_to_idx)

    class_names = get_class_names(class_to_idx)
    num_classes = len(class_names)
    model_name = args.model

    test_dataset = datasets.ImageFolder(args.data_dir / "test", transform=build_eval_transform())
    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError("data/test 的类别目录与训练时保存的 class_to_idx 不一致，请检查数据集或类别映射文件。")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(model_name, num_classes).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 兼容只保存 state_dict 的情况。
        model.load_state_dict(checkpoint)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    labels = list(range(num_classes))
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_path = args.reports_dir / f"classification_report_{args.model}.txt"
    confusion_matrix_path = args.figures_dir / f"confusion_matrix_{args.model}.png"

    report_text = (
        f"Model: {model_name}\n"
        f"Model path: {args.model_path}\n"
        f"Test samples: {len(test_dataset)}\n\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro Precision: {precision_macro:.4f}\n"
        f"Macro Recall: {recall_macro:.4f}\n"
        f"Macro F1-score: {f1_macro:.4f}\n"
        f"Weighted Precision: {precision_weighted:.4f}\n"
        f"Weighted Recall: {recall_weighted:.4f}\n"
        f"Weighted F1-score: {f1_weighted:.4f}\n\n"
        f"Classification Report:\n{report}\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    plot_confusion_matrix(cm, class_names, confusion_matrix_path)

    print(report_text)
    print(f"分类报告已保存到: {report_path.resolve()}")
    print(f"混淆矩阵图片已保存到: {confusion_matrix_path.resolve()}")


if __name__ == "__main__":
    evaluate()

"""
分析 ResNet18 / ResNet50 在测试集上的类别混淆情况。

运行示例：
python src/analyze_confusions.py \
    --model resnet50 \
    --model-path outputs/models/best_resnet50.pth \
    --data-dir data/test \
    --batch-size 32 \
    --top-n 20
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# 避免 Matplotlib/fontconfig 在受限环境中写入用户目录。
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAIR_COLUMNS = [
    "true_class",
    "pred_class",
    "count",
    "true_total",
    "error_rate_in_true_class",
]
FOCUS_GROUPS = [
    ("bell pepper", "capsicum", "paprika"),
    ("corn", "sweetcorn"),
    ("chilli pepper", "jalepeno"),
    ("orange", "mandarine", "grapefruit"),
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="分析水果蔬菜分类模型最常见的类别混淆对"
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50"],
        default="resnet50",
        help="模型结构（默认：resnet50）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="模型权重路径；默认 outputs/models/best_<model>.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test",
        help="测试集目录，内部直接包含类别文件夹（默认：data/test）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="推理批大小（默认：32）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数（默认：0）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="输出和绘制的混淆类别对数量（默认：20）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="输出根目录（默认：outputs）",
    )
    parser.add_argument(
        "--class-to-idx",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "class_to_idx.json",
        help="训练时保存的类别映射 JSON",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """将用户路径转换为绝对路径。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    """检查数值参数是否合法。"""
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")
    if args.top_n <= 0:
        raise ValueError("--top-n 必须大于 0")


def get_device() -> torch.device:
    """优先使用 CUDA，其次 Apple MPS，最后使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform() -> transforms.Compose:
    """使用与项目 evaluate.py 一致的测试预处理。"""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """构建与训练阶段一致的 ResNet 分类模型。"""
    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
        else:
            raise ValueError(f"不支持的模型结构：{model_name}")
    except AttributeError:
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"不支持的模型结构：{model_name}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def safe_torch_load(model_path: Path, device: torch.device):
    """兼容不同 PyTorch 版本的 torch.load 参数。"""
    try:
        return torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(model_path, map_location=device)


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """兼容三种常见 checkpoint 格式并提取 state_dict。"""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 必须是字典或模型 state_dict")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("模型文件中没有有效的 state_dict")

    if all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def infer_output_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    """从全连接层权重推断模型输出类别数。"""
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None or not hasattr(fc_weight, "shape"):
        raise ValueError("state_dict 中缺少 fc.weight，无法确认输出类别数")
    return int(fc_weight.shape[0])


def load_checkpoint(
    model: nn.Module,
    model_path: Path,
    device: torch.device,
    expected_model_name: str = None,
    expected_num_classes: int = None,
) -> dict:
    """加载权重，并校验模型结构和输出类别数。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    checkpoint = safe_torch_load(model_path, device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}

    checkpoint_model_name = metadata.get("model_name")
    if (
        expected_model_name
        and checkpoint_model_name
        and checkpoint_model_name != expected_model_name
    ):
        raise ValueError(
            f"--model={expected_model_name} 与 checkpoint 中的 "
            f"model_name={checkpoint_model_name} 不一致"
        )

    state_dict = extract_state_dict(checkpoint)
    output_classes = infer_output_classes(state_dict)
    if expected_num_classes is not None and output_classes != expected_num_classes:
        raise ValueError(
            "模型输出层类别数与测试集类别数不一致："
            f"模型输出 {output_classes} 类，测试集/映射包含 "
            f"{expected_num_classes} 类"
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return metadata


def load_class_mapping(class_to_idx_path: Path) -> Dict[str, int]:
    """读取并验证训练时保存的 class_to_idx.json。"""
    if not class_to_idx_path.is_file():
        raise FileNotFoundError(
            f"类别映射文件不存在：{class_to_idx_path}"
        )

    with class_to_idx_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    mapping = payload.get("class_to_idx", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class_to_idx.json 中没有有效的类别映射")

    try:
        class_to_idx = {
            str(class_name): int(class_index)
            for class_name, class_index in mapping.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("class_to_idx.json 中的类别索引必须是整数") from exc

    expected_indices = list(range(len(class_to_idx)))
    actual_indices = sorted(class_to_idx.values())
    if actual_indices != expected_indices:
        raise ValueError(
            "类别索引必须唯一且从 0 连续编号；"
            f"实际索引为 {actual_indices}"
        )
    return class_to_idx


def class_names_from_mapping(class_to_idx: Dict[str, int]) -> List[str]:
    """按训练索引顺序恢复类别名称。"""
    return [
        class_name
        for class_name, _ in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]


def build_dataset_and_loader(
    data_dir: Path,
    class_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[datasets.ImageFolder, DataLoader]:
    """构建测试集，并确保 ImageFolder 标签顺序与训练时一致。"""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"测试集目录不存在：{data_dir}")

    dataset = datasets.ImageFolder(
        data_dir,
        transform=build_eval_transform(),
    )
    if not dataset.samples:
        raise ValueError(f"测试集目录中没有可读取的图片：{data_dir}")
    if dataset.class_to_idx != class_to_idx:
        missing = sorted(set(class_to_idx) - set(dataset.class_to_idx))
        extra = sorted(set(dataset.class_to_idx) - set(class_to_idx))
        details = []
        if missing:
            details.append(f"测试集缺少：{', '.join(missing)}")
        if extra:
            details.append(f"测试集多出：{', '.join(extra)}")
        if not details:
            details.append("类别索引顺序不一致")
        raise ValueError(
            "data/test 的类别映射与训练时 class_to_idx 不一致；"
            + "；".join(details)
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return dataset, DataLoader(dataset, **loader_kwargs)


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[dict]]:
    """推理并记录真实标签、预测标签和每张图片的分类结果。"""
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    if len(y_true) != len(dataloader.dataset):
        raise RuntimeError(
            "推理样本数与数据集样本数不一致："
            f"{len(y_true)} != {len(dataloader.dataset)}"
        )

    classes = dataloader.dataset.classes
    sample_records = []
    for (image_path, _), true_index, pred_index in zip(
        dataloader.dataset.samples,
        y_true,
        y_pred,
    ):
        sample_records.append(
            {
                "image_path": image_path,
                "true_class": classes[true_index],
                "pred_class": classes[pred_index],
                "correct": true_index == pred_index,
            }
        )
    return y_true, y_pred, sample_records


def analyze_confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    top_n: int,
) -> pd.DataFrame:
    """提取非对角线混淆项，并按次数和类内错误率排序。"""
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []

    for true_index, true_class in enumerate(classes):
        true_total = int(cm[true_index].sum())
        if true_total == 0:
            continue

        for pred_index, pred_class in enumerate(classes):
            if true_index == pred_index:
                continue
            count = int(cm[true_index, pred_index])
            if count == 0:
                continue
            rows.append(
                {
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "count": count,
                    "true_total": true_total,
                    "error_rate_in_true_class": count / true_total,
                }
            )

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    dataframe = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    dataframe = dataframe.sort_values(
        by=["count", "error_rate_in_true_class", "true_class", "pred_class"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return dataframe.head(top_n).reset_index(drop=True)


def plot_confusion_pairs(
    dataframe: pd.DataFrame,
    save_path: Path,
    requested_top_n: int = None,
) -> None:
    """绘制 Top-N 定向混淆类别对水平柱状图。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    displayed_n = len(dataframe)
    title_n = requested_top_n if requested_top_n is not None else displayed_n

    if dataframe.empty:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.text(
            0.5,
            0.5,
            "No misclassified class pairs",
            ha="center",
            va="center",
            fontsize=14,
        )
        axis.set_axis_off()
        axis.set_title(f"Top-{title_n} Confused Class Pairs")
    else:
        plot_data = dataframe.iloc[::-1].copy()
        plot_data["pair"] = (
            plot_data["true_class"] + " → " + plot_data["pred_class"]
        )
        figure_height = max(6.0, displayed_n * 0.42 + 1.8)
        figure, axis = plt.subplots(figsize=(13, figure_height))
        bars = axis.barh(
            plot_data["pair"],
            plot_data["count"],
            color="#2E86AB",
        )
        axis.set_xlabel("Misclassified Sample Count")
        axis.set_ylabel("True Class → Predicted Class")
        axis.set_title(f"Top-{title_n} Confused Class Pairs")
        axis.grid(axis="x", alpha=0.25)

        max_count = max(int(plot_data["count"].max()), 1)
        axis.set_xlim(0, max_count * 1.18)
        for bar, count in zip(bars, plot_data["count"]):
            axis.text(
                bar.get_width() + max_count * 0.015,
                bar.get_y() + bar.get_height() / 2,
                str(int(count)),
                va="center",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    cm,
    classes: Sequence[str],
    save_path: Path,
) -> None:
    """绘制包含全部类别的混淆矩阵。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure_size = max(18, min(26, len(classes) * 0.45))
    figure, axis = plt.subplots(
        figsize=(figure_size, figure_size * 0.9)
    )
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=False,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("Confusion Matrix")
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    axis.tick_params(axis="y", labelrotation=0, labelsize=8)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_top_pairs(dataframe: pd.DataFrame) -> None:
    """在控制台打印 Top-N 混淆类别对。"""
    print(f"\nTop-{len(dataframe)} 混淆类别对：")
    if dataframe.empty:
        print("  测试集中没有误分类样本。")
        return

    for rank, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(
            f"  {rank:>2}. {row.true_class} → {row.pred_class}: "
            f"{row.count}/{row.true_total} "
            f"({row.error_rate_in_true_class:.2%})"
        )


def print_focus_group_confusions(cm, classes: Sequence[str]) -> None:
    """打印预设相似类别组中的双向定向混淆情况。"""
    class_to_index = {
        class_name: index for index, class_name in enumerate(classes)
    }
    print("\n重点相似类别分析：")

    for group in FOCUS_GROUPS:
        missing = [
            class_name
            for class_name in group
            if class_name not in class_to_index
        ]
        if missing:
            for class_name in missing:
                print(f"  该类别不存在，跳过：{class_name}")

        available = [
            class_name
            for class_name in group
            if class_name in class_to_index
        ]
        if len(available) < 2:
            print(
                "  可用类别不足两个，跳过该组："
                + " / ".join(group)
            )
            continue

        print(f"  {' / '.join(available)}：")
        group_confusions = 0
        for true_class in available:
            true_index = class_to_index[true_class]
            true_total = int(cm[true_index].sum())
            for pred_class in available:
                if true_class == pred_class:
                    continue
                pred_index = class_to_index[pred_class]
                count = int(cm[true_index, pred_index])
                rate = count / true_total if true_total else 0.0
                group_confusions += count
                print(
                    f"    {true_class} → {pred_class}: "
                    f"{count}/{true_total} ({rate:.2%})"
                )
        print(f"    组内定向误判合计：{group_confusions}")


def check_checkpoint_mapping(
    checkpoint_metadata: dict,
    class_to_idx: Dict[str, int],
) -> None:
    """如果 checkpoint 带类别映射，则与 JSON 做一致性校验。"""
    checkpoint_mapping = checkpoint_metadata.get("class_to_idx")
    if checkpoint_mapping is None:
        return

    normalized_mapping = {
        str(class_name): int(class_index)
        for class_name, class_index in checkpoint_mapping.items()
    }
    if normalized_mapping != class_to_idx:
        raise ValueError(
            "checkpoint 中的 class_to_idx 与 "
            "outputs/reports/class_to_idx.json 不一致"
        )


def main() -> int:
    """运行测试集推理、混淆对分析并保存报告。"""
    args = parse_args()
    validate_args(args)

    model_path = args.model_path
    if model_path is None:
        model_path = (
            PROJECT_ROOT
            / "outputs"
            / "models"
            / f"best_{args.model}.pth"
        )

    model_path = resolve_path(model_path)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    class_to_idx_path = resolve_path(args.class_to_idx)

    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"confusion_pairs_{args.model}.csv"
    pairs_figure_path = (
        figures_dir / f"confusion_pairs_{args.model}.png"
    )
    matrix_figure_path = (
        figures_dir / f"confusion_matrix_{args.model}.png"
    )

    class_to_idx = load_class_mapping(class_to_idx_path)
    classes = class_names_from_mapping(class_to_idx)
    num_classes = len(classes)
    device = get_device()

    print(f"使用设备：{device}")
    print(f"模型结构：{args.model}")
    print(f"模型文件：{model_path}")
    print(f"测试集目录：{data_dir}")
    print(f"类别数量：{num_classes}")

    dataset, dataloader = build_dataset_and_loader(
        data_dir=data_dir,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if len(dataset.classes) != num_classes:
        raise ValueError(
            "ImageFolder 类别数量与 class_to_idx.json 不一致："
            f"{len(dataset.classes)} != {num_classes}"
        )

    model = build_model(args.model, num_classes)
    checkpoint_metadata = load_checkpoint(
        model=model,
        model_path=model_path,
        device=device,
        expected_model_name=args.model,
        expected_num_classes=num_classes,
    )
    check_checkpoint_mapping(checkpoint_metadata, class_to_idx)

    print(f"测试样本数：{len(dataset)}")
    print("正在执行测试集推理...")
    y_true, y_pred, sample_records = run_inference(
        model,
        dataloader,
        device,
    )
    if len(sample_records) != len(dataset):
        raise RuntimeError("逐图片预测记录数量与测试集数量不一致")

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    pairs_dataframe = analyze_confusion_pairs(
        y_true,
        y_pred,
        classes,
        args.top_n,
    )

    pairs_dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    plot_confusion_pairs(
        pairs_dataframe,
        pairs_figure_path,
        requested_top_n=args.top_n,
    )
    plot_confusion_matrix(cm, classes, matrix_figure_path)

    print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print_top_pairs(pairs_dataframe)
    print_focus_group_confusions(cm, classes)
    print("\n输出文件：")
    print(f"  CSV：{csv_path}")
    print(f"  Top-N 混淆对图：{pairs_figure_path}")
    print(f"  完整混淆矩阵：{matrix_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

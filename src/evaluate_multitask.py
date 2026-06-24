"""
水果/蔬菜一级分组 + 50 类细分类的多任务 ResNet 评估脚本。

运行示例：
python src/evaluate_multitask.py \
    --model resnet50 \
    --model-path outputs/models/best_multitask_resnet50.pth \
    --data-dir data/test \
    --batch-size 16
"""

import argparse
import csv
import json
import os
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

try:
    from train_multitask import (
        GROUP_TO_IDX,
        MultiTaskImageFolder,
        MultiTaskResNet,
        build_transforms,
        get_device,
        validate_group_definitions,
    )
except ImportError:
    from .train_multitask import (
        GROUP_TO_IDX,
        MultiTaskImageFolder,
        MultiTaskResNet,
        build_transforms,
        get_device,
        validate_group_definitions,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """解析评估参数。"""
    parser = argparse.ArgumentParser(
        description="水果/蔬菜分组与 50 类细分类多任务模型评估脚本"
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50"],
        default="resnet50",
        help="模型结构",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="多任务模型路径；默认 outputs/models/best_multitask_<model>.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test",
        help="测试集目录，内部直接包含 50 个类别文件夹",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="评估批大小",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="报告和图表的输出根目录",
    )
    parser.add_argument(
        "--class-to-idx",
        type=Path,
        default=None,
        help="多任务类别映射；默认 outputs/reports/class_to_idx_multitask.json",
    )
    parser.add_argument(
        "--group-to-idx",
        type=Path,
        default=None,
        help="分组映射；默认 outputs/reports/group_to_idx.json",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """将路径转换为绝对路径。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def validate_args(args) -> None:
    """检查数值参数。"""
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")


def safe_torch_load(path, device):
    """兼容不同 PyTorch 版本的 checkpoint 加载。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    """兼容 model_state_dict、state_dict 和直接 state_dict。"""
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


def load_json_mapping(path, nested_key=None):
    """读取映射 JSON，并兼容嵌套格式。"""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if nested_key and isinstance(payload, dict) and nested_key in payload:
        payload = payload[nested_key]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"映射文件内容无效：{path}")
    return {
        str(name): int(index)
        for name, index in payload.items()
    }


def validate_contiguous_mapping(mapping, mapping_name):
    """确保映射索引唯一且从 0 连续编号。"""
    expected = list(range(len(mapping)))
    actual = sorted(mapping.values())
    if actual != expected:
        raise ValueError(
            f"{mapping_name} 索引必须从 0 连续编号；实际为 {actual}"
        )


def resolve_mappings(
    checkpoint,
    class_mapping_path,
    group_mapping_path,
):
    """综合 checkpoint 与 JSON，确定并校验两个任务的映射。"""
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    checkpoint_class_mapping = metadata.get("class_to_idx")
    checkpoint_group_mapping = metadata.get("group_to_idx")

    if checkpoint_class_mapping is not None:
        checkpoint_class_mapping = {
            str(name): int(index)
            for name, index in checkpoint_class_mapping.items()
        }
    if checkpoint_group_mapping is not None:
        checkpoint_group_mapping = {
            str(name): int(index)
            for name, index in checkpoint_group_mapping.items()
        }

    json_class_mapping = load_json_mapping(
        class_mapping_path,
        nested_key="class_to_idx",
    )
    json_group_mapping = load_json_mapping(
        group_mapping_path,
        nested_key="group_to_idx",
    )

    if (
        checkpoint_class_mapping is not None
        and json_class_mapping is not None
        and checkpoint_class_mapping != json_class_mapping
    ):
        raise ValueError(
            "checkpoint 中的 class_to_idx 与 "
            "class_to_idx_multitask.json 不一致"
        )
    if (
        checkpoint_group_mapping is not None
        and json_group_mapping is not None
        and checkpoint_group_mapping != json_group_mapping
    ):
        raise ValueError(
            "checkpoint 中的 group_to_idx 与 group_to_idx.json 不一致"
        )

    class_to_idx = checkpoint_class_mapping or json_class_mapping
    group_to_idx = checkpoint_group_mapping or json_group_mapping
    if class_to_idx is None:
        raise FileNotFoundError(
            "无法获得 class_to_idx：checkpoint 未保存该字段，且映射文件不存在："
            f"{class_mapping_path}"
        )
    if group_to_idx is None:
        raise FileNotFoundError(
            "无法获得 group_to_idx：checkpoint 未保存该字段，且映射文件不存在："
            f"{group_mapping_path}"
        )

    validate_contiguous_mapping(class_to_idx, "class_to_idx")
    validate_contiguous_mapping(group_to_idx, "group_to_idx")
    if group_to_idx != GROUP_TO_IDX:
        raise ValueError(
            f"group_to_idx 必须固定为 {GROUP_TO_IDX}，实际为 {group_to_idx}"
        )
    return class_to_idx, group_to_idx


def infer_head_sizes(state_dict):
    """从两个 head 的权重推断输出维度。"""
    class_weight = state_dict.get("class_head.weight")
    group_weight = state_dict.get("group_head.weight")
    if class_weight is None or group_weight is None:
        raise ValueError(
            "state_dict 缺少 class_head.weight 或 group_head.weight；"
            "请确认加载的是多任务模型"
        )
    return int(class_weight.shape[0]), int(group_weight.shape[0])


def load_multitask_model(
    model_name,
    model_path,
    device,
    class_mapping_path,
    group_mapping_path,
):
    """加载多任务模型、类别映射和 checkpoint 元数据。"""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"多任务模型文件不存在：{model_path}。"
            "请检查 --model-path 或先运行 train_multitask.py。"
        )

    checkpoint = safe_torch_load(model_path, device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    checkpoint_model_name = metadata.get("model_name")
    if checkpoint_model_name and checkpoint_model_name != model_name:
        raise ValueError(
            f"--model={model_name} 与 checkpoint 中的 "
            f"model_name={checkpoint_model_name} 不一致"
        )

    class_to_idx, group_to_idx = resolve_mappings(
        checkpoint,
        class_mapping_path,
        group_mapping_path,
    )
    state_dict = extract_state_dict(checkpoint)
    head_num_classes, head_num_groups = infer_head_sizes(state_dict)

    if head_num_classes != len(class_to_idx):
        raise ValueError(
            "class_head 输出维度与类别映射不一致："
            f"{head_num_classes} != {len(class_to_idx)}"
        )
    if head_num_groups != len(group_to_idx):
        raise ValueError(
            "group_head 输出维度与分组映射不一致："
            f"{head_num_groups} != {len(group_to_idx)}"
        )

    model = MultiTaskResNet(
        model_name=model_name,
        num_classes=len(class_to_idx),
        num_groups=len(group_to_idx),
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_to_idx, group_to_idx, metadata


def build_test_loader(data_dir, class_to_idx, batch_size, num_workers, device):
    """构建测试集并校验类别映射。"""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"测试集目录不存在：{data_dir}。请检查 --data-dir。"
        )

    _, eval_transform = build_transforms()
    dataset = MultiTaskImageFolder(
        data_dir,
        transform=eval_transform,
    )
    if dataset.class_to_idx != class_to_idx:
        missing = sorted(set(class_to_idx) - set(dataset.class_to_idx))
        extra = sorted(set(dataset.class_to_idx) - set(class_to_idx))
        details = []
        if missing:
            details.append("测试集缺少：" + ", ".join(missing))
        if extra:
            details.append("测试集多出：" + ", ".join(extra))
        if not details:
            details.append("类别索引顺序不一致")
        raise ValueError(
            "测试集类别与多任务模型不一致；" + "；".join(details)
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


def class_names_from_mapping(mapping):
    """按索引顺序恢复名称列表。"""
    return [
        name
        for name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]


def run_inference(model, dataloader, device, class_names, group_names):
    """执行双任务推理，并生成评估数组和逐图片结果。"""
    y_true_class = []
    y_pred_class = []
    y_true_group = []
    y_pred_group = []

    with torch.inference_mode():
        for images, class_labels, group_labels in dataloader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            group_logits, class_logits = model(images)
            class_predictions = class_logits.argmax(dim=1).cpu()
            group_predictions = group_logits.argmax(dim=1).cpu()

            y_true_class.extend(class_labels.tolist())
            y_pred_class.extend(class_predictions.tolist())
            y_true_group.extend(group_labels.tolist())
            y_pred_group.extend(group_predictions.tolist())

    if len(y_true_class) != len(dataloader.dataset):
        raise RuntimeError(
            "推理样本数与测试集样本数不一致："
            f"{len(y_true_class)} != {len(dataloader.dataset)}"
        )

    prediction_rows = []
    for (image_path, _), true_class, pred_class, true_group, pred_group in zip(
        dataloader.dataset.samples,
        y_true_class,
        y_pred_class,
        y_true_group,
        y_pred_group,
    ):
        prediction_rows.append(
            {
                "image_path": image_path,
                "true_class": class_names[true_class],
                "pred_class": class_names[pred_class],
                "true_group": group_names[true_group],
                "pred_group": group_names[pred_group],
                "class_correct": true_class == pred_class,
                "group_correct": true_group == pred_group,
            }
        )

    return {
        "y_true_class": y_true_class,
        "y_pred_class": y_pred_class,
        "y_true_group": y_true_group,
        "y_pred_group": y_pred_group,
        "prediction_rows": prediction_rows,
    }


def save_classification_report(
    y_true,
    y_pred,
    target_names,
    title,
    output_path,
):
    """保存带整体准确率的 sklearn classification report。"""
    labels = list(range(len(target_names)))
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    text = (
        f"{title}\n"
        f"Samples: {len(y_true)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def plot_confusion_matrix(
    matrix,
    labels,
    title,
    output_path,
    annotate=False,
):
    """绘制并保存细分类或分组混淆矩阵。"""
    if len(labels) <= 2:
        figure_size = (7, 6)
        label_size = 11
    else:
        size = max(18, min(26, len(labels) * 0.45))
        figure_size = (size, size * 0.9)
        label_size = 8

    figure, axis = plt.subplots(figsize=figure_size)
    sns.heatmap(
        matrix,
        annot=annotate,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=True,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=90 if len(labels) > 2 else 0)
    axis.tick_params(axis="y", labelrotation=0)
    axis.tick_params(labelsize=label_size)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_predictions_csv(rows, output_path):
    """保存逐图片双任务预测结果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "true_class",
        "pred_class",
        "true_group",
        "pred_group",
        "class_correct",
        "group_correct",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate():
    """执行多任务测试集评估并保存全部报告。"""
    args = parse_args()
    validate_args(args)
    validate_group_definitions()

    output_dir = resolve_path(args.output_dir)
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or (
        output_dir / "models" / f"best_multitask_{args.model}.pth"
    )
    class_mapping_path = args.class_to_idx or (
        reports_dir / "class_to_idx_multitask.json"
    )
    group_mapping_path = args.group_to_idx or (
        reports_dir / "group_to_idx.json"
    )

    model_path = resolve_path(model_path)
    data_dir = resolve_path(args.data_dir)
    class_mapping_path = resolve_path(class_mapping_path)
    group_mapping_path = resolve_path(group_mapping_path)

    class_report_path = (
        reports_dir
        / f"classification_report_multitask_{args.model}.txt"
    )
    group_report_path = (
        reports_dir
        / f"group_classification_report_multitask_{args.model}.txt"
    )
    predictions_path = (
        reports_dir / f"predictions_multitask_{args.model}.csv"
    )
    class_matrix_path = (
        figures_dir / f"confusion_matrix_multitask_{args.model}.png"
    )
    group_matrix_path = (
        figures_dir / f"group_confusion_matrix_multitask_{args.model}.png"
    )

    device = get_device()
    print(f"使用设备：{device}")
    print(f"模型文件：{model_path}")
    print(f"测试集目录：{data_dir}")

    model, class_to_idx, group_to_idx, metadata = load_multitask_model(
        model_name=args.model,
        model_path=model_path,
        device=device,
        class_mapping_path=class_mapping_path,
        group_mapping_path=group_mapping_path,
    )
    dataset, dataloader = build_test_loader(
        data_dir,
        class_to_idx,
        args.batch_size,
        args.num_workers,
        device,
    )
    class_names = class_names_from_mapping(class_to_idx)
    group_names = class_names_from_mapping(group_to_idx)

    print(f"测试样本数：{len(dataset)}")
    results = run_inference(
        model,
        dataloader,
        device,
        class_names,
        group_names,
    )

    class_accuracy = accuracy_score(
        results["y_true_class"],
        results["y_pred_class"],
    )
    group_accuracy = accuracy_score(
        results["y_true_group"],
        results["y_pred_group"],
    )
    class_cm = confusion_matrix(
        results["y_true_class"],
        results["y_pred_class"],
        labels=list(range(len(class_names))),
    )
    group_cm = confusion_matrix(
        results["y_true_group"],
        results["y_pred_group"],
        labels=list(range(len(group_names))),
    )

    save_classification_report(
        results["y_true_class"],
        results["y_pred_class"],
        class_names,
        f"Multi-task Fine-grained Classification Report ({args.model})",
        class_report_path,
    )
    save_classification_report(
        results["y_true_group"],
        results["y_pred_group"],
        group_names,
        f"Multi-task Group Classification Report ({args.model})",
        group_report_path,
    )
    plot_confusion_matrix(
        class_cm,
        class_names,
        "Multi-task Fine-grained Confusion Matrix",
        class_matrix_path,
        annotate=False,
    )
    plot_confusion_matrix(
        group_cm,
        group_names,
        "Fruit / Vegetable Confusion Matrix",
        group_matrix_path,
        annotate=True,
    )
    save_predictions_csv(results["prediction_rows"], predictions_path)

    print("\n多任务评估总结：")
    if "best_val_class_acc" in metadata:
        print(
            f"  最佳 val_class_acc："
            f"{float(metadata['best_val_class_acc']):.4f}"
        )
    if "best_val_group_acc" in metadata:
        print(
            f"  最佳 val_group_acc："
            f"{float(metadata['best_val_group_acc']):.4f}"
        )
    print(f"  test class_accuracy：{class_accuracy:.4f}")
    print(f"  test group_accuracy：{group_accuracy:.4f}")
    print(f"  模型保存路径：{model_path}")
    print(
        f"  报告保存路径：{class_report_path}；"
        f"{group_report_path}；{predictions_path}"
    )
    print(
        f"  图表保存路径：{class_matrix_path}；{group_matrix_path}"
    )


if __name__ == "__main__":
    evaluate()

"""
评估 ResNet18 / ResNet50 在常见图像扰动下的分类鲁棒性。

运行示例：
python src/robustness_eval.py \
    --model resnet18 \
    --model-path outputs/models/best_resnet18.pth \
    --data-dir data/test \
    --batch-size 32
"""

import argparse
import csv
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# 避免受限环境中的 Matplotlib/fontconfig 缓存写入用户目录。
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
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as transform_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class EvaluationResult:
    """单种扰动的评估结果。"""

    perturbation: str
    top1_accuracy: float
    total_samples: int
    correct_samples: int


class CenterOcclusion:
    """将 PIL 图片中心 30% 的矩形区域填充为黑色。"""

    def __init__(self, fraction: float = 0.30):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction 必须在 (0, 1] 范围内")
        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        result = image.copy()
        width, height = result.size
        occlusion_width = max(1, round(width * self.fraction))
        occlusion_height = max(1, round(height * self.fraction))
        left = (width - occlusion_width) // 2
        top = (height - occlusion_height) // 2
        right = left + occlusion_width
        bottom = top + occlusion_height
        ImageDraw.Draw(result).rectangle(
            (left, top, right - 1, bottom - 1),
            fill=(0, 0, 0),
        )
        return result


class GaussianNoise:
    """在 [0, 1] 图像 tensor 上添加轻微高斯噪声。"""

    def __init__(self, standard_deviation: float = 0.05):
        if standard_deviation < 0:
            raise ValueError("standard_deviation 不能为负数")
        self.standard_deviation = standard_deviation

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.standard_deviation
        return torch.clamp(tensor + noise, 0.0, 1.0)


class FixedBrightness:
    """使用固定系数调整 PIL 图片亮度。"""

    def __init__(self, factor: float):
        if factor < 0:
            raise ValueError("factor 不能为负数")
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self.factor)


class FixedGaussianBlur:
    """对 PIL 图片应用固定半径的高斯模糊。"""

    def __init__(self, radius: float):
        if radius < 0:
            raise ValueError("radius 不能为负数")
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class ClassMappedImageDataset(Dataset):
    """
    按 class_to_idx.json 中的索引读取图片。

    不依赖 ImageFolder 自己生成的字母序映射，从而保证标签顺序与训练时一致。
    """

    def __init__(
        self,
        data_dir: Path,
        class_to_idx: Dict[str, int],
        transform: Callable,
    ):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"测试目录中没有找到支持的图片：{data_dir}")

    @staticmethod
    def _natural_sort_key(path: Path) -> Tuple[Tuple[int, object], ...]:
        parts = re.split(r"(\d+)", path.as_posix().casefold())
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part)
            for part in parts
            if part
        )

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"测试集目录不存在：{self.data_dir.resolve()}")

        actual_class_dirs = {
            path.name
            for path in self.data_dir.iterdir()
            if path.is_dir() and not path.is_symlink()
        }
        unknown_classes = sorted(actual_class_dirs - set(self.class_to_idx))
        if unknown_classes:
            raise ValueError(
                "测试集包含 class_to_idx.json 中不存在的类别目录："
                + ", ".join(unknown_classes)
            )

        samples: List[Tuple[Path, int]] = []
        for class_name, class_index in sorted(
            self.class_to_idx.items(),
            key=lambda item: item[1],
        ):
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                print(f"[提示] 测试集中缺少类别目录，已跳过：{class_dir}")
                continue

            image_paths = sorted(
                (
                    path
                    for path in class_dir.rglob("*")
                    if path.is_file()
                    and not path.is_symlink()
                    and path.suffix.casefold() in IMAGE_EXTENSIONS
                ),
                key=self._natural_sort_key,
            )
            if not image_paths:
                print(f"[提示] 类别目录中没有支持的图片：{class_dir}")
                continue

            samples.extend((path, class_index) for path in image_paths)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"无法读取图片 {image_path}：{exc}") from exc
        return tensor, label


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="ResNet18 / ResNet50 图像扰动鲁棒性评估脚本"
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50"),
        default="resnet18",
        help="待评估的模型结构",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="模型 checkpoint；默认 outputs/models/best_<model>.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/test",
        help="测试集目录，内部直接包含类别文件夹",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="评估批大小",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="输出根目录，报告和图片分别写入 reports、figures",
    )
    parser.add_argument(
        "--class-to-idx",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/class_to_idx.json",
        help="训练时保存的类别映射 JSON",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数；macOS 上默认 0 更稳健",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于固定 noise 扰动",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """将相对路径按当前工作目录解析为绝对路径。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def set_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 随机数，便于重复 noise 评估。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """优先使用 CUDA，其次 Apple MPS，最后使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_torch_load(path: Path) -> object:
    """兼容新旧 PyTorch 版本，并先将 checkpoint 安全加载到 CPU。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_class_mapping(path: Path) -> Dict[str, int]:
    """读取并验证 class_to_idx.json。"""
    if not path.is_file():
        raise FileNotFoundError(f"类别映射文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
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
        raise ValueError("类别索引必须是整数") from exc

    expected_indices = list(range(len(class_to_idx)))
    actual_indices = sorted(class_to_idx.values())
    if actual_indices != expected_indices:
        raise ValueError(
            "类别索引必须唯一且从 0 连续编号；"
            f"期望 {expected_indices}，实际 {actual_indices}"
        )
    return class_to_idx


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """构建与训练脚本一致的 ResNet 分类器，不下载预训练权重。"""
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


def load_model(
    model_name: str,
    model_path: Path,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> nn.Module:
    """加载模型并检查 checkpoint 元数据与命令行参数是否一致。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    checkpoint = safe_torch_load(model_path)
    if isinstance(checkpoint, dict):
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != model_name:
            raise ValueError(
                f"--model={model_name} 与 checkpoint 中的 "
                f"model_name={checkpoint_model_name} 不一致"
            )

        checkpoint_mapping = checkpoint.get("class_to_idx")
        if checkpoint_mapping is not None:
            normalized_checkpoint_mapping = {
                str(class_name): int(class_index)
                for class_name, class_index in checkpoint_mapping.items()
            }
            if normalized_checkpoint_mapping != class_to_idx:
                raise ValueError(
                    "checkpoint 中的 class_to_idx 与 JSON 映射不一致"
                )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("模型文件既不是 state_dict，也不是有效 checkpoint")

    model = build_model(model_name, len(class_to_idx))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rotate_15(image: Image.Image) -> Image.Image:
    """固定逆时针旋转 15 度，空白区域使用黑色填充。"""
    return transform_functional.rotate(
        image,
        angle=15,
        interpolation=transforms.InterpolationMode.BILINEAR,
        expand=False,
        fill=0,
    )


def build_transforms() -> Dict[str, Callable]:
    """构建 clean 和六种扰动对应的预处理流程。"""
    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def compose(
        pil_perturbation: Optional[Callable] = None,
        tensor_perturbation: Optional[Callable] = None,
    ) -> transforms.Compose:
        steps: List[Callable] = [resize]
        if pil_perturbation is not None:
            steps.append(pil_perturbation)
        steps.append(to_tensor)
        if tensor_perturbation is not None:
            steps.append(tensor_perturbation)
        steps.append(normalize)
        return transforms.Compose(steps)

    return {
        "clean": compose(),
        "brightness_low": compose(FixedBrightness(factor=0.5)),
        "brightness_high": compose(FixedBrightness(factor=1.5)),
        "gaussian_blur": compose(FixedGaussianBlur(radius=2.0)),
        "rotate_15": compose(rotate_15),
        "center_occlusion": compose(CenterOcclusion(fraction=0.30)),
        "noise": compose(
            tensor_perturbation=GaussianNoise(standard_deviation=0.05)
        ),
    }


def build_loader(
    data_dir: Path,
    class_to_idx: Dict[str, int],
    transform: Callable,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """为一种扰动创建测试集 DataLoader。"""
    dataset = ClassMappedImageDataset(data_dir, class_to_idx, transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def evaluate_perturbation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    perturbation: str,
) -> EvaluationResult:
    """计算一种扰动下的 Top-1 accuracy。"""
    correct_samples = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )
            predictions = model(images).argmax(dim=1)
            correct_samples += int((predictions == labels).sum().item())
            total_samples += int(labels.size(0))

    if total_samples == 0:
        raise ValueError(f"{perturbation} 没有可评估样本")

    return EvaluationResult(
        perturbation=perturbation,
        top1_accuracy=correct_samples / total_samples,
        total_samples=total_samples,
        correct_samples=correct_samples,
    )


def write_csv(results: Sequence[EvaluationResult], output_path: Path) -> None:
    """保存鲁棒性评估 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "perturbation",
                "top1_accuracy",
                "total_samples",
                "correct_samples",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "perturbation": result.perturbation,
                    "top1_accuracy": f"{result.top1_accuracy:.6f}",
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                }
            )


def plot_results(
    results: Sequence[EvaluationResult],
    model_name: str,
    output_path: Path,
) -> None:
    """绘制并保存各扰动准确率柱状图。"""
    labels = [result.perturbation for result in results]
    accuracies = [result.top1_accuracy * 100.0 for result in results]
    colors = ["#2E86AB"] + ["#F18F01"] * (len(results) - 1)

    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(labels, accuracies, color=colors)
    axis.set_title(f"Robustness Evaluation - {model_name}")
    axis.set_xlabel("Perturbation")
    axis.set_ylabel("Top-1 Accuracy (%)")
    axis.set_ylim(0, max(100.0, max(accuracies, default=0.0) + 8.0))
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)

    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 1.2,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def validate_args(args: argparse.Namespace) -> None:
    """提前检查数值参数，给出清晰错误信息。"""
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")


def main() -> int:
    """执行全部鲁棒性评估并生成 CSV 与柱状图。"""
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    model_path = args.model_path
    if model_path is None:
        model_path = PROJECT_ROOT / "outputs/models" / f"best_{args.model}.pth"

    model_path = resolve_path(model_path)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    class_mapping_path = resolve_path(args.class_to_idx)

    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    csv_path = reports_dir / f"robustness_report_{args.model}.csv"
    figure_path = figures_dir / f"robustness_bar_{args.model}.png"

    device = get_device()
    print(f"使用设备：{device}")
    print(f"模型文件：{model_path}")
    print(f"测试目录：{data_dir}")
    print(f"类别映射：{class_mapping_path}")

    class_to_idx = load_class_mapping(class_mapping_path)
    model = load_model(
        model_name=args.model,
        model_path=model_path,
        class_to_idx=class_to_idx,
        device=device,
    )
    perturbation_transforms = build_transforms()

    results: List[EvaluationResult] = []
    expected_total: Optional[int] = None
    for perturbation, transform in perturbation_transforms.items():
        print(f"\n正在评估：{perturbation}")
        loader = build_loader(
            data_dir=data_dir,
            class_to_idx=class_to_idx,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        result = evaluate_perturbation(
            model=model,
            loader=loader,
            device=device,
            perturbation=perturbation,
        )
        if expected_total is None:
            expected_total = result.total_samples
        elif result.total_samples != expected_total:
            raise RuntimeError(
                "不同扰动评估到的样本数不一致，结果不可比较："
                f"期望 {expected_total}，实际 {result.total_samples}"
            )

        results.append(result)
        print(
            f"top1_accuracy={result.top1_accuracy:.4f} "
            f"({result.correct_samples}/{result.total_samples})"
        )

    write_csv(results, csv_path)
    plot_results(results, args.model, figure_path)

    print("\n鲁棒性评估完成。")
    print(f"CSV 报告：{csv_path}")
    print(f"柱状图：{figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
单张图片预测脚本。

用于对 real_test 或用户上传的单张水果蔬菜图片进行分类预测。
运行示例：
python src/predict.py --image data/real_test/example.jpg --model-path outputs/models/best_resnet18.pth
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 在受限环境中避免 PyTorch 预训练权重缓存写入不可写的用户目录。
if "TORCH_HOME" not in os.environ and not os.access(Path.home() / ".cache", os.W_OK):
    os.environ["TORCH_HOME"] = str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_torch")


def parse_args():
    """解析预测参数。"""
    parser = argparse.ArgumentParser(description="水果蔬菜图像单张图片 Top-3 预测脚本")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片路径")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/best_resnet18.pth",
        help="训练得到的最佳模型路径",
    )
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18", help="模型结构，checkpoint 中存在时优先使用 checkpoint")
    parser.add_argument(
        "--class-to-idx",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/class_to_idx.json",
        help="类别映射 JSON 路径",
    )
    parser.add_argument("--topk", type=int, default=3, help="输出概率最高的前 K 个类别")
    return parser.parse_args()


def get_device():
    """自动选择可用设备：优先 CUDA，其次 Apple MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval_transform():
    """定义预测阶段图像预处理，需与训练验证阶段保持一致。"""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name, num_classes):
    """构建与训练阶段一致的 ResNet18 / ResNet50 分类模型。"""
    # 推理阶段只需要网络结构，不需要再次加载 ImageNet 预训练权重。
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
    """从 class_to_idx.json 读取类别映射。"""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "class_to_idx" in payload:
        return payload["class_to_idx"]
    return payload


def get_class_names(class_to_idx):
    """根据 class_to_idx 还原按索引排序的类别名称列表。"""
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    return [idx_to_class[index] for index in range(len(idx_to_class))]


def load_image(image):
    """读取图片并转换为 RGB 格式，兼容路径和 Streamlit 上传后的 PIL 图片。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def load_model_for_inference(model_path, class_to_idx_path=None, model_name="resnet18", device=None):
    """加载模型、类别名称和设备，供命令行预测、Grad-CAM 和 Streamlit 共用。"""
    device = device or get_device()
    model_path = Path(model_path)
    checkpoint = safe_torch_load(model_path, device)

    # 优先使用 train.py 保存到 checkpoint 中的类别映射；否则读取 JSON 文件。
    if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
    elif class_to_idx_path is not None:
        class_to_idx = load_class_mapping(Path(class_to_idx_path))
    else:
        raise ValueError("checkpoint 中没有 class_to_idx，请通过 --class-to-idx 指定类别映射文件。")

    class_names = get_class_names(class_to_idx)
    resolved_model_name = checkpoint.get("model_name", model_name) if isinstance(checkpoint, dict) else model_name
    model = build_model(resolved_model_name, len(class_names))

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # 兼容只保存 model.state_dict() 的情况。
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, resolved_model_name, device


def predict_pil_image(image, model, class_names, device, topk=3):
    """对 PIL 图片进行 Top-K 预测，返回类别名和置信度。"""
    image = load_image(image)
    transform = build_eval_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        k = min(topk, len(class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k)

    results = []
    for probability, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        results.append(
            {
                "class_name": class_names[index],
                "class_index": index,
                "confidence": float(probability),
            }
        )
    return results


def predict_image_path(image_path, model_path, class_to_idx_path=None, model_name="resnet18", topk=3):
    """加载模型并预测单张图片路径，返回 Top-K 结果。"""
    model, class_names, resolved_model_name, device = load_model_for_inference(
        model_path=model_path,
        class_to_idx_path=class_to_idx_path,
        model_name=model_name,
    )
    results = predict_pil_image(load_image(image_path), model, class_names, device, topk=topk)
    return results, resolved_model_name


def predict_image():
    """预测单张图片类别。"""
    args = parse_args()
    results, model_name = predict_image_path(
        image_path=args.image,
        model_path=args.model_path,
        class_to_idx_path=args.class_to_idx,
        model_name=args.model,
        topk=args.topk,
    )

    print(f"模型: {model_name}")
    print(f"图片: {args.image}")
    print("Top-3 预测结果:")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['class_name']} - {item['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    predict_image()

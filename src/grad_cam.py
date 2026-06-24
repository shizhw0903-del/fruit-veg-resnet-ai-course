"""
Grad-CAM 可视化脚本。

用于解释 ResNet 模型关注图像中哪些区域，辅助展示模型可解释性。
运行示例：
python src/grad_cam.py --image data/real_test/example.jpg --model-path outputs/models/best_resnet18.pth
"""

import argparse
import os
import tempfile
from pathlib import Path

# 在受限环境中避免 Matplotlib / fontconfig 尝试写入用户主目录缓存。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fruit_veg_resnet_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from predict import build_eval_transform, load_image, load_model_for_inference, predict_pil_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """解析 Grad-CAM 参数。"""
    parser = argparse.ArgumentParser(description="水果蔬菜图像分类 Grad-CAM 可解释性可视化脚本")
    parser.add_argument("--image", type=Path, required=True, help="待解释图片路径")
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
    parser.add_argument("--target-class", default=None, help="指定解释的目标类别名称或类别索引；默认解释模型预测类别")
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/gradcam_examples", help="Grad-CAM 图片保存目录")
    return parser.parse_args()


def get_target_layer(model, model_name):
    """选择 Grad-CAM 的目标卷积层。"""
    if model_name == "resnet18":
        return model.layer4[-1].conv2
    if model_name == "resnet50":
        return model.layer4[-1].conv3
    raise ValueError("model_name 只能是 resnet18 或 resnet50")


def resolve_target_index(target_class, class_names, logits):
    """将目标类别名称或索引转换为类别索引；未指定时使用预测类别。"""
    if target_class is None:
        return int(torch.argmax(logits, dim=1).item())

    target_text = str(target_class).strip()
    if target_text.isdigit():
        target_index = int(target_text)
        if 0 <= target_index < len(class_names):
            return target_index
        raise ValueError(f"目标类别索引超出范围: {target_index}")

    if target_text in class_names:
        return class_names.index(target_text)
    raise ValueError(f"找不到目标类别: {target_text}")


def make_grad_cam(model, image, class_names, model_name, device, target_class=None, alpha=0.45):
    """生成 Grad-CAM 热力图和叠加图，返回 PIL 图片与目标类别信息。"""
    original_image = load_image(image)
    input_tensor = build_eval_transform()(original_image).unsqueeze(0).to(device)
    target_layer = get_target_layer(model, model_name)

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        """保存目标层的前向特征图。"""
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        """保存目标层输出相对于目标类别分数的梯度。"""
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    try:
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    except AttributeError:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        target_index = resolve_target_index(target_class, class_names, logits)
        target_score = logits[:, target_index].sum()
        target_score.backward()

        feature_maps = activations["value"].detach()
        gradient_maps = gradients["value"].detach()

        # Grad-CAM 权重是每个通道梯度在空间维度上的平均值。
        weights = gradient_maps.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam_tensor = F.relu(cam_tensor)
        cam_tensor = F.interpolate(cam_tensor, size=original_image.size[::-1], mode="bilinear", align_corners=False)
        cam_array = cam_tensor.squeeze().cpu().numpy()

        if cam_array.max() > cam_array.min():
            cam_array = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min())
        else:
            cam_array = np.zeros_like(cam_array)

        heatmap_array = cm.get_cmap("jet")(cam_array)[..., :3]
        original_array = np.asarray(original_image).astype(np.float32) / 255.0
        overlay_array = np.clip((1 - alpha) * original_array + alpha * heatmap_array, 0, 1)

        heatmap_image = Image.fromarray((heatmap_array * 255).astype(np.uint8))
        overlay_image = Image.fromarray((overlay_array * 255).astype(np.uint8))
        target_probability = torch.softmax(logits, dim=1)[0, target_index].detach().cpu().item()

        return {
            "target_index": target_index,
            "target_class": class_names[target_index],
            "target_probability": float(target_probability),
            "heatmap": heatmap_image,
            "overlay": overlay_image,
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_grad_cam_result(overlay_image, image_path, output_dir, model_name, target_class):
    """保存 Grad-CAM 叠加图到 outputs/gradcam_examples。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    safe_class_name = str(target_class).replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"{image_stem}_{model_name}_{safe_class_name}_gradcam.png"
    overlay_image.save(output_path)
    return output_path


def generate_grad_cam():
    """生成并保存 Grad-CAM 热力图。"""
    args = parse_args()
    model, class_names, model_name, device = load_model_for_inference(
        model_path=args.model_path,
        class_to_idx_path=args.class_to_idx,
        model_name=args.model,
    )
    result = make_grad_cam(
        model=model,
        image=args.image,
        class_names=class_names,
        model_name=model_name,
        device=device,
        target_class=args.target_class,
        alpha=args.alpha,
    )
    output_path = save_grad_cam_result(result["overlay"], args.image, args.output_dir, model_name, result["target_class"])

    top3 = predict_pil_image(load_image(args.image), model, class_names, device, topk=3)
    print(f"模型: {model_name}")
    print(f"图片: {args.image}")
    print(f"Grad-CAM 目标类别: {result['target_class']} ({result['target_probability'] * 100:.2f}%)")
    print("Top-3 预测结果:")
    for rank, item in enumerate(top3, start=1):
        print(f"{rank}. {item['class_name']} - {item['confidence'] * 100:.2f}%")
    print(f"Grad-CAM 叠加图已保存到: {output_path}")


if __name__ == "__main__":
    generate_grad_cam()

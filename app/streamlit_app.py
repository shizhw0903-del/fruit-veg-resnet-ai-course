"""
Streamlit 可视化应用入口。

用于上传水果蔬菜图片、展示预测结果、Top-3 置信度和 Grad-CAM 可解释性热力图。
运行示例：
streamlit run app/streamlit_app.py
"""

import base64
import json
import subprocess
import sys
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grad_cam import make_grad_cam
from predict import (
    build_eval_transform,
    get_device,
    load_image,
    load_model_for_inference,
)
from train_multitask import (
    FRUIT_CLASSES,
    GROUP_TO_IDX,
    VEGETABLE_CLASSES,
    MultiTaskResNet,
)


DEFAULT_MODEL_PATHS = {
    "resnet18": PROJECT_ROOT / "outputs/models/best_resnet18.pth",
    "resnet50": PROJECT_ROOT / "outputs/models/best_resnet50.pth",
}
DEFAULT_CLASS_TO_IDX_PATH = PROJECT_ROOT / "outputs/reports/class_to_idx.json"
DEFAULT_MULTITASK_MODEL_PATH = (
    PROJECT_ROOT / "outputs/models/best_multitask_resnet50.pth"
)
DEFAULT_MULTITASK_CLASS_TO_IDX_PATH = (
    PROJECT_ROOT / "outputs/reports/class_to_idx_multitask.json"
)
DEFAULT_GROUP_TO_IDX_PATH = (
    PROJECT_ROOT / "outputs/reports/group_to_idx.json"
)
MODEL_OPTIONS = {
    "普通 ResNet50 分类模型": {
        "mode": "single",
        "model_name": "resnet50",
        "model_path": DEFAULT_MODEL_PATHS["resnet50"],
    },
    "多任务 ResNet50 层级分类模型": {
        "mode": "multitask",
        "model_name": "resnet50",
        "model_path": DEFAULT_MULTITASK_MODEL_PATH,
    },
    "普通 ResNet18 分类模型": {
        "mode": "single",
        "model_name": "resnet18",
        "model_path": DEFAULT_MODEL_PATHS["resnet18"],
    },
}


class MultiTaskClassWrapper(nn.Module):
    """只返回多任务模型的 class_logits，供现有 Grad-CAM 使用。"""

    def __init__(self, multitask_model):
        super().__init__()
        self.multitask_model = multitask_model

    @property
    def layer4(self):
        """透传 ResNet 共享 backbone 的目标卷积层。"""
        return self.multitask_model.shared_feature.layer4

    def forward(self, x):
        _, class_logits = self.multitask_model(x)
        return class_logits


def inject_custom_css():
    """注入少量 CSS，形成浅色、留白充足、圆角卡片化的页面风格。"""
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f5f5f7;
            --card-bg: #ffffff;
            --primary-text: #1d1d1f;
            --secondary-text: #6e6e73;
            --muted-text: #86868b;
            --accent-blue: #0071e3;
            --border-soft: rgba(0, 0, 0, 0.08);
            --shadow-soft: 0 18px 45px rgba(0, 0, 0, 0.06);
            --shadow-light: 0 10px 28px rgba(0, 0, 0, 0.05);
        }
        .stApp {
            background: var(--page-bg);
        }
        [data-testid="stAppViewContainer"] > .main {
            background: var(--page-bg);
        }
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1120px;
            padding-top: 3.2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.78);
            border-right: 1px solid rgba(0, 0, 0, 0.06);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        h1, h2, h3, h4, h5, h6, p, label {
            color: var(--primary-text);
        }
        .hero-title {
            max-width: 940px;
            margin: 0 auto;
            color: var(--primary-text);
            font-size: clamp(2.55rem, 6vw, 4.7rem);
            font-weight: 800;
            line-height: 1.04;
            text-align: center;
            letter-spacing: 0;
        }
        .hero-subtitle {
            max-width: 760px;
            margin: 1.15rem auto 2.45rem auto;
            color: var(--secondary-text);
            font-size: clamp(1.05rem, 2vw, 1.35rem);
            line-height: 1.55;
            text-align: center;
            letter-spacing: 0;
        }
        .apple-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: var(--shadow-light);
        }
        .upload-card {
            padding: 1.35rem 1.5rem;
            margin: 0 auto 1rem auto;
        }
        .upload-title {
            color: var(--primary-text);
            font-size: 1.22rem;
            font-weight: 750;
            margin-bottom: 0.28rem;
        }
        .upload-copy {
            color: var(--secondary-text);
            font-size: 0.98rem;
            line-height: 1.55;
        }
        [data-testid="stFileUploader"] {
            margin-bottom: 1.9rem;
        }
        [data-testid="stFileUploader"] section {
            background: #fbfbfd;
            border: 1px dashed rgba(0, 0, 0, 0.18);
            border-radius: 20px;
        }
        [data-testid="stFileUploader"] button {
            border-radius: 999px;
            border-color: rgba(0, 113, 227, 0.25);
            color: var(--accent-blue);
        }
        .result-surface {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: var(--shadow-soft);
            padding: 1.6rem 1.7rem;
            margin: 0.35rem 0 1rem 0;
        }
        .image-card {
            padding: 1.35rem;
            margin: 0.35rem 0 1rem 0;
        }
        .image-card-title {
            color: var(--primary-text);
            font-size: 1.15rem;
            font-weight: 750;
            margin: 0.2rem 0 0.72rem 0;
        }
        .image-card-note {
            color: var(--secondary-text);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: -0.25rem 0 0.85rem 0;
        }
        .image-card img {
            display: block;
            width: 100%;
            max-height: 560px;
            border-radius: 19px;
            object-fit: contain;
            background: #f5f5f7;
        }
        .image-caption {
            color: var(--muted-text);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 0.72rem;
            text-align: center;
        }
        .section-title {
            color: var(--primary-text);
            font-size: 1.35rem;
            font-weight: 750;
            margin: 1rem 0 0.7rem 0;
        }
        .prediction-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: var(--shadow-soft);
            padding: 1.6rem 1.7rem;
            margin: 0.35rem 0 1rem 0;
            min-height: 245px;
            box-sizing: border-box;
        }
        .prediction-title {
            color: var(--primary-text);
            font-size: 1.35rem;
            font-weight: 750;
            margin-bottom: 1.15rem;
        }
        .prediction-label {
            color: var(--muted-text);
            font-size: 0.92rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }
        .prediction-class {
            color: var(--primary-text);
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
            line-height: 1.08;
            margin-bottom: 0.75rem;
            word-break: break-word;
        }
        .prediction-meta {
            color: var(--secondary-text);
            font-size: 1rem;
            line-height: 1.75;
        }
        .prediction-meta strong {
            color: var(--primary-text);
        }
        .multitask-prediction-card,
        .multitask-confidence-card {
            height: 300px;
            min-height: 300px;
        }
        .multitask-prediction-card {
            padding: 1.2rem 1.4rem;
        }
        .multitask-prediction-card .prediction-title {
            margin-bottom: 0.65rem;
        }
        .multitask-prediction-card .prediction-meta {
            font-size: 0.92rem;
            line-height: 1.4;
        }
        .hierarchy-result {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 0.45rem;
            margin-bottom: 0.6rem;
        }
        .hierarchy-item {
            background: #f7f7f9;
            border: 1px solid rgba(0, 0, 0, 0.055);
            border-radius: 14px;
            padding: 0.38rem 0.75rem 0.48rem 0.75rem;
            min-width: 0;
        }
        .multitask-prediction-card .prediction-label {
            margin-bottom: 0.12rem;
        }
        .hierarchy-value {
            color: var(--primary-text);
            font-size: clamp(1.25rem, 2.3vw, 1.65rem);
            font-weight: 800;
            line-height: 1.08;
            margin: 0.12rem 0 0 0;
            word-break: break-word;
        }
        .hierarchy-confidence {
            color: var(--secondary-text);
            font-size: 0.86rem;
            line-height: 1.3;
        }
        .confidence-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: var(--shadow-soft);
            padding: 1.6rem 1.7rem 1.35rem 1.7rem;
            margin: 0.35rem 0 1rem 0;
            min-height: 245px;
            box-sizing: border-box;
        }
        .st-key-uncertainty_card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: var(--shadow-soft);
            padding: 1.6rem 1.7rem 1.35rem 1.7rem;
            margin: 0.35rem 0 1rem 0;
            gap: 0.85rem;
            box-sizing: border-box;
        }
        .uncertainty-title {
            color: var(--primary-text);
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.3;
            margin: 0;
        }
        .uncertainty-scope {
            color: var(--muted-text);
            font-size: 0.92rem;
            font-weight: 650;
            line-height: 1.6;
            margin: 0.55rem 0 0.25rem 0;
        }
        .st-key-rejection_summary_card,
        .st-key-uncertainty_metrics_card {
            background: #fbfbfd;
            border: 1px solid rgba(0, 0, 0, 0.07);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            gap: 0.75rem;
            box-sizing: border-box;
        }
        .st-key-rejection_summary_card {
            margin-top: 0.15rem;
        }
        .st-key-uncertainty_metrics_card {
            margin-top: 0.1rem;
        }
        .confidence-item {
            margin: 0.75rem 0 1.05rem 0;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            color: var(--primary-text);
            font-weight: 650;
            margin-bottom: 0.4rem;
        }
        .confidence-percent {
            color: var(--accent-blue);
            white-space: nowrap;
            text-align: right;
            font-weight: 650;
        }
        .stProgress > div > div > div > div {
            background-color: var(--accent-blue);
        }
        .stProgress > div > div {
            background-color: #e8e8ed;
        }
        .progress-track {
            width: 100%;
            height: 9px;
            background: #e8e8ed;
            border-radius: 999px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent-blue);
            border-radius: 999px;
        }
        .uncertain-message {
            color: #7a4a00;
            background: #fff7e6;
            border: 1px solid #f5d58a;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
        }
        .system-copy {
            color: var(--secondary-text);
            line-height: 1.8;
        }
        .system-copy strong {
            color: var(--primary-text);
        }
        @media (max-width: 768px) {
            [data-testid="stAppViewContainer"] .block-container {
                padding-top: 2rem;
            }
            .upload-card,
            .image-card,
            .prediction-card,
            .confidence-card,
            .st-key-uncertainty_card,
            .st-key-rejection_summary_card,
            .st-key-uncertainty_metrics_card {
                border-radius: 20px;
            }
            .multitask-prediction-card,
            .multitask-confidence-card {
                height: auto;
                min-height: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_single_task_model(model_path, class_to_idx_path, model_name):
    """缓存并加载原有普通分类模型。"""
    return load_model_for_inference(
        model_path=Path(model_path),
        class_to_idx_path=Path(class_to_idx_path),
        model_name=model_name,
    )


def safe_torch_load(path, device):
    """兼容不同 PyTorch 版本的 checkpoint 加载参数。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mapping_file(path, nested_key=None):
    """读取类别映射，并兼容嵌套 JSON 格式。"""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if nested_key and nested_key in payload:
        payload = payload[nested_key]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"映射文件内容无效：{path}")
    mapping = {str(name): int(index) for name, index in payload.items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f"映射索引必须唯一且从 0 连续编号：{path}")
    return mapping


def extract_multitask_state_dict(checkpoint):
    """兼容 model_state_dict、state_dict 和直接 state_dict。"""
    if not isinstance(checkpoint, dict):
        raise ValueError("多任务 checkpoint 必须是字典或 state_dict")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("多任务模型文件中没有有效的 state_dict")
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def names_from_mapping(mapping):
    """按索引顺序恢复名称列表。"""
    return [
        name
        for name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]


@st.cache_resource(show_spinner=False)
def load_multitask_model(
    model_path,
    class_to_idx_path,
    group_to_idx_path,
    model_name,
):
    """独立加载双 head 多任务模型及其两个标签映射。"""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"未找到多任务模型文件：{model_path}。"
            "请先运行 train_multitask.py，或在高级设置中填写正确路径。"
        )

    device = get_device()
    checkpoint = safe_torch_load(model_path, device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    resolved_model_name = metadata.get("model_name", model_name)
    if resolved_model_name != model_name:
        raise ValueError(
            f"当前选择的是 {model_name}，但 checkpoint 中记录的是 "
            f"{resolved_model_name}"
        )

    checkpoint_class_mapping = metadata.get("class_to_idx")
    checkpoint_group_mapping = metadata.get("group_to_idx")
    if checkpoint_class_mapping is not None:
        checkpoint_class_mapping = {
            str(name): int(index)
            for name, index in checkpoint_class_mapping.items()
        }
        if sorted(checkpoint_class_mapping.values()) != list(
            range(len(checkpoint_class_mapping))
        ):
            raise ValueError(
                "checkpoint 中的 class_to_idx 索引必须从 0 连续编号"
            )
    if checkpoint_group_mapping is not None:
        checkpoint_group_mapping = {
            str(name): int(index)
            for name, index in checkpoint_group_mapping.items()
        }
        if sorted(checkpoint_group_mapping.values()) != list(
            range(len(checkpoint_group_mapping))
        ):
            raise ValueError(
                "checkpoint 中的 group_to_idx 索引必须从 0 连续编号"
            )

    class_to_idx = checkpoint_class_mapping
    if class_to_idx is None:
        class_to_idx = load_mapping_file(
            class_to_idx_path,
            nested_key="class_to_idx",
        )
    group_to_idx = checkpoint_group_mapping
    if group_to_idx is None:
        group_to_idx = load_mapping_file(
            group_to_idx_path,
            nested_key="group_to_idx",
        )
    if class_to_idx is None:
        raise FileNotFoundError(
            "多任务 checkpoint 未包含 class_to_idx，且未找到有效的 "
            f"类别映射文件：{class_to_idx_path}"
        )
    if group_to_idx is None:
        raise FileNotFoundError(
            "多任务 checkpoint 未包含 group_to_idx，且未找到有效的 "
            f"分组映射文件：{group_to_idx_path}"
        )
    if group_to_idx != GROUP_TO_IDX:
        raise ValueError(
            f"group_to_idx 必须为 {GROUP_TO_IDX}，实际为 {group_to_idx}"
        )
    if len(class_to_idx) != 50:
        raise ValueError(
            f"多任务模型应包含 50 个细分类，当前映射为 {len(class_to_idx)} 类"
        )

    class_names = names_from_mapping(class_to_idx)
    group_names = names_from_mapping(group_to_idx)
    for class_name in class_names:
        get_class_group(class_name)

    state_dict = extract_multitask_state_dict(checkpoint)
    class_head_weight = state_dict.get("class_head.weight")
    group_head_weight = state_dict.get("group_head.weight")
    if class_head_weight is None or group_head_weight is None:
        raise ValueError(
            "模型权重缺少 class_head 或 group_head，请确认使用的是多任务模型"
        )
    if int(class_head_weight.shape[0]) != len(class_names):
        raise ValueError(
            "class_head 输出维度与 class_to_idx 类别数量不一致"
        )
    if int(group_head_weight.shape[0]) != len(group_names):
        raise ValueError(
            "group_head 输出维度与 group_to_idx 分组数量不一致"
        )

    model = MultiTaskResNet(
        model_name=resolved_model_name,
        num_classes=len(class_names),
        num_groups=len(group_names),
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, group_names, resolved_model_name, device


def pil_to_data_uri(image):
    """将 PIL 图片转换为 HTML 可直接显示的 data URI。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def draw_image_card(title, image, caption=None, note=None):
    """以统一白底圆角卡片展示图片。"""
    title = escape(title)
    image_uri = pil_to_data_uri(image)
    note_html = f'<div class="image-card-note">{escape(note)}</div>' if note else ""
    caption_html = f'<div class="image-caption">{escape(caption)}</div>' if caption else ""
    html = (
        '<div class="result-surface image-card">'
        f'<div class="image-card-title">{title}</div>'
        f"{note_html}"
        f'<img src="{image_uri}" alt="{title}">'
        f"{caption_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def draw_prediction_card(best_result, model_name):
    """使用卡片样式展示 Top-1 预测结果。"""
    class_name = escape(best_result["class_name"])
    confidence_percent = best_result["confidence"] * 100
    model_name = escape(model_name)
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-title">识别结果</div>
            <div class="prediction-label">Top-1 预测类别</div>
            <div class="prediction-class">{class_name}</div>
            <div class="prediction-meta">
                置信度：<strong>{confidence_percent:.2f}%</strong><br>
                当前模型：<strong>{model_name}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_multitask_prediction_card(multitask_result, model_name):
    """在同一卡片中展示一级分组与二级细分类结果。"""
    best_result = multitask_result["top3_results"][0]
    group_name = escape(multitask_result["group_name"])
    class_name = escape(best_result["class_name"])
    group_confidence = multitask_result["group_confidence"] * 100
    class_confidence = best_result["confidence"] * 100
    model_name = escape(model_name)
    st.markdown(
        f"""
        <div class="prediction-card multitask-prediction-card">
            <div class="prediction-title">层级识别结果</div>
            <div class="hierarchy-result">
                <div class="hierarchy-item">
                    <div class="prediction-label">一级类别</div>
                    <div class="hierarchy-confidence">
                        置信度：<strong>{group_confidence:.2f}%</strong>
                    </div>
                    <div class="hierarchy-value">{group_name}</div>
                </div>
                <div class="hierarchy-item">
                    <div class="prediction-label">二级类别</div>
                    <div class="hierarchy-confidence">
                        置信度：<strong>{class_confidence:.2f}%</strong>
                    </div>
                    <div class="hierarchy-value">{class_name}</div>
                </div>
            </div>
            <div class="prediction-meta">
                当前模型：<strong>{model_name} multitask</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_confidence_progress(top3_results, title="Top-3 置信度"):
    """使用统一白底圆角卡片展示 Top-3 预测置信度。"""
    card_class = (
        "confidence-card multitask-confidence-card"
        if title == "二级类别 Top-3 置信度"
        else "confidence-card"
    )
    rows = []
    for rank, item in enumerate(top3_results, start=1):
        confidence_percent = item["confidence"] * 100
        progress_width = min(max(confidence_percent, 0.0), 100.0)
        class_name = escape(item["class_name"])
        rows.append(
            '<div class="confidence-item">'
            '<div class="confidence-label">'
            f"<span>Top-{rank}：{class_name}</span>"
            f'<span class="confidence-percent">{confidence_percent:.2f}%</span>'
            "</div>"
            '<div class="progress-track">'
            f'<div class="progress-fill" style="width: {progress_width:.2f}%;"></div>'
            "</div>"
            "</div>"
        )
    html = (
        f'<div class="{card_class}">'
        f'<div class="section-title" style="margin-top: 0;">{escape(title)}</div>'
        f"{''.join(rows)}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def compute_uncertainty_metrics(probs):
    """根据完整 softmax 分布计算 Top-1、Top-2、margin 和熵。"""
    probabilities = probs.detach().flatten()
    if probabilities.numel() < 2:
        raise ValueError("至少需要两个类别才能计算置信度 margin")

    top_probs, _ = torch.topk(probabilities, k=2)
    top1_conf = float(top_probs[0].item())
    top2_conf = float(top_probs[1].item())
    margin = top1_conf - top2_conf
    entropy = float(
        -(probabilities * torch.log(probabilities + 1e-12)).sum().item()
    )
    return {
        "top1_conf": top1_conf,
        "top2_conf": top2_conf,
        "margin": margin,
        "entropy": entropy,
    }


def get_confidence_level(top1_conf, margin):
    """使用 Top-1 概率和前两名差值给出启发式可信度等级。"""
    if top1_conf >= 0.75 and margin >= 0.25:
        return "高可信"
    if top1_conf >= 0.55 and margin >= 0.15:
        return "中可信"
    return "低可信"


def build_top_results(probs, class_names, topk=3):
    """将概率分布转换为统一的 Top-K 结果结构。"""
    k = min(topk, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)
    results = []
    for probability, index in zip(
        top_probs.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        results.append(
            {
                "class_name": class_names[index],
                "class_index": index,
                "confidence": float(probability),
            }
        )
    return results


def get_class_group(class_name):
    """返回细分类所属的一级分组，不修正任何类别拼写。"""
    if class_name in FRUIT_CLASSES:
        return "fruit"
    if class_name in VEGETABLE_CLASSES:
        return "vegetable"
    raise ValueError(
        f"类别“{class_name}”未包含在 fruit/vegetable 分组规则中，"
        "请检查 class_to_idx 与训练数据文件夹名。"
    )


def predict_single_task(image, model, class_names, device, topk=3):
    """执行普通分类模型推理并计算拒识指标。"""
    image = load_image(image)
    input_tensor = build_eval_transform()(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_results = build_top_results(probs, class_names, topk)
    metrics = compute_uncertainty_metrics(probs)
    metrics["level"] = get_confidence_level(
        metrics["top1_conf"],
        metrics["margin"],
    )
    return top_results, metrics


def predict_multitask(
    image,
    model,
    class_names,
    group_names,
    device,
    topk=3,
):
    """执行双 head 推理，并检查一级与二级类别是否一致。"""
    image = load_image(image)
    input_tensor = build_eval_transform()(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        group_logits, class_logits = model(input_tensor)
        group_probs = torch.softmax(group_logits, dim=1)[0]
        class_probs = torch.softmax(class_logits, dim=1)[0]

    top_results = build_top_results(class_probs, class_names, topk)
    metrics = compute_uncertainty_metrics(class_probs)
    metrics["level"] = get_confidence_level(
        metrics["top1_conf"],
        metrics["margin"],
    )

    group_confidence, group_index = torch.max(group_probs, dim=0)
    predicted_group = group_names[int(group_index.item())]
    expected_group = get_class_group(top_results[0]["class_name"])
    return {
        "top3_results": top_results,
        "metrics": metrics,
        "group_name": predicted_group,
        "group_index": int(group_index.item()),
        "group_confidence": float(group_confidence.item()),
        "expected_group": expected_group,
        "group_consistent": predicted_group == expected_group,
    }


def display_prediction_with_rejection(metrics, consistency_warning=None):
    """展示可信度等级、拒识提醒和不确定性指标。"""
    with st.container(border=True, key="uncertainty_card"):
        st.markdown(
            """
            <div class="uncertainty-title">可信度与未知类别提醒</div>
            <div class="uncertainty-scope">
                当前模型只覆盖训练集中的 50 个水果蔬菜类别，无法真正识别训练集之外的未知类别。
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True, key="rejection_summary_card"):
            level = metrics["level"]
            if level == "高可信":
                st.success("可信度等级：高可信")
            elif level == "中可信":
                st.info("可信度等级：中可信")
                st.info(
                    "该图片可能不属于当前系统支持的 50 类水果蔬菜类别，"
                    "建议谨慎参考结果。"
                )
                st.info(
                    "模型对该图片判断不够确定，"
                    "可能存在相似类别混淆或未知类别输入。"
                )
            else:
                st.error("可信度等级：低可信")
                st.error(
                    "该图片可能不属于当前系统支持的 50 类水果蔬菜类别，"
                    "建议谨慎参考结果。"
                )
                st.error(
                    "模型对该图片判断不够确定，"
                    "可能存在相似类别混淆或未知类别输入。"
                )

            if consistency_warning:
                if level == "低可信":
                    st.error(consistency_warning)
                elif level == "中可信":
                    st.info(consistency_warning)
                else:
                    st.warning(consistency_warning)

        with st.container(border=True, key="uncertainty_metrics_card"):
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Top-1 置信度",
                f"{metrics['top1_conf'] * 100:.2f}%",
            )
            metric_columns[1].metric(
                "Top-1 / Top-2 差值",
                f"{metrics['margin']:.4f}",
            )
            metric_columns[2].metric(
                "预测分布熵",
                f"{metrics['entropy']:.4f}",
            )


def draw_system_info():
    """展示课程项目说明。"""
    with st.expander("系统说明", expanded=False):
        st.markdown(
            """
            <div class="system-copy">
            <strong>方法：</strong>普通 ResNet 分类与水果/蔬菜层级多任务分类；<br>
            <strong>框架：</strong>PyTorch + Streamlit；<br>
            <strong>功能：</strong>一级/二级分类、Top-3 置信度、低置信度提醒、Grad-CAM 可解释性；<br>
            <strong>项目价值：</strong>帮助用户识别水果蔬菜类别，并提升模型预测过程的可解释性。
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    """启动 Streamlit 应用。"""
    st.set_page_config(page_title="水果蔬菜智能识别系统", layout="wide")
    inject_custom_css()

    st.markdown('<div class="hero-title">基于 ResNet 迁移学习的水果蔬菜智能识别系统</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-subtitle">
        上传一张水果或蔬菜图片，系统将自动识别类别，并通过 Grad-CAM 展示模型关注区域。
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("模型设置")
        selected_option = st.selectbox(
            "模型类型",
            options=list(MODEL_OPTIONS),
            index=0,
        )
        option_config = MODEL_OPTIONS[selected_option]
        model_mode = option_config["mode"]
        selected_model = option_config["model_name"]
        default_model_path = option_config["model_path"]
        with st.expander("高级设置", expanded=False):
            model_path = Path(
                st.text_input(
                    "模型权重路径",
                    value=str(default_model_path),
                    key=f"model_path_{model_mode}_{selected_model}",
                )
            )
            if model_mode == "multitask":
                class_to_idx_path = Path(
                    st.text_input(
                        "多任务类别映射路径",
                        value=str(DEFAULT_MULTITASK_CLASS_TO_IDX_PATH),
                    )
                )
                group_to_idx_path = Path(
                    st.text_input(
                        "一级分组映射路径",
                        value=str(DEFAULT_GROUP_TO_IDX_PATH),
                    )
                )
            else:
                class_to_idx_path = Path(
                    st.text_input(
                        "类别映射路径",
                        value=str(DEFAULT_CLASS_TO_IDX_PATH),
                        key=f"class_mapping_{selected_model}",
                    )
                )
                group_to_idx_path = None
            alpha = st.slider("Grad-CAM 透明度", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    st.markdown(
        """
        <div class="apple-card upload-card">
            <div class="upload-title">上传图片</div>
            <div class="upload-copy">请上传主体清晰、光照较好的水果或蔬菜图片。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("上传水果或蔬菜图片", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    if uploaded_file is None:
        draw_system_info()
        return

    if not model_path.exists():
        if model_mode == "multitask":
            st.error(
                f"未找到多任务模型文件：{model_path}。"
                "请先训练多任务模型，或在高级设置中填写正确路径。"
            )
        else:
            st.error(f"未找到普通分类模型文件：{model_path}")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"图片读取失败：{exc}")
        st.stop()

    try:
        with st.spinner("正在加载模型..."):
            if model_mode == "multitask":
                (
                    model,
                    class_names,
                    group_names,
                    resolved_model_name,
                    device,
                ) = load_multitask_model(
                    str(model_path),
                    str(class_to_idx_path),
                    str(group_to_idx_path),
                    selected_model,
                )
            else:
                (
                    model,
                    class_names,
                    resolved_model_name,
                    device,
                ) = load_single_task_model(
                    str(model_path),
                    str(class_to_idx_path),
                    selected_model,
                )
                group_names = None
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        st.stop()

    with st.spinner("正在预测..."):
        if model_mode == "multitask":
            multitask_result = predict_multitask(
                image,
                model,
                class_names,
                group_names,
                device,
                topk=3,
            )
            top3_results = multitask_result["top3_results"]
            uncertainty_metrics = multitask_result["metrics"]
        else:
            multitask_result = None
            top3_results, uncertainty_metrics = predict_single_task(
                image,
                model,
                class_names,
                device,
                topk=3,
            )
        best_result = top3_results[0]

    with st.spinner("正在生成 Grad-CAM..."):
        try:
            grad_cam_model = (
                MultiTaskClassWrapper(model).to(device).eval()
                if model_mode == "multitask"
                else model
            )
            grad_cam_result = make_grad_cam(
                model=grad_cam_model,
                image=image,
                class_names=class_names,
                model_name=resolved_model_name,
                device=device,
                target_class=best_result["class_index"],
                alpha=alpha,
            )
        except Exception as exc:
            st.error(f"Grad-CAM 生成失败：{exc}")
            st.stop()

    column_widths = (
        [1.08, 0.97]
        if model_mode == "multitask"
        else [1.70, 1.00]
    )
    result_col, top3_col = st.columns(column_widths, gap="large")
    with result_col:
        if model_mode == "multitask":
            draw_multitask_prediction_card(
                multitask_result,
                resolved_model_name,
            )
        else:
            draw_prediction_card(best_result, resolved_model_name)

    with top3_col:
        draw_confidence_progress(
            top3_results,
            title=(
                "二级类别 Top-3 置信度"
                if model_mode == "multitask"
                else "Top-3 置信度"
            ),
        )

    consistency_warning = None
    if model_mode == "multitask" and not multitask_result["group_consistent"]:
        consistency_warning = (
            "一级分类与二级分类结果不一致，说明模型判断存在不确定性，"
            "建议谨慎参考。"
        )
    display_prediction_with_rejection(
        uncertainty_metrics,
        consistency_warning=consistency_warning,
    )

    image_col, gradcam_col = st.columns(2, gap="large")
    with image_col:
        draw_image_card("原图", image, caption="用户上传的待识别图片")

    with gradcam_col:
        draw_image_card(
            "Grad-CAM 热力图",
            grad_cam_result["overlay"],
            caption=f"模型关注区域：{grad_cam_result['target_class']} ({grad_cam_result['target_probability'] * 100:.2f}%)",
            note="颜色越明显表示模型关注程度越高。",
        )

    draw_system_info()


def is_running_with_streamlit():
    """判断当前脚本是否由 streamlit run 启动。"""
    try:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except ImportError:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def launch_with_streamlit():
    """直接 python 运行时，自动转为 streamlit run，避免 bare mode 警告刷屏。"""
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        str(script_path),
        *sys.argv[1:],
    ]
    print("检测到当前不是 Streamlit 运行上下文，正在自动启动：")
    print(" ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    if is_running_with_streamlit():
        main()
    else:
        raise SystemExit(launch_with_streamlit())

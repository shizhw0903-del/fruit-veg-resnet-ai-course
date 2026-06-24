# fruit_veg_resnet

基于 PyTorch 和 ResNet18 / ResNet50 迁移学习的水果蔬菜图像分类系统。

本项目适合作为大学《人工智能原理》期末项目，支持 ResNet18 / ResNet50 迁移学习训练、测试集评估、单图预测、Grad-CAM 可解释性分析和 Streamlit 可视化应用。

## 项目结构

```text
fruit_veg_resnet/
├── data/
│   ├── train/              # 训练集，按类别子文件夹存放图片
│   ├── val/                # 验证集，按类别子文件夹存放图片
│   ├── test/               # 测试集，按类别子文件夹存放图片
│   └── real_test/          # 真实场景或自定义预测图片
├── src/
│   ├── dataset.py          # 数据集读取和 DataLoader 构建
│   ├── train.py            # ResNet 迁移学习训练脚本
│   ├── evaluate.py         # 测试集评估脚本
│   ├── predict.py          # 单张图片预测脚本
│   ├── grad_cam.py         # Grad-CAM 可解释性可视化
│   └── utils.py            # 通用工具函数
├── app/
│   └── streamlit_app.py    # Streamlit Web 演示应用
├── outputs/
│   ├── models/             # 保存训练好的模型权重
│   ├── figures/            # 保存训练曲线、混淆矩阵等图片
│   ├── reports/            # 保存分类报告和实验记录
│   └── gradcam_examples/   # 保存 Grad-CAM 示例结果
├── requirements.txt
└── README.md
```

## 数据集格式

建议使用 `torchvision.datasets.ImageFolder` 兼容的数据组织方式：

```text
data/train/apple/*.jpg
data/train/banana/*.jpg
data/val/apple/*.jpg
data/val/banana/*.jpg
data/test/apple/*.jpg
data/test/banana/*.jpg
```

每个类别对应一个子文件夹，子文件夹名称即类别名称。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练与评估

默认训练 ResNet18：

```bash
python src/train.py --model resnet18 --epochs 10 --batch-size 32
```

训练完成后会保存：

```text
outputs/models/best_resnet18.pth
outputs/figures/training_curve_resnet18.png
outputs/reports/class_to_idx.json
```

评估 ResNet18：

```bash
python src/evaluate.py --model resnet18 --model-path outputs/models/best_resnet18.pth
```

评估完成后会保存：

```text
outputs/reports/classification_report_resnet18.txt
outputs/figures/confusion_matrix_resnet18.png
```

训练 ResNet50：

```bash
python src/train.py --model resnet50 --epochs 10 --batch-size 16
```

训练完成后会保存：

```text
outputs/models/best_resnet50.pth
outputs/figures/training_curve_resnet50.png
outputs/reports/class_to_idx.json
```

评估 ResNet50：

```bash
python src/evaluate.py --model resnet50 --model-path outputs/models/best_resnet50.pth
```

评估完成后会保存：

```text
outputs/reports/classification_report_resnet50.txt
outputs/figures/confusion_matrix_resnet50.png
```

如果电脑显存或内存不足，可以把 `--batch-size` 调小，例如：

```bash
python src/train.py --model resnet50 --epochs 10 --batch-size 8
python src/evaluate.py --model resnet50 --model-path outputs/models/best_resnet50.pth --batch-size 8
```

## ResNet18 / ResNet50 对比流程

1. 分别训练 ResNet18 和 ResNet50。
2. 分别运行测试集评估命令。
3. 对比 `outputs/reports/classification_report_resnet18.txt` 和 `outputs/reports/classification_report_resnet50.txt` 中的 Accuracy、Precision、Recall、F1-score。
4. 对比 `outputs/figures/training_curve_resnet18.png`、`outputs/figures/training_curve_resnet50.png` 以及两个混淆矩阵图片。

## Streamlit 展示

```bash
streamlit run app/streamlit_app.py
```

## 后续可扩展方向

- 增加更多数据增强策略。
- 尝试解冻更多 ResNet 层进行微调。
- 对比不同学习率、batch size 和训练轮数对结果的影响。
- 收集真实拍摄图片放入 `data/real_test/` 做泛化测试。

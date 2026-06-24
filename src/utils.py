"""
通用工具函数模块。

这里放置项目中重复使用的函数，例如随机种子设置、路径创建、日志记录和模型保存等。
"""


def set_seed(seed=42):
    """设置随机种子，保证实验尽可能可复现。"""
    # TODO: 设置 random、numpy、torch 的随机种子
    pass


def ensure_dir(path):
    """确保指定目录存在。"""
    # TODO: 使用 pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    pass


def save_json(data, path):
    """保存 JSON 文件，例如类别索引映射或实验配置。"""
    # TODO: 将字典数据保存为 JSON 文件，便于复现实验
    pass


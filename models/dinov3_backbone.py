import os
import sys
sys.path.insert(0, "/data/truenas_B2/yyi/dinov3_pretrain")

import torch
from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg

def build_dinov3_backbone(
    cfg_path: str,
    weights_path: str,
    device_id: int = 0,
):
    """
    使用 DINOv3 官方 build_model_from_cfg 创建 teacher backbone
    """

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    # 1. 读取配置
    cfg = OmegaConf.load(cfg_path)

    # 2. 构建 teacher backbone（官方逻辑）
    backbone, embed_dim = build_model_from_cfg(
        cfg,
        only_teacher=True,
    )

    # 3. 延迟初始化 + 放到 GPU（FSDP 兼容）
    backbone.to_empty(device=device)

    # 4. 加载 checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["teacher"]

    # 5. 只取 backbone 权重
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    msg = backbone.load_state_dict(backbone_state, strict=False)

    print(
        f"[DINOv3] Loaded backbone | "
        f"Missing: {len(msg.missing_keys)}, "
        f"Unexpected: {len(msg.unexpected_keys)}"
    )

    backbone.eval()

    return backbone, embed_dim, device

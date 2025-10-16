from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Tuple

from .constants import SAMPLE_RATE
from .inference import Inference
from .utils import to_local_average_cents


def compute_salience(
    audio: np.ndarray,
    model,
    hop_length_ms: int = 20,
    seg_seconds: float = 2.56,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    计算声部显著性矩阵（salience）。

    参数:
      audio: 1D numpy 数组，采样率需为 SAMPLE_RATE。
      model: 已加载好的 PyTorch 模型（如 rmvpe.E2E 实例）。
      hop_length_ms: 帧移（毫秒）。
      seg_seconds: 分割段长度（秒），用于批推理，默认 2.56。
      batch_size: 推理批大小。
      device: torch.device；若为 None 则自动选 cuda 或 cpu。

    返回:
      salience: 形如 [T, 360] 的 numpy 数组。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert audio.ndim == 1, "audio 必须是一维 numpy 数组"

    hop_samples = int(hop_length_ms / 1000 * SAMPLE_RATE)
    seg_len = int(seg_seconds * SAMPLE_RATE)
    seg_frames = seg_len // hop_samples + 1

    model = model.to(device).eval()

    audio_t = torch.from_numpy(audio).float().to(device)
    infer = Inference(model, seg_len, seg_frames, hop_samples, batch_size, device)
    _, salience_t = infer.inference(audio_t)
    return salience_t.detach().cpu().numpy()


def extract_cents(salience: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """将显著性矩阵转换为每帧的音高（美分，0 表示非鸣音）。"""
    return to_local_average_cents(salience, None, threshold)


def extract_melody(
    audio: np.ndarray,
    model,
    hop_length_ms: int = 20,
    seg_seconds: float = 2.56,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    return_salience: bool = False,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    端到端提取旋律：返回每帧音高（美分，0 表示非鸣音）。

    返回:
      cents: [T] numpy 数组，单位为美分；0 表示无声。
      salience(可选): [T, 360] 显著性矩阵。
    """
    salience = compute_salience(
        audio=audio,
        model=model,
        hop_length_ms=hop_length_ms,
        seg_seconds=seg_seconds,
        batch_size=batch_size,
        device=device,
    )
    cents = extract_cents(salience, threshold)
    if return_salience:
        return cents, salience
    return cents, None


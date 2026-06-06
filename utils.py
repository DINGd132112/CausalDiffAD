"""
utils.py
========
通用工具：seed / logger / checkpoint / score normalization / threshold / plotting / EMA
"""
import os
import sys
import json
import random
import logging
import datetime
from copy import deepcopy
from typing import Optional, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Seed
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_dir: str, filename: Optional[str] = None) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复 handler
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    ema=None, cfg=None, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "config": cfg.to_json() if cfg is not None else None,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, ema=None,
                    map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    return state.get("extra", {})


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class EMA:
    """简单 EMA：在每次 optimizer.step() 后调用 update()。"""
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        """临时把 EMA 权重套到 model 上（评估前调用）。"""
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        msd = model.state_dict()
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])

    def restore(self, model):
        if not self.backup:
            return
        msd = model.state_dict()
        for k, v in self.backup.items():
            msd[k].copy_(v)
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


# --------------------------------------------------------------------------- #
# Score normalization
# --------------------------------------------------------------------------- #
def normalize_scores(scores: np.ndarray, mode: str = "robust",
                     ref: Optional[np.ndarray] = None) -> np.ndarray:
    """
    scores: [N] 待归一化分数
    ref   : 用 ref 的统计量来归一化 scores（避免测试集统计泄漏）
    """
    s = scores.astype(np.float64)
    r = s if ref is None else ref.astype(np.float64)
    if mode == "minmax":
        mn, mx = r.min(), r.max()
        return (s - mn) / (mx - mn + 1e-12)
    elif mode == "robust":
        med = np.median(r)
        iqr = np.percentile(r, 75) - np.percentile(r, 25) + 1e-12
        return (s - med) / iqr
    else:
        return s


def fuse_scores(diff_score: np.ndarray, causal_score: np.ndarray,
                alpha: float, beta: float, mode: str = "robust",
                ref_diff: Optional[np.ndarray] = None,
                ref_causal: Optional[np.ndarray] = None) -> np.ndarray:
    """加权融合两路分数。可传入参考分布（如 val 分数）做归一化。"""
    d = normalize_scores(diff_score, mode=mode, ref=ref_diff)
    c = normalize_scores(causal_score, mode=mode, ref=ref_causal)
    return alpha * d + beta * c


# --------------------------------------------------------------------------- #
# Threshold search & PA
# --------------------------------------------------------------------------- #
def point_adjustment(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    Point Adjustment: 若一段 ground-truth 异常区间内有任一点被预测为异常，
    则整段视为预测为异常。
    """
    pred = pred.copy().astype(np.int32)
    label = label.astype(np.int32)
    anomaly_state = False
    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # 向前回填
            for j in range(i, -1, -1):
                if label[j] == 0:
                    break
                pred[j] = 1
            # 向后填充
            for j in range(i, len(label)):
                if label[j] == 0:
                    break
                pred[j] = 1
        elif label[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_scores(scores: np.ndarray, labels: np.ndarray, threshold: float,
                save_path: str, title: str = "Anomaly Scores"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(scores, lw=0.6, label="score")
    ax.axhline(threshold, color="red", ls="--", lw=1.0, label=f"thr={threshold:.4f}")
    # 标出真异常区间
    in_seg = False
    start = 0
    for i, y in enumerate(labels):
        if y == 1 and not in_seg:
            start = i
            in_seg = True
        elif y == 0 and in_seg:
            ax.axvspan(start, i, color="orange", alpha=0.25)
            in_seg = False
    if in_seg:
        ax.axvspan(start, len(labels), color="orange", alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# JSON-safe dump
# --------------------------------------------------------------------------- #
def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def save_results(results: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2)


def save_results_markdown(results: Dict[str, Any], path: str, title: str = "Results"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# {title}", ""]
    def _dump(d, depth=0):
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{'#' * (depth + 2)} {k}")
                _dump(v, depth + 1)
            else:
                lines.append(f"- **{k}**: {v}")
    _dump(results)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

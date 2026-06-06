"""
evaluator.py
============
指标 & 阈值搜索。
所有评估同时输出 raw 与 PA 版本。
"""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)
from utils import point_adjustment


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _f1_set(label: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        label, pred, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def auc_metrics(label: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(label, score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(label, score))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# threshold search
# --------------------------------------------------------------------------- #
def find_best_f1_threshold(score: np.ndarray, label: np.ndarray,
                           n_grid: int = 200, use_pa: bool = False) -> Tuple[float, Dict[str, float]]:
    """在 [min, max] 上等距网格搜索使 F1 最高的阈值。"""
    lo, hi = float(np.min(score)), float(np.max(score))
    if hi - lo < 1e-12:
        return lo, {"precision": 0., "recall": 0., "f1": 0.}
    grid = np.linspace(lo, hi, n_grid)
    best_thr, best = grid[0], {"precision": 0., "recall": 0., "f1": -1.0}
    for thr in grid:
        pred = (score > thr).astype(np.int32)
        if use_pa:
            pred = point_adjustment(pred, label)
        m = _f1_set(label, pred)
        if m["f1"] > best["f1"]:
            best = m
            best_thr = float(thr)
    return best_thr, best


def quantile_threshold(score: np.ndarray, q: float) -> float:
    return float(np.quantile(score, q))


# --------------------------------------------------------------------------- #
# 顶层 evaluate
# --------------------------------------------------------------------------- #
def evaluate(score: np.ndarray, label: np.ndarray,
             threshold_mode: str = "best_f1",
             threshold_quantile: float = 0.99,
             val_score: np.ndarray = None) -> Dict[str, Dict[str, float]]:
    """
    返回:
        {
          'raw' : {threshold, precision, recall, f1, roc_auc, pr_auc},
          'pa'  : {threshold, precision, recall, f1, roc_auc, pr_auc(PA)}
        }
    PA: Point Adjustment
    """
    assert len(score) == len(label), f"len mismatch: {len(score)} vs {len(label)}"
    label = label.astype(np.int32)

    # AUC（与阈值无关，PA 版本对预测做调整后再求）
    auc_raw = auc_metrics(label, score)

    # ---- 选阈值 ----
    if threshold_mode == "best_f1":
        thr_raw, _ = find_best_f1_threshold(score, label, use_pa=False)
        thr_pa, _ = find_best_f1_threshold(score, label, use_pa=True)
    elif threshold_mode == "quantile":
        thr_raw = quantile_threshold(score, threshold_quantile)
        thr_pa = thr_raw
    elif threshold_mode == "val_quantile":
        assert val_score is not None, "val_quantile 需要 val_score"
        thr_raw = quantile_threshold(val_score, threshold_quantile)
        thr_pa = thr_raw
    else:
        raise ValueError(threshold_mode)

    # raw
    pred_raw = (score > thr_raw).astype(np.int32)
    m_raw = _f1_set(label, pred_raw)
    raw = {"threshold": float(thr_raw), **m_raw, **auc_raw}

    # PA
    pred_pa = (score > thr_pa).astype(np.int32)
    pred_pa = point_adjustment(pred_pa, label)
    m_pa = _f1_set(label, pred_pa)
    # PA-AUC: 把预测做PA后形成"PA-score" --- 仍报告原始 AUC，因为AUC是阈值无关的；
    # 这里额外报告"以PA后的硬预测来近似计算"的P/R/F1
    pa = {"threshold": float(thr_pa), **m_pa, **auc_raw}

    return {"raw": raw, "pa": pa}

"""
data_loader.py
==============
数据读取与窗口化。
约定数据集目录结构: data/{dataset}/{train.csv, test.csv, test_label.csv}
"""
import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# CSV 读取
# --------------------------------------------------------------------------- #
def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 列名 strip
    df.columns = [str(c).strip() for c in df.columns]
    # 丢掉明显的时间列
    for tcol in ["timestamp", "Timestamp", "time", "Time", "datetime", "DateTime"]:
        if tcol in df.columns:
            df = df.drop(columns=[tcol])
    return df


def _coerce_numeric(df: pd.DataFrame, fillna: str = "ffill") -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    if fillna == "ffill":
        df = df.ffill().bfill().fillna(0.0)
    elif fillna == "mean":
        df = df.fillna(df.mean()).fillna(0.0)
    else:
        df = df.fillna(0.0)
    return df


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("label", "labels", "attack", "anomaly", "y"):
            return c
    return None


def load_dataset(data_root: str, dataset: str,
                 fillna: str = "ffill") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    返回:
        train: [T_tr, C]   无标签
        test : [T_te, C]
        test_label: [T_te]  0/1
        feature_names: list[str]
    """
    dpath = os.path.join(data_root, dataset)
    train = _read_csv(os.path.join(dpath, "train.csv"))
    test = _read_csv(os.path.join(dpath, "test.csv"))

    # 若 train 内含 label 列 -> 去掉；test 内含 label 列 -> 提取
    train_label_col = _detect_label_column(train)
    if train_label_col is not None:
        train = train.drop(columns=[train_label_col])
    test_label_col = _detect_label_column(test)
    test_label_inline = None
    if test_label_col is not None:
        test_label_inline = test[test_label_col].values.astype(np.float32)
        test = test.drop(columns=[test_label_col])

    # test_label.csv（如果有）
    tl_path = os.path.join(dpath, "test_label.csv")
    if os.path.exists(tl_path):
        tl = _read_csv(tl_path)
        # 取第一列为标签
        test_label = tl.iloc[:, 0].values.astype(np.float32)
    elif test_label_inline is not None:
        test_label = test_label_inline
    else:
        raise FileNotFoundError(
            f"找不到 {tl_path}，且 test.csv 中无 label 列。"
        )

    # 对齐特征列：以 train 列为准
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    train = _coerce_numeric(train, fillna)
    test = _coerce_numeric(test, fillna)

    # 丢掉训练集上的常数列与几乎常数列（std 太小会被 MinMax 放大为 spike）
    std = train.std(axis=0).values
    keep = std > 1e-4
    if not keep.all():
        dropped = [c for c, k in zip(train.columns, keep) if not k]
        print(f"[data_loader] dropping {len(dropped)} near-constant columns: {dropped[:10]}...")
    train = train.loc[:, train.columns[keep]]
    test = test.loc[:, test.columns[keep]]
    feature_names = list(train.columns)

    train_arr = train.values.astype(np.float32)
    test_arr = test.values.astype(np.float32)
    # 标签长度对齐 test
    if len(test_label) != len(test_arr):
        m = min(len(test_label), len(test_arr))
        test_label = test_label[:m]
        test_arr = test_arr[:m]
    return train_arr, test_arr, test_label, feature_names


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class Scaler:
    """train-only fit; 应用到 train/val/test。"""
    def __init__(self, mode: str = "minmax"):
        self.mode = mode
        self.a = None
        self.b = None

    def fit(self, x: np.ndarray):
        if self.mode == "minmax":
            self.a = x.min(axis=0)
            self.b = x.max(axis=0) - x.min(axis=0)
            self.b = np.where(self.b < 1e-8, 1.0, self.b)
        elif self.mode == "standard":
            self.a = x.mean(axis=0)
            self.b = x.std(axis=0)
            self.b = np.where(self.b < 1e-8, 1.0, self.b)
        else:
            raise ValueError(self.mode)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.a) / self.b).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# --------------------------------------------------------------------------- #
# Sliding window
# --------------------------------------------------------------------------- #
def make_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """x: [T, C] -> [N, W, C]"""
    T = x.shape[0]
    if T < window:
        pad = np.zeros((window - T, x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
        T = window
    n = (T - window) // stride + 1
    idx = (np.arange(n)[:, None] * stride) + np.arange(window)[None, :]
    return x[idx]  # [N, W, C]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        # windows: [N, W, C] float32
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.windows[idx]
        return self.windows[idx], self.labels[idx]


# --------------------------------------------------------------------------- #
# 顶层 builder
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg, dataset_name: Optional[str] = None,
                      train_ratio: float = 1.0,
                      fast_sample_ratio: Optional[float] = None):
    """
    返回 dict:
        train_loader, val_loader, test_loader,
        n_features, scaler, raw_test, raw_test_label
    raw_test_label: [T_te]  原始（未窗口化）测试标签
    """
    dname = dataset_name or cfg.dataset
    train_raw, test_raw, test_label, feat_names = load_dataset(
        cfg.data_root, dname, fillna=cfg.fillna
    )

    # 鲁棒性：截取训练集前 train_ratio 比例
    if train_ratio < 1.0:
        cut = max(int(len(train_raw) * train_ratio), cfg.window_size * 2)
        train_raw = train_raw[:cut]

    # 快速实验：等差采样
    if fast_sample_ratio is not None and fast_sample_ratio < 1.0:
        k = max(int(1.0 / fast_sample_ratio), 1)
        train_raw = train_raw[::k]

    # 划 val
    val_cut = int(len(train_raw) * (1.0 - cfg.val_ratio))
    val_cut = max(val_cut, cfg.window_size * 2)
    val_cut = min(val_cut, len(train_raw) - cfg.window_size)
    train_part = train_raw[:val_cut]
    val_part = train_raw[val_cut:]

    scaler = Scaler(cfg.normalize).fit(train_part)
    train_s = scaler.transform(train_part)
    val_s = scaler.transform(val_part)
    test_s = scaler.transform(test_raw)

    train_w = make_windows(train_s, cfg.window_size, cfg.stride)
    val_w = make_windows(val_s, cfg.window_size, cfg.stride)
    test_w = make_windows(test_s, cfg.window_size, cfg.test_stride)

    n_features = train_s.shape[1]

    train_ds = WindowDataset(train_w)
    val_ds = WindowDataset(val_w)
    test_ds = WindowDataset(test_w)

    pin = cfg.pin_memory and torch.cuda.is_available()
    persistent = cfg.persistent_workers and cfg.num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin,
                              persistent_workers=persistent, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin,
                            persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin,
                             persistent_workers=persistent)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_features": n_features,
        "scaler": scaler,
        "feature_names": feat_names,
        "raw_test": test_s,
        "raw_test_label": test_label,
        "test_windows": test_w,
        "stride": cfg.test_stride,
        "window_size": cfg.window_size,
    }


def windows_to_pointwise(window_scores: np.ndarray, window: int, stride: int,
                         total_len: int) -> np.ndarray:
    """
    将窗口级分数 [N, W] 还原为点级分数 [total_len]。
    重叠位置取平均。
    """
    out = np.zeros(total_len, dtype=np.float64)
    cnt = np.zeros(total_len, dtype=np.float64)
    n = window_scores.shape[0]
    for i in range(n):
        s = i * stride
        e = s + window
        if e > total_len:
            e = total_len
            w = e - s
            out[s:e] += window_scores[i, :w]
            cnt[s:e] += 1.0
        else:
            out[s:e] += window_scores[i]
            cnt[s:e] += 1.0
    cnt = np.where(cnt < 1e-8, 1.0, cnt)
    return out / cnt

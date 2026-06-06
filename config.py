"""
config.py
=========
所有超参数、路径、消融开关集中于此。
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os
import json


@dataclass
class Config:
    # ============ 路径 ============
    data_root: str = "data"
    ckpt_root: str = "checkpoints"
    log_root: str = "logs"
    result_root: str = "results"
    dataset: str = "SWaT"           # 单数据集名；当 dataset == "all" 时遍历 data_root 下所有子目录
    all_datasets: List[str] = field(default_factory=lambda: ["SWaT", "MSL", "SMAP", "PSM", "SMD"])

    # ============ 数据 ============
    window_size: int = 32           # W
    stride: int = 2                 # 训练滑窗步长
    test_stride: int = 1            # 测试时使用步长1（点级评估）
    val_ratio: float = 0.2          # 从 train 末尾切出验证集
    normalize: str = "standard"       # 'minmax' | 'standard'
    fillna: str = "ffill"           # 缺失值填充策略

    # ============ DataLoader ============
    batch_size: int = 1024
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    # ============ 模型 ============
    unet_dim: int = 32              # U-Net 基础通道数
    unet_dim_mults: tuple = (1, 2, 4)
    hidden_dim: int = 128           # 混杂/因果模块隐藏维度
    n_heads: int = 4
    dropout: float = 0.1
    mask_ratio: float = 0.3

    # ============ 扩散 ============
    diffusion_steps: int = 100      # 训练时 T
    sampling_steps: int = 20        # DDIM 采样步
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"   # 'linear' | 'cosine'
    sampler: str = "ddim"           # 'ddpm' | 'ddim'
    n_recon_samples: int = 3        # 推理时重建采样次数（取平均）

    # ============ 训练 ============
    epochs: int = 300
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    grad_accum: int = 1
    warmup_epochs: int = 2
    use_amp: bool = True
    use_compile: bool = False       # torch.compile（Windows 上常出问题，默认关）
    use_ema: bool = True
    ema_decay: float = 0.999
    cudnn_benchmark: bool = False
    early_stop_patience: int = 10
    seed: int = 42

    # ============ Loss 权重 ============
    w_diffusion: float = 1.0
    w_causal: float = 0.5
    w_consistency: float = 0.0
    w_sparsity: float = 0.0

    # ============ 异常分数融合 ============
    alpha: float = 0.6              # diffusion score 权重
    beta: float = 0.4               # causal score 权重
    score_norm: str = "robust"      # 'minmax' | 'robust' (median/IQR)

    # ============ Threshold 搜索 ============
    threshold_mode: str = "best_f1" # 'best_f1' | 'quantile' | 'val_quantile'
    threshold_quantile: float = 0.99

    fuse_mode: str = "weighted"          # 暂时回退
    threshold_mode: str = "best_f1"      # 暂时回退
    score_channel_topk: float = 0.3      # 0 = 纯 mean，等同旧版
    score_recon_samples: int = 1         # 暂时关 ensemble
    score_t_start_ratios: tuple = (0.5,)
    diff_target: str = "eps"             # 回到 eps-prediction
    use_min_snr_weight: bool = False
    use_contrastive_aux: bool = False    # 关
    causal_per_time: bool = False        # 关

    # ============ 消融开关 ============
    ablation: str = "full"
    # 可选值:
    #   'full'             : Diffusion + Confounder + Causal + Feedback
    #   'only_diffusion'   : 仅扩散
    #   'only_causal'      : 仅因果（无扩散重建）
    #   'diff_causal'      : 扩散 + 因果（无 backdoor 调整）
    #   'mean_confounder'  : 用全局均值替代 confounder
    #   'remove_backdoor'  : 移除 backdoor adjustment
    #   'remove_feedback'  : 移除因果反馈到扩散

    # ============ 鲁棒性实验 ============
    robustness_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # ============ 快速实验 ============
    fast_mode: bool = False
    fast_sample_ratio: float = 0.1  # 等差采样 n%
    fast_epochs: int = 5

    # ============ Optuna ============
    optuna_trials: int = 30
    optuna_timeout: Optional[int] = None   # 秒；None 表示无限制

    # ============ 设备 ============
    device: str = "cuda"            # 'cuda' | 'cpu'

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cfg = cls()
        for k, v in obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def get_config() -> Config:
    return Config()

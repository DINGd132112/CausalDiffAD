"""
eval_only.py
============
用已训练好的 best.pt 直接跑测试评估，不重训。
用于快速验证推理侧改动（融合策略、归一化、阈值搜索等）。

修改：支持 --dataset all ，遍历 all_datasets 中的所有数据集进行评估。
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model import CausalDiffAD
from data_loader import build_dataloaders
from utils import get_logger, set_seed, load_checkpoint, EMA
from trainer import Trainer

# 所有数据集的列表，当 --dataset all 时依次评估
ALL_DATASETS = ["SWaT", "MSL", "SMAP", "PSM", "SMD"]


def parse_args():
    p = argparse.ArgumentParser("eval_only")
    p.add_argument("--dataset", type=str, required=True,
                   help="数据集名称，或 'all' 表示评估所有数据集")
    p.add_argument("--ckpt", type=str, default=None,
                   help="路径，默认 checkpoints/{dataset}/best.pt")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # 确定要评估的数据集列表
    if args.dataset.lower() == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    # 全局设置随机种子（只需一次）
    cfg0 = get_config()
    set_seed(cfg0.seed)

    # 依次评估每个数据集
    for ds in datasets:
        # 每个数据集使用独立的配置
        cfg = get_config()
        cfg.dataset = ds
        if args.no_amp:
            cfg.use_amp = False

        logger = get_logger(f"eval_{ds}", cfg.log_root)
        logger.info(f"=== eval-only on {ds} ===")

        # 加载数据
        bundle = build_dataloaders(cfg, dataset_name=ds)
        n_features = bundle["n_features"]
        logger.info(f"n_features={n_features}, test_windows={len(bundle['test_loader'].dataset)}")

        # 建模型并加载权重
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model = CausalDiffAD(n_features=n_features, cfg=cfg).to(device)
        ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

        # 确定 checkpoint 路径
        if args.ckpt:
            # 用户指定了全局 ckpt 路径，仅在非 all 模式下合理；all 模式下依然使用该路径（不推荐）
            ckpt_path = args.ckpt
            if len(datasets) > 1:
                logger.warning("--dataset all 时建议不要指定 --ckpt，将使用同一 checkpoint 评估所有数据集")
        else:
            ckpt_path = os.path.join(cfg.ckpt_root, ds, "best.pt")

        if not os.path.exists(ckpt_path):
            logger.error(f"找不到 checkpoint: {ckpt_path}")
            if len(datasets) > 1:
                logger.warning(f"跳过数据集 {ds}")
                continue
            else:
                sys.exit(1)

        logger.info(f"Loading checkpoint from {ckpt_path}")
        load_checkpoint(ckpt_path, model, ema=ema, map_location=device)

        # 跑测试
        tr = Trainer(cfg, logger=logger, tag="eval_only")
        metrics = tr._test(model, bundle, ema)

        # 打印 5 路指标
        logger.info(f"=== TEST DONE for {ds} ===")
        paths = ['diffusion', 'causal', 'final', 'final_max', 'final_diff_only']
        for k in paths:
            if k not in metrics:
                continue
            m = metrics[k]['raw']
            pa = metrics[k]['pa']
            logger.info(f"  [{k:<16}] F1={m['f1']:.4f} P={m['precision']:.4f} "
                        f"R={m['recall']:.4f} ROC-AUC={m['roc_auc']:.4f} "
                        f"PR-AUC={m['pr_auc']:.4f}  PA-F1={pa['f1']:.4f}")

    if len(datasets) > 1:
        print("所有数据集评估完毕。")


if __name__ == "__main__":
    main()
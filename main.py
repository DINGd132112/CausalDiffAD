import os
import argparse
from config import get_config
from utils import get_logger, set_seed
from trainer import run_single, run_all_datasets, run_ablations, run_robustness, run_fast


def parse_args():
    p = argparse.ArgumentParser("Causal-Diff-AD")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "ablation", "robustness", "fast", "optuna"])
    p.add_argument("--dataset", type=str, default=None,
                   help="数据集名；'all' 表示遍历 cfg.all_datasets")
    p.add_argument("--ablation", type=str, default=None,
                   help="单次训练时的消融模式: full | only_diffusion | only_causal | "
                        "diff_causal | mean_confounder | remove_backdoor | remove_feedback")
    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--diffusion_steps", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--unet_dim", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threshold_mode", type=str, default=None,
                   choices=["best_f1", "quantile", "val_quantile"])
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    # Optuna
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--use_pa", action="store_true",
                   help="Optuna 优化 PA-F1（默认优化原始 F1）")
    p.add_argument("--fast_epochs", type=int, default=100,
                   help="Optuna 每个 trial 的 epoch 数")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.dataset: cfg.dataset = args.dataset
    if args.ablation: cfg.ablation = args.ablation
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None: cfg.lr = args.lr
    if args.window_size is not None: cfg.window_size = args.window_size
    if args.diffusion_steps is not None: cfg.diffusion_steps = args.diffusion_steps
    if args.hidden_dim is not None: cfg.hidden_dim = args.hidden_dim
    if args.unet_dim is not None: cfg.unet_dim = args.unet_dim
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.beta is not None: cfg.beta = args.beta
    if args.seed is not None: cfg.seed = args.seed
    if args.threshold_mode is not None: cfg.threshold_mode = args.threshold_mode
    if args.no_amp: cfg.use_amp = False
    if args.compile: cfg.use_compile = True
    return cfg


def main():
    args = parse_args()
    cfg = get_config()
    cfg = apply_overrides(cfg, args)

    os.makedirs(cfg.ckpt_root, exist_ok=True)
    os.makedirs(cfg.log_root, exist_ok=True)
    os.makedirs(cfg.result_root, exist_ok=True)

    set_seed(cfg.seed)
    logger = get_logger(f"causaldiff_{args.mode}", cfg.log_root)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config:\n{cfg.to_json()}")

    if args.mode == "train":
        if cfg.dataset == "all":
            run_all_datasets(cfg, logger)
        else:
            run_single(cfg, logger)

    elif args.mode == "ablation":
        target = None if cfg.dataset == "all" else cfg.dataset
        if cfg.dataset == "all":
            for ds in cfg.all_datasets:
                if not os.path.isdir(os.path.join(cfg.data_root, ds)):
                    logger.warning(f"Skip {ds}: not found")
                    continue
                run_ablations(cfg, logger, dataset_name=ds)
        else:
            run_ablations(cfg, logger, dataset_name=target)

    elif args.mode == "robustness":
        target = None if cfg.dataset == "all" else cfg.dataset
        if cfg.dataset == "all":
            for ds in cfg.all_datasets:
                if not os.path.isdir(os.path.join(cfg.data_root, ds)):
                    continue
                run_robustness(cfg, logger, dataset_name=ds)
        else:
            run_robustness(cfg, logger, dataset_name=target)

    elif args.mode == "fast":
        target = None if cfg.dataset == "all" else cfg.dataset
        if cfg.dataset == "all":
            for ds in cfg.all_datasets:
                if not os.path.isdir(os.path.join(cfg.data_root, ds)):
                    continue
                run_fast(cfg, logger, dataset_name=ds)
        else:
            run_fast(cfg, logger, dataset_name=target)


    elif args.mode == "optuna":
        from hyperopt import run_optuna
        if cfg.dataset == "all":
            for ds in cfg.all_datasets:
                if not os.path.isdir(os.path.join(cfg.data_root, ds)):
                    logger.warning(f"Skip Optuna for {ds}: directory not found")
                    continue
                logger.info(f"\n========== Optuna on dataset: {ds} ==========")
                try:
                    run_optuna(cfg, dataset=ds, n_trials=args.trials,
                               use_pa=args.use_pa, fast_epochs=args.fast_epochs)
                except Exception as e:
                    logger.error(f"Optuna on {ds} failed: {e}")
        else:
            run_optuna(cfg, dataset=cfg.dataset, n_trials=args.trials,
                       use_pa=args.use_pa, fast_epochs=args.fast_epochs)
    else:
        raise ValueError(args.mode)
    logger.info("Done.")


if __name__ == "__main__":
    main()

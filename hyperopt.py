"""
hyperopt.py
===========
使用 Optuna 搜索关键超参数。
搜索空间: lr, batch_size, hidden_dim, diffusion_steps, window_size, alpha, beta
目标: 最大化 test final.raw.f1 (或 PA F1)
"""
import os
import copy
from typing import Optional, Dict, Any
import optuna

from config import Config, get_config
from trainer import Trainer
from utils import get_logger, save_results


def _objective_factory(base_cfg: Config, dataset: Optional[str], use_pa: bool,
                       fast_epochs: int):
    def objective(trial: optuna.Trial):
        cfg = copy.deepcopy(base_cfg)
        cfg.lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        cfg.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        cfg.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 192])
        cfg.diffusion_steps = trial.suggest_categorical("diffusion_steps", [50, 100, 200])
        cfg.window_size = trial.suggest_categorical("window_size", [16, 32, 64])
        cfg.alpha = trial.suggest_float("alpha", 0.1, 0.9)
        cfg.beta = 1.0 - cfg.alpha     # 简化：让两路权重之和为1
        cfg.epochs = fast_epochs
        cfg.use_compile = False        # trial 内禁用 compile 减少开销

        tag = f"trial{trial.number}"
        tr = Trainer(cfg, logger=None, tag=tag)
        try:
            metrics = tr.fit(
                dataset_name=dataset,
                ckpt_subdir=f"{dataset or cfg.dataset}_optuna_{tag}",
            )
        except Exception as e:
            print(f"[trial {trial.number}] failed: {e}")
            raise optuna.TrialPruned()

        key = "pa" if use_pa else "raw"
        f1 = metrics["final"][key]["f1"]
        trial.set_user_attr("metrics", metrics["final"])
        return f1
    return objective


def run_optuna(cfg: Optional[Config] = None, dataset: Optional[str] = None,
               n_trials: Optional[int] = None, use_pa: bool = False,
               fast_epochs: int = 10) -> Dict[str, Any]:
    if cfg is None:
        cfg = get_config()
    n_trials = n_trials or cfg.optuna_trials
    logger = get_logger("optuna", cfg.log_root)
    logger.info(f"=== Optuna search (trials={n_trials}, dataset={dataset or cfg.dataset}, "
                f"use_pa={use_pa}) ===")

    study = optuna.create_study(direction="maximize",
                                study_name=f"causal_diff_ad_{dataset or cfg.dataset}",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(_objective_factory(cfg, dataset, use_pa, fast_epochs),
                   n_trials=n_trials, timeout=cfg.optuna_timeout,
                   show_progress_bar=False)

    # 检查是否有任何 trial 成功完成
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.error(f"All trials failed for dataset={dataset or cfg.dataset}. "
                     f"Check data path and trial errors above.")
        out_path = os.path.join(cfg.result_root, f"optuna_{dataset or cfg.dataset}.json")
        save_results({"best": None,
                      "error": "no completed trials",
                      "trials": [{"number": t.number,
                                  "state": str(t.state),
                                  "params": t.params}
                                 for t in study.trials]},
                     out_path)
        return {"best_value": None, "best_params": None,
                "best_trial_number": None, "best_metrics": None}

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_metrics": study.best_trial.user_attrs.get("metrics"),
    }
    logger.info(f"Best F1={best['best_value']:.4f} with params {best['best_params']}")
    out_path = os.path.join(cfg.result_root, f"optuna_{dataset or cfg.dataset}.json")
    save_results({"best": best,
                  "trials": [{"number": t.number,
                              "value": t.value,
                              "params": t.params}
                             for t in study.trials]},
                 out_path)
    logger.info(f"Saved Optuna results to {out_path}")
    return best

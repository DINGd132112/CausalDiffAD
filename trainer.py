"""
trainer.py
==========
训练/验证/测试 + 消融 + 鲁棒性 + 快速实验。
"""
import os
import time
import math
from typing import Dict, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
# torch>=2.4 推荐 torch.amp，更早版本回退 torch.cuda.amp
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled: bool, dtype=None):
        return _autocast(device_type="cuda", enabled=enabled, dtype=dtype)

    def GradScaler(enabled: bool):
        return _GradScaler(device="cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_legacy, GradScaler as _GradScaler_legacy

    def autocast(enabled: bool, dtype=None):
        return _autocast_legacy(enabled=enabled, dtype=dtype)

    def GradScaler(enabled: bool):
        return _GradScaler_legacy(enabled=enabled)

from model import CausalDiffAD
from data_loader import build_dataloaders, windows_to_pointwise
from evaluator import evaluate
from utils import (
    set_seed, get_logger, save_checkpoint, load_checkpoint, EMA,
    fuse_scores, save_results, save_results_markdown, plot_scores
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """warmup + cosine"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move(batch, device, non_blocking=True):
    if isinstance(batch, (list, tuple)):
        return [b.to(device, non_blocking=non_blocking) for b in batch]
    return batch.to(device, non_blocking=non_blocking)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg, logger=None, tag: str = ""):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.tag = tag
        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # ---------------- train one dataset ---------------- #
    def fit(self, dataset_name: Optional[str] = None,
            train_ratio: float = 1.0,
            fast_sample_ratio: Optional[float] = None,
            epochs_override: Optional[int] = None,
            ckpt_subdir: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        set_seed(cfg.seed)
        device = self.device

        dname = dataset_name or cfg.dataset
        self._log(f"=== [{self.tag}] Fit on dataset: {dname} (train_ratio={train_ratio}, "
                  f"fast={fast_sample_ratio}, ablation={cfg.ablation}) ===")
        bundle = build_dataloaders(cfg, dataset_name=dname,
                                   train_ratio=train_ratio,
                                   fast_sample_ratio=fast_sample_ratio)
        train_loader = bundle["train_loader"]
        val_loader = bundle["val_loader"]
        test_loader = bundle["test_loader"]
        n_features = bundle["n_features"]
        self._log(f"n_features={n_features}, "
                  f"train_windows={len(train_loader.dataset)}, "
                  f"val_windows={len(val_loader.dataset)}, "
                  f"test_windows={len(test_loader.dataset)}")

        # 模型
        model = CausalDiffAD(n_features=n_features, cfg=cfg).to(device)
        if cfg.use_compile:
            try:
                model = torch.compile(model)
                self._log("torch.compile enabled")
            except Exception as e:
                self._log(f"torch.compile failed, fallback: {e}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
        epochs = epochs_override or cfg.epochs
        total_steps = max(epochs * max(len(train_loader), 1) // max(cfg.grad_accum, 1), 1)
        warmup_steps = max(cfg.warmup_epochs * max(len(train_loader), 1) // max(cfg.grad_accum, 1), 1)
        scheduler = _build_scheduler(optimizer, total_steps, warmup_steps)
        use_scaler = (cfg.use_amp and device.type == "cuda"
                      and not torch.cuda.is_bf16_supported())
        scaler = GradScaler(use_scaler)
        ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

        ckpt_dir = os.path.join(cfg.ckpt_root, ckpt_subdir or dname)
        os.makedirs(ckpt_dir, exist_ok=True)
        best_path = os.path.join(ckpt_dir, "best.pt")
        last_path = os.path.join(ckpt_dir, "last.pt")
        cfg_path = os.path.join(ckpt_dir, "config.json")
        cfg.save(cfg_path)

        best_val = float("inf")
        patience = 0

        # ---- training loop ----
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            agg = {"loss": 0., "L_diff": 0., "L_causal": 0., "L_cons": 0., "L_sparse": 0., "n": 0}
            for step, batch in enumerate(train_loader):
                x = _move(batch, device)
                with autocast(cfg.use_amp and device.type == "cuda",
                              dtype=torch.bfloat16):
                    out = model.compute_loss(x)
                    loss = out["loss"] / cfg.grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % cfg.grad_accum == 0:
                    # 1) unscale 梯度以便做 clip
                    scaler.unscale_(optimizer)
                    # 2) 检查梯度有效性
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    skip_step = not torch.isfinite(grad_norm)
                    # 3) 记录当前 scale, 让 scaler.step 自己决定是否跳过
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)  # 内部检测 inf 会自动 skip 真正的 optimizer.step
                    scaler.update()  # 必要：根据 inf 历史自动调整 scale
                    optimizer.zero_grad(set_to_none=True)
                    # 4) 仅在 scale 未下降且梯度有效时才推进 scheduler/ema
                    new_scale = scaler.get_scale()
                    did_step = (new_scale >= prev_scale) and (not skip_step)
                    if did_step:
                        scheduler.step()
                        if ema is not None:
                            ema.update(model)
                    else:
                        # 监控：连续跳步说明 AMP 不稳定
                        if hasattr(self, '_skip_cnt'):
                            self._skip_cnt += 1
                        else:
                            self._skip_cnt = 1
                        if self._skip_cnt % 20 == 0:
                            self._log(f"[WARN] AMP skipped {self._skip_cnt} steps, "
                                      f"current scale={new_scale}")

                bs = x.size(0)
                for k in ("loss", "L_diff", "L_causal", "L_cons", "L_sparse"):
                    v = out[k] if k != "loss" else out["loss"]
                    agg[k] += float(v.detach()) * bs
                agg["n"] += bs

            for k in ("loss", "L_diff", "L_causal", "L_cons", "L_sparse"):
                agg[k] /= max(agg["n"], 1)

            # ---- validate ----
            val_loss = self._validate(model, val_loader, ema)
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            self._log(f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.2e} | "
                      f"train_loss={agg['loss']:.4f} (diff={agg['L_diff']:.4f}, "
                      f"causal={agg['L_causal']:.4f}, cons={agg['L_cons']:.4f}, "
                      f"sparse={agg['L_sparse']:.4f}) | val_loss={val_loss:.4f} | "
                      f"time={dt:.1f}s")

            # save last（每 5 epoch 才保存以减少写盘）
            if epoch % 5 == 0 or epoch == epochs:
                save_checkpoint(last_path, model, optimizer=None, scheduler=None,
                                ema=ema, cfg=cfg,
                                extra={"epoch": epoch, "val_loss": val_loss})

            # early stop
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience = 0
                # 轻量保存：只保存模型权重和EMA，跳过 optimizer/scheduler 减少写盘
                save_checkpoint(best_path, model, optimizer=None, scheduler=None,
                                ema=ema, cfg=cfg,
                                extra={"epoch": epoch, "val_loss": val_loss})
                #self._log(f"  ✓ best val_loss={best_val:.4f}, saved to {best_path}")
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    self._log(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        # ---- load best & test ----
        self._log("Loading best checkpoint and running test...")
        try:
            load_checkpoint(best_path, model, ema=ema, map_location=device)
        except Exception as e:
            self._log(f"[ERROR] load_checkpoint failed: {e}")
            raise
        try:
            metrics = self._test(model, bundle, ema)
        except Exception as e:
            import traceback
            self._log(f"[ERROR] _test failed: {e}\n{traceback.format_exc()}")
            raise
        # 关键指标打印到 log
        try:
            paths_to_log = ['diffusion', 'causal', 'final', 'final_max', 'final_diff_only']
            self._log("=== TEST DONE ===")
            for k in paths_to_log:
                if k not in metrics: continue
                m = metrics[k]['raw']
                pa = metrics[k]['pa']
                self._log(f"  [{k:<16}] F1={m['f1']:.4f} P={m['precision']:.4f} "
                          f"R={m['recall']:.4f} ROC-AUC={m['roc_auc']:.4f} "
                          f"PR-AUC={m['pr_auc']:.4f}  PA-F1={pa['f1']:.4f}")
        except Exception as e:
            self._log(f"[WARN] cannot print summary: {e}")
        # 保存最终结果
        result_dir = os.path.join(cfg.result_root, dname)
        os.makedirs(result_dir, exist_ok=True)
        tag = self.tag or "default"
        save_results(metrics, os.path.join(result_dir, f"metrics_{tag}.json"))
        save_results_markdown(metrics, os.path.join(result_dir, f"metrics_{tag}.md"),
                              title=f"{dname} - {tag}")
        return metrics

    # ---------------- val ---------------- #
    @torch.no_grad()
    def _validate(self, model, val_loader, ema) -> float:
        cfg = self.cfg
        device = self.device
        model.eval()
        if ema is not None:
            ema.apply_to(model)
        total, cnt = 0.0, 0
        for batch in val_loader:
            x = _move(batch, device)
            with autocast(cfg.use_amp and device.type == "cuda",
                          dtype=torch.bfloat16):
                out = model.compute_loss(x)
            bs = x.size(0)
            total += float(out["loss"].detach()) * bs
            cnt += bs
        if ema is not None:
            ema.restore(model)
        return total / max(cnt, 1)

    # ---------------- test ---------------- #
    @torch.no_grad()
    def _test(self, model, bundle, ema) -> Dict[str, Any]:
        cfg = self.cfg
        device = self.device
        model.eval()
        if ema is not None:
            ema.apply_to(model)

        # 用 val 计算分数分布作为归一化参考
        val_loader = bundle["val_loader"]
        test_loader = bundle["test_loader"]
        val_diff_seq, val_causal = [], []
        for batch in val_loader:
            x = _move(batch, device)
            s = model.anomaly_score(x)
            val_diff_seq.append(s["diff_score_seq"].float().cpu().numpy())
            val_causal.append(s["causal_score"].float().cpu().numpy())
        val_diff_seq = np.concatenate(val_diff_seq, axis=0) if val_diff_seq else np.zeros((0,))
        val_causal = np.concatenate(val_causal, axis=0) if val_causal else np.zeros((0,))

        test_diff_seq, test_causal = [], []
        for batch in test_loader:
            x = _move(batch, device)
            s = model.anomaly_score(x)
            test_diff_seq.append(s["diff_score_seq"].float().cpu().numpy())
            test_causal.append(s["causal_score"].float().cpu().numpy())
        test_diff_seq = np.concatenate(test_diff_seq, axis=0) if test_diff_seq else np.zeros((0,))
        test_causal = np.concatenate(test_causal, axis=0) if test_causal else np.zeros((0,))

        if ema is not None:
            ema.restore(model)

        # 还原到点级
        T_total = len(bundle["raw_test_label"])
        W = bundle["window_size"]
        stride = bundle["stride"]
        pt_diff = windows_to_pointwise(test_diff_seq, W, stride, T_total)
        # causal 是窗口级 scalar，广播到整个窗口
        causal_seq = np.repeat(test_causal[:, None], W, axis=1)   # [N, W]
        pt_causal = windows_to_pointwise(causal_seq, W, stride, T_total)

        # 同样还原 val (用于阈值参考分布)
        val_T = val_diff_seq.shape[0] * stride + W if val_diff_seq.shape[0] > 0 else 0
        if val_diff_seq.shape[0] > 0:
            val_pt_diff = windows_to_pointwise(val_diff_seq, W, stride,
                                               max(val_T, W))
            val_causal_seq = np.repeat(val_causal[:, None], W, axis=1)
            val_pt_causal = windows_to_pointwise(val_causal_seq, W, stride,
                                                 max(val_T, W))
        else:
            val_pt_diff = pt_diff.copy()
            val_pt_causal = pt_causal.copy()

        labels = bundle["raw_test_label"].astype(np.int32)

        # 融合
        # 融合：先 robust 归一化，再做 max （取两路最强信号）
        from utils import normalize_scores
        d_n = normalize_scores(pt_diff, mode=cfg.score_norm, ref=val_pt_diff)
        c_n = normalize_scores(pt_causal, mode=cfg.score_norm, ref=val_pt_causal)
        v_d = normalize_scores(val_pt_diff, mode=cfg.score_norm, ref=val_pt_diff)
        v_c = normalize_scores(val_pt_causal, mode=cfg.score_norm, ref=val_pt_causal)

        # 三种融合策略，取最好的（运行时自动选）
        # 三种融合策略
        final_weighted = cfg.alpha * d_n + cfg.beta * c_n
        final_max = np.maximum(d_n, c_n)
        final_diff_only = d_n.copy()

        v_weighted = cfg.alpha * v_d + cfg.beta * v_c
        v_max = np.maximum(v_d, v_c)
        v_diff_only = v_d.copy()

        # 用 val 上的"高分集中度"做无监督融合策略选择（不能用 test label）
        # 启发式：好的异常分数应该是右偏长尾分布（少数高分对应异常）
        # 用 val 上 99% 分位数 / 50% 分位数 的比值作为信号强度，越大越好
        # 方案 A.1: 用 Spearman 相关 + 重尾度共同决策
        from scipy.stats import spearmanr, skew
        try:
            rho, _ = spearmanr(v_d, v_c)
            rho = float(rho) if not np.isnan(rho) else 0.0
        except Exception:
            rho = 0.0
        # 重尾度（右偏 skewness 越大 = 越多极端高分 = 异常分数越好）
        sk_d = float(skew(v_d)) if len(v_d) > 10 else 0.0
        sk_c = float(skew(v_c)) if len(v_c) > 10 else 0.0
        # 标准化到非负
        sk_d = max(sk_d, 0.0);
        sk_c = max(sk_c, 0.0)

        # 决策树:
        #   1) 任一路 skew < 0.5 → 该路退化（无右尾），不可用
        #   2) 高度同向(rho>0.7)：选 skew 更大那路（单路）
        #   3) 高度反向或无关(rho<0.2)：选 skew 更大那路
        #   4) 中等正相关：weighted 融合
        if sk_d < 0.5 and sk_c < 0.5:
            best_strategy = "diff_only";
            reason = "both degenerate"
        elif rho > 0.7:
            if sk_c > sk_d * 1.2:
                best_strategy = "causal_only";
                reason = f"rho={rho:.2f}, causal stronger sk={sk_c:.2f}>{sk_d:.2f}"
            else:
                best_strategy = "diff_only";
                reason = f"rho={rho:.2f}, diff stronger sk={sk_d:.2f}>={sk_c:.2f}"
        elif rho < 0.2:
            if sk_c > sk_d * 1.2:
                best_strategy = "causal_only";
                reason = f"rho={rho:.2f}, causal stronger sk={sk_c:.2f}>{sk_d:.2f}"
            elif sk_d > sk_c * 1.2:
                best_strategy = "diff_only";
                reason = f"rho={rho:.2f}, diff stronger sk={sk_d:.2f}>{sk_c:.2f}"
            else:
                best_strategy = "max";
                reason = f"rho={rho:.2f}, tie sk, use max"
        else:
            best_strategy = "weighted";
            reason = f"rho={rho:.2f} complementary"

        self._log(f"  [auto fusion] rho={rho:.3f} sk_d={sk_d:.2f} sk_c={sk_c:.2f} "
                  f"→ {best_strategy} ({reason})")

        if best_strategy == "weighted":
            final_score, val_final = final_weighted, v_weighted
        elif best_strategy == "max":
            final_score, val_final = final_max, v_max
        elif best_strategy == "causal_only":
            final_score, val_final = c_n.copy(), v_c.copy()
        else:
            final_score, val_final = final_diff_only, v_diff_only

        if best_strategy == "weighted":
            final_score, val_final = final_weighted, v_weighted
        elif best_strategy == "max":
            final_score, val_final = final_max, v_max
        else:
            final_score, val_final = final_diff_only, v_diff_only

        # 同时计算另外两种，作为额外输出
        extra_metrics = {
            "final_max": evaluate(final_max, labels,
                                  threshold_mode=cfg.threshold_mode,
                                  threshold_quantile=cfg.threshold_quantile,
                                  val_score=np.maximum(v_d, v_c)),
            "final_diff_only": evaluate(final_diff_only, labels,
                                        threshold_mode=cfg.threshold_mode,
                                        threshold_quantile=cfg.threshold_quantile,
                                        val_score=v_d),
        }

        metrics = {
            "diffusion": evaluate(pt_diff, labels,
                                  threshold_mode=cfg.threshold_mode,
                                  threshold_quantile=cfg.threshold_quantile,
                                  val_score=val_pt_diff),
            "causal": evaluate(pt_causal, labels,
                               threshold_mode=cfg.threshold_mode,
                               threshold_quantile=cfg.threshold_quantile,
                               val_score=val_pt_causal),
            "final": evaluate(final_score, labels,
                              threshold_mode=cfg.threshold_mode,
                              threshold_quantile=cfg.threshold_quantile,
                              val_score=val_final),
            **extra_metrics,
        }

        # 可视化最终分数
        try:
            dname = bundle.get("name", "dataset")
            plot_path = os.path.join(cfg.result_root, dname,
                                     f"scores_{self.tag or 'default'}.png")
            thr = metrics["final"]["raw"]["threshold"]
            plot_scores(final_score, labels, thr, plot_path,
                        title=f"final score - {self.tag or 'default'}")
        except Exception as e:
            self._log(f"plot failed: {e}")

        return metrics


# --------------------------------------------------------------------------- #
# 高级 wrappers: 多数据集 / 消融 / 鲁棒性 / 快速实验
# --------------------------------------------------------------------------- #
def run_single(cfg, logger):
    tr = Trainer(cfg, logger=logger, tag=cfg.ablation)
    return tr.fit(dataset_name=cfg.dataset)


def run_all_datasets(cfg, logger) -> Dict[str, Any]:
    results = {}
    for ds in cfg.all_datasets:
        # 自动跳过不存在的数据集
        if not os.path.isdir(os.path.join(cfg.data_root, ds)):
            logger.warning(f"Skip {ds}: directory not found.")
            continue
        tr = Trainer(cfg, logger=logger, tag=f"{cfg.ablation}_{ds}")
        results[ds] = tr.fit(dataset_name=ds)
    save_results(results, os.path.join(cfg.result_root, "all_datasets_summary.json"))
    save_results_markdown(results, os.path.join(cfg.result_root, "all_datasets_summary.md"),
                          title="All Datasets Summary")
    return results


def run_ablations(cfg, logger, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    abls = ["full", "only_diffusion", "only_causal", "diff_causal",
            "mean_confounder", "remove_backdoor", "remove_feedback"]
    results = {}
    base = vars(cfg).copy() if hasattr(cfg, "__dict__") else {}
    for a in abls:
        cfg.ablation = a
        tr = Trainer(cfg, logger=logger, tag=f"abl_{a}")
        try:
            results[a] = tr.fit(dataset_name=dataset_name,
                                ckpt_subdir=f"{dataset_name or cfg.dataset}_abl_{a}")
        except Exception as e:
            logger.error(f"Ablation {a} failed: {e}")
            results[a] = {"error": str(e)}
    save_results(results, os.path.join(cfg.result_root,
                 f"ablation_{dataset_name or cfg.dataset}.json"))
    save_results_markdown(results,
                          os.path.join(cfg.result_root,
                                       f"ablation_{dataset_name or cfg.dataset}.md"),
                          title=f"Ablation - {dataset_name or cfg.dataset}")
    return results


def run_robustness(cfg, logger, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    results = {}
    for r in cfg.robustness_ratios:
        tr = Trainer(cfg, logger=logger, tag=f"rob_{int(r*100)}")
        try:
            results[f"{int(r*100)}%"] = tr.fit(
                dataset_name=dataset_name, train_ratio=r,
                ckpt_subdir=f"{dataset_name or cfg.dataset}_rob_{int(r*100)}"
            )
        except Exception as e:
            logger.error(f"Robustness {r} failed: {e}")
            results[f"{int(r*100)}%"] = {"error": str(e)}
    save_results(results, os.path.join(cfg.result_root,
                 f"robustness_{dataset_name or cfg.dataset}.json"))
    save_results_markdown(results,
                          os.path.join(cfg.result_root,
                                       f"robustness_{dataset_name or cfg.dataset}.md"),
                          title=f"Robustness - {dataset_name or cfg.dataset}")
    return results


def run_fast(cfg, logger, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    tr = Trainer(cfg, logger=logger, tag="fast")
    return tr.fit(dataset_name=dataset_name,
                  fast_sample_ratio=cfg.fast_sample_ratio,
                  epochs_override=cfg.fast_epochs,
                  ckpt_subdir=f"{dataset_name or cfg.dataset}_fast")

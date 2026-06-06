"""
model.py
========
核心模型：
    - TemporalUNet     : 1D 时序 U-Net（条件扩散的去噪网络）
    - GaussianDiffusion: DDPM/DDIM forward & reverse
    - ConfounderExtractor : 从 |x - x_hat| 中提取混杂表示 z_c
    - CausalModule     : backdoor adjustment 计算因果表示 h_causal & causal score
    - CausalDiffAD     : 顶层模型，联合优化 + 闭环反馈
"""
from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Relation Predictor (方案 B 核心: masked-channel prediction)
# =========================================================================== #
class RelationPredictor(nn.Module):
    """
    给定窗口 x: [B, T, C] 和通道掩码 mask: [B, C] (1=可见, 0=被遮蔽)，
    预测被遮蔽通道在每个时刻的值。

    核心思想：正常窗口中通道间存在稳定相关结构，遮蔽掩码下能从可见通道恢复
    被遮蔽通道；异常窗口中关系破坏，恢复误差大。
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1, n_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        # 输入: x with masked channels zeroed-out, concat with mask indicator
        # shape [B, T, 2*C]
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        # 输出: 预测每个通道在每个时刻的值
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels),
        )
        # 窗口级摘要 head (供 backdoor 用)
        self.summary_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x    : [B, T, C]
        mask : [B, C] (1=visible, 0=masked)
        return:
            x_hat   : [B, T, C]  predicted full x
            summary : [B, hidden_dim]  窗口级摘要表示
        """
        B, T, C = x.shape
        # mask broadcast 到 [B, T, C]
        mask_t = mask.unsqueeze(1).expand(-1, T, -1).float()
        x_visible = x * mask_t  # masked channels set to 0
        inp = torch.cat([x_visible, mask_t], dim=-1)  # [B, T, 2C]
        h = self.input_proj(inp)  # [B, T, H]
        h = self.encoder(h)  # [B, T, H]
        summary = self.summary_norm(h.mean(dim=1))  # [B, H]
        x_hat = self.output_proj(h)  # [B, T, C]
        return x_hat, summary

# =========================================================================== #
# 1. Sinusoidal time embedding
# =========================================================================== #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================================================================== #
# 2. Temporal U-Net block
# =========================================================================== #
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 cond_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.cond_mlp = nn.Linear(cond_dim, out_ch) if cond_dim > 0 else None
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, T]
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None]
        if self.cond_mlp is not None and cond is not None:
            h = h + self.cond_mlp(F.silu(cond))[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, T)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # [B, h, d, T]
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(C // self.heads)
        # 减去 max 提升数值稳定性
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(B, C, T)
        return x + self.proj(out)


class TemporalUNet(nn.Module):
    """
    1D U-Net. 输入 x: [B, C, T] (噪声序列), 条件 cond: [B, cond_dim].
    输出预测噪声: [B, C, T].
    """
    def __init__(self, in_channels: int, base_dim: int = 32,
                 dim_mults=(1, 2, 4), time_dim: int = 128,
                 cond_dim: int = 0, dropout: float = 0.0,
                 n_heads: int = 4):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        dims = [base_dim * m for m in dim_mults]   # e.g. [32, 64, 128]
        self.init_conv = nn.Conv1d(in_channels, dims[0], 3, padding=1)

        # Down
        self.downs = nn.ModuleList()
        in_d = dims[0]
        for i, d in enumerate(dims):
            self.downs.append(nn.ModuleList([
                ResBlock1D(in_d, d, time_dim, cond_dim, dropout),
                ResBlock1D(d, d, time_dim, cond_dim, dropout),
                SelfAttention1D(d, n_heads) if i == len(dims) - 1 else nn.Identity(),
                nn.Conv1d(d, d, 3, stride=2, padding=1) if i < len(dims) - 1 else nn.Identity(),
            ]))
            in_d = d

        # Mid
        mid = dims[-1]
        self.mid1 = ResBlock1D(mid, mid, time_dim, cond_dim, dropout)
        self.mid_attn = SelfAttention1D(mid, n_heads)
        self.mid2 = ResBlock1D(mid, mid, time_dim, cond_dim, dropout)

        # Up
        self.ups = nn.ModuleList()
        rev = list(reversed(dims))
        for i, d in enumerate(rev):
            next_d = rev[i + 1] if i + 1 < len(rev) else d
            self.ups.append(nn.ModuleList([
                ResBlock1D(d * 2, d, time_dim, cond_dim, dropout),     # skip cat
                ResBlock1D(d, d, time_dim, cond_dim, dropout),
                SelfAttention1D(d, n_heads) if i == 0 else nn.Identity(),
                nn.ConvTranspose1d(d, next_d, 4, stride=2, padding=1)
                    if i < len(rev) - 1 else nn.Identity(),
            ]))

        self.final_norm = nn.GroupNorm(min(8, dims[0]), dims[0])
        self.final_conv = nn.Conv1d(dims[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x   : [B, T, C]
        t   : [B]
        cond: [B, cond_dim] or None
        return: [B, T, C]  predicted noise
        """
        # 转 [B, C, T]
        h = x.transpose(1, 2).contiguous()
        T_in = h.shape[-1]

        t_emb = self.time_emb(t)
        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(h.size(0), self.cond_dim, device=h.device, dtype=h.dtype)

        h = self.init_conv(h)
        skips = []
        for res1, res2, attn, down in self.downs:
            h = res1(h, t_emb, cond)
            h = res2(h, t_emb, cond)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            skips.append(h)
            h = down(h) if not isinstance(down, nn.Identity) else h

        h = self.mid1(h, t_emb, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, cond)

        for res1, res2, attn, up in self.ups:
            s = skips.pop()
            # 形状对齐（步长池化可能产生奇偶长度差）
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, s], dim=1)
            h = res1(h, t_emb, cond)
            h = res2(h, t_emb, cond)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            h = up(h) if not isinstance(up, nn.Identity) else h

        if h.shape[-1] != T_in:
            h = F.interpolate(h, size=T_in, mode="linear", align_corners=False)

        h = F.silu(self.final_norm(h))
        h = self.final_conv(h)
        return h.transpose(1, 2).contiguous()   # [B, T, C]


# =========================================================================== #
# 3. Diffusion (DDPM / DDIM)
# =========================================================================== #
def make_beta_schedule(steps: int, schedule: str = "cosine",
                       beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
    elif schedule == "cosine":
        s = 0.008
        steps_full = steps + 1
        t = torch.linspace(0, steps, steps_full, dtype=torch.float64)
        alpha_bar = torch.cos(((t / steps + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-6, 0.999).float()
    else:
        raise ValueError(schedule)


class GaussianDiffusion(nn.Module):
    def __init__(self, n_steps: int = 100, beta_schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.n_steps = n_steps
        betas = make_beta_schedule(n_steps, beta_schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        somab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sab * x0 + somab * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, x_cond_init: torch.Tensor,
                    n_steps: int, cond: Optional[torch.Tensor] = None,
                    eta: float = 0.0, t_start_ratio: float = 0.5) -> torch.Tensor:
        """
        以输入 x 作为先验：在 t = t_start_ratio * T 处加噪后再 DDIM 反向去噪。
        相比从 T-1 完全噪化，这样能保留原始信号 → 异常处重建误差更可分辨。
        """
        B = x_cond_init.size(0)
        device = x_cond_init.device
        T = self.n_steps
        t_start = max(int(T * t_start_ratio) - 1, 0)
        n_steps = min(n_steps, t_start + 1)
        # 反向步骤索引: 从 t_start 等距下降到 0
        step_idx = torch.linspace(t_start, 0, n_steps + 1).long().to(device)
        # 起点：在 x_cond_init 上加 t_start 步噪声
        t0 = step_idx[0].repeat(B)
        x_t, _ = self.q_sample(x_cond_init, t0)

        for i in range(n_steps):
            t = step_idx[i].repeat(B)
            t_prev = step_idx[i + 1].repeat(B)
            eps = model(x_t, t, cond)
            ab = self.alpha_bar[t].view(-1, 1, 1)
            ab_prev = self.alpha_bar[t_prev].view(-1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab).clamp_min(1e-8)
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab).clamp_min(1e-8)) * \
                    torch.sqrt((1 - ab / ab_prev.clamp_min(1e-8)).clamp_min(0.0))
            noise = torch.randn_like(x_t) if eta > 0 else 0.0
            x_t = torch.sqrt(ab_prev) * x0_pred + \
                  torch.sqrt((1 - ab_prev - sigma ** 2).clamp_min(0.0)) * eps + \
                  sigma * noise
        return x_t   # 近似 x0

    @torch.no_grad()
    def ddpm_sample(self, model: nn.Module, x_cond_init: torch.Tensor,
                    cond: Optional[torch.Tensor] = None,
                    t_start_ratio: float = 0.5) -> torch.Tensor:
        B = x_cond_init.size(0)
        device = x_cond_init.device
        T = self.n_steps
        t_start = max(int(T * t_start_ratio) - 1, 0)
        # 起点：在 x_cond_init 上加 t_start 步噪声
        t0 = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t, _ = self.q_sample(x_cond_init, t0)
        for step in reversed(range(t_start + 1)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps = model(x_t, t, cond)
            beta = self.betas[t].view(-1, 1, 1)
            alpha = self.alphas[t].view(-1, 1, 1)
            ab = self.alpha_bar[t].view(-1, 1, 1)
            mean = (x_t - beta / torch.sqrt(1 - ab).clamp_min(1e-8) * eps) / torch.sqrt(alpha)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta) * noise
            else:
                x_t = mean
        return x_t


# =========================================================================== #
# 4. Confounder Extractor
# =========================================================================== #
class ConfounderExtractor(nn.Module):
    """
    输入: diff = |x - x_hat|, shape [B, T, C]
    输出: z_c [B, hidden_dim]  (序列级混杂表示)
    结构: MLP per-time -> Temporal Self-Attn -> 时间维度 mean pool
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        diff: [B, T, C]
        return:
            z_seq : [B, T, hidden_dim]
            z_c   : [B, hidden_dim]  (pool 后的全局混杂表示)
        """
        h = self.proj(diff)
        h = self.encoder(h)
        h = self.norm(h)
        z_c = h.mean(dim=1)
        return h, z_c


# =========================================================================== #
# 5. Causal Module（backdoor adjustment via prototype-conditioned prediction）
# =========================================================================== #
class CausalModule(nn.Module):
    """
    方案 B：基于 prototype 的 backdoor adjustment。

    输入:
        summary : [B, hidden_dim] — RelationPredictor 输出的窗口摘要
        z_c     : [B, hidden_dim] — 来自 ConfounderExtractor 的混杂表示
    输出:
        h_causal : [B, hidden_dim]  混杂调整后的因果表示
        adjusted_summary : [B, hidden_dim]  用于二次预测
        info dict
    """

    def __init__(self, hidden_dim: int = 128, n_prototypes: int = 16,
                 dropout: float = 0.1, use_backdoor: bool = True):
        super().__init__()
        self.use_backdoor = use_backdoor
        self.hidden_dim = hidden_dim
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, hidden_dim) * 0.02)
        self.log_prior = nn.Parameter(torch.zeros(n_prototypes))
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_prototypes),
        )
        # 结构方程: f(summary, c) -> adjusted_summary
        self.f_xc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, summary: torch.Tensor, z_c: torch.Tensor,
                mean_confounder: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B = summary.size(0)
        protos = self.prototypes
        prior = F.softmax(self.log_prior, dim=0)

        if mean_confounder:
            mean_c = (prior[:, None] * protos).sum(0, keepdim=True).expand(B, -1)
            inp = torch.cat([summary, mean_c], dim=-1)
            h_causal = self.f_xc(inp)
            return h_causal, h_causal, {"q": None}

        if self.use_backdoor:
            # backdoor: 用 prior P(c) 加权 f(summary, c)
            q = F.softmax(self.q_net(z_c), dim=-1)
            s_exp = summary.unsqueeze(1).expand(-1, self.n_prototypes, -1)
            p_exp = protos.unsqueeze(0).expand(B, -1, -1)
            inp = torch.cat([s_exp, p_exp], dim=-1)
            y_per_c = self.f_xc(inp)
            h_causal = (prior[None, :, None] * y_per_c).sum(dim=1)
            return h_causal, h_causal, {"q": q, "prior": prior}
        else:
            q = F.softmax(self.q_net(z_c), dim=-1)
            s_exp = summary.unsqueeze(1).expand(-1, self.n_prototypes, -1)
            p_exp = protos.unsqueeze(0).expand(B, -1, -1)
            inp = torch.cat([s_exp, p_exp], dim=-1)
            y_per_c = self.f_xc(inp)
            h_causal = (q.unsqueeze(-1) * y_per_c).sum(dim=1)
            return h_causal, h_causal, {"q": q, "prior": prior}


# =========================================================================== #
# 6. Top-level model
# =========================================================================== #
class CausalDiffAD(nn.Module):
    """
    顶层模型：闭环联合 (Diffusion <-> Causal)。

    训练:
        1) 前向扩散，预测噪声 -> L_diff
        2) 一步重建估计 x_hat（由预测噪声反推 x0）
        3) diff = |x - x_hat| -> ConfounderExtractor -> z_c
        4) 编码 x 自身 -> h_x
        5) CausalModule(h_x, z_c) -> h_causal, causal_score
        6) (反馈) 第二次 UNet 前向，以 h_causal 作为 cond，再预测噪声 -> consistency

    推理 (评估时):
        - DDIM/DDPM 反向采样得到 x_hat
        - 计算 diffusion_score = MSE(x, x_hat) per-window
        - 计算 causal_score
        - 融合得 final_score
    """
    def __init__(self, n_features: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features

        cond_dim = cfg.hidden_dim if cfg.ablation not in ("only_diffusion",) else 0

        self.unet = TemporalUNet(
            in_channels=n_features,
            base_dim=cfg.unet_dim,
            dim_mults=cfg.unet_dim_mults,
            time_dim=cfg.hidden_dim,
            cond_dim=cond_dim,
            dropout=cfg.dropout,
            n_heads=cfg.n_heads,
        )
        self.diffusion = GaussianDiffusion(
            n_steps=cfg.diffusion_steps,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
        # x 自身编码器: 用 ConfounderExtractor 一样的结构，但作用在 x 上得到 h_x
        self.x_encoder = ConfounderExtractor(
            in_channels=n_features,
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        # 方案 B: relation predictor (masked-channel prediction)
        self.relation = RelationPredictor(
            in_channels=n_features,
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            n_layers=2,
        )
        # decoder: 把 causal 调整后的 summary 解码回 [B, T, C] 用于 refined prediction
        self.refined_decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, n_features),
        )
        # mask ratio
        self.mask_ratio = getattr(cfg, 'mask_ratio', 0.3)
        self.confounder = ConfounderExtractor(
            in_channels=n_features,
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.causal = CausalModule(
            hidden_dim=cfg.hidden_dim,
            n_prototypes=16,
            dropout=cfg.dropout,
            use_backdoor=(cfg.ablation != "remove_backdoor"),
        )
        self.cond_dim = cond_dim

    # ---------------- helpers ---------------- #
    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                             eps: torch.Tensor) -> torch.Tensor:
        sab = self.diffusion.sqrt_alpha_bar[t].view(-1, 1, 1)
        somab = self.diffusion.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return (x_t - somab * eps) / sab.clamp_min(1e-8)

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        _, h_x = self.x_encoder(x)
        return h_x

    # ---------------- training step ---------------- #
    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 临时关闭 consistency loss 调试用 -- 它是 train loss 不下降的另一个嫌疑
        # 因为 cons 用的 eps_pred2 在没有 causal feedback 时是退化的
        """
        x: [B, T, C]
        返回各损失项与总 loss
        """
        cfg = self.cfg
        B = x.size(0)
        device = x.device
        abl = cfg.ablation

        # ---- Diffusion 前向 ----
        t = torch.randint(0, self.diffusion.n_steps, (B,), device=device, dtype=torch.long)
        x_t, noise = self.diffusion.q_sample(x, t)

        # 第一次 UNet：可不带条件（学一个无条件先验）
        cond_first = None
        if self.cond_dim > 0 and abl not in ("only_diffusion",):
            cond_first = torch.zeros(B, self.cond_dim, device=device, dtype=x.dtype)
        eps_pred1 = self.unet(x_t, t, cond_first)
        #L_diff = F.mse_loss(eps_pred1, noise)
        # F.mse_loss 默认 reduction='mean'；我们 explicit 一些，并加上调试统计
        L_diff = F.mse_loss(eps_pred1, noise, reduction='mean')

        # === 诊断：如果你怀疑训练没在学，取消下面注释 ===
        # if torch.rand(1).item() < 0.01:  # 1% 概率打印
        #     with torch.no_grad():
        #         x_var = x.var().item()
        #         xt_var = x_t.var().item()
        #         eps_pred_var = eps_pred1.var().item()
        #         eps_var = noise.var().item()
        #         print(f"[DIAG] x_var={x_var:.3f} x_t_var={xt_var:.3f} "
        #               f"eps_pred_var={eps_pred_var:.3f} eps_var={eps_var:.3f}")

        # 仅扩散：直接返回
        if abl == "only_diffusion":
            return {
                "loss": cfg.w_diffusion * L_diff,
                "L_diff": L_diff.detach(),
                "L_causal": torch.tensor(0., device=device),
                "L_cons": torch.tensor(0., device=device),
                "L_sparse": torch.tensor(0., device=device),
            }

        # ---- 一步重建 x_hat（detach 以隔离 causal 路径对 UNet 的影响）----
        x0_pred = self._predict_x0_from_eps(x_t, t, eps_pred1.detach())
        diff = (x - x0_pred).abs().detach()

        # ---- 混杂提取（从 diffusion 残差）----
        _, z_c = self.confounder(diff)

        # ===== 方案 B 核心: masked-channel prediction =====
        # 1) 随机生成通道掩码 [B, C]
        x_in = x.detach()  # 不让 causal 训练影响 UNet/扩散
        B_, T_, C_ = x_in.shape
        # 每个样本独立生成 mask: 至少留 1 个通道可见，至少遮蔽 1 个
        mask = torch.ones(B_, C_, device=device)
        n_mask = max(1, min(C_ - 1, int(C_ * self.mask_ratio)))
        for i in range(B_):
            idx = torch.randperm(C_, device=device)[:n_mask]
            mask[i, idx] = 0.0

        # 2) RelationPredictor 第一次预测
        x_hat_rel, summary = self.relation(x_in, mask)  # [B,T,C], [B,H]

        # 3) Causal 模块对 summary 做 backdoor 调整
        mean_conf = (abl == "mean_confounder")
        h_causal, adj_summary, _ = self.causal(summary, z_c, mean_confounder=mean_conf)

        # 4) Refined prediction: 用 adj_summary 解码回每个时刻的通道值
        # adj_summary [B,H] -> [B,T,C]，先扩展到 [B,T,H] 然后通过 decoder
        refined_seq_h = adj_summary.unsqueeze(1).expand(-1, T_, -1)  # [B, T, H]
        x_hat_refined = x_hat_rel + self.refined_decoder(refined_seq_h)  # 残差式

        # 5) L_causal: 只在被遮蔽通道上计算 MSE
        mask_inv = (1.0 - mask).unsqueeze(1).expand(-1, T_, -1)  # [B,T,C], 1=masked
        # 主损失: refined prediction MSE on masked channels
        mse_per_pos = (x_hat_refined - x_in) ** 2 * mask_inv
        denom = mask_inv.sum().clamp_min(1.0)
        L_causal = mse_per_pos.sum() / denom

        # 辅助: relation predictor 本身的 MSE on masked channels（小权重）
        mse_rel = ((x_hat_rel - x_in) ** 2 * mask_inv).sum() / denom
        L_causal = L_causal + 0.5 * mse_rel

        if abl == "only_causal":
            loss = cfg.w_causal * L_causal + cfg.w_sparsity * z_c.abs().mean()
            return {
                "loss": loss,
                "L_diff": torch.tensor(0., device=device),
                "L_causal": L_causal.detach(),
                "L_cons": torch.tensor(0., device=device),
                "L_sparse": z_c.abs().mean().detach(),
            }

        # consistency 保持关闭
        L_cons = torch.tensor(0., device=device)

        # 稀疏正则：放松到 0 或微小权重
        L_sparse = z_c.abs().mean()

        total = (cfg.w_diffusion * L_diff
                 + cfg.w_causal * L_causal
                 + cfg.w_consistency * L_cons
                 + cfg.w_sparsity * L_sparse)
        return {
            "loss": total,
            "L_diff": L_diff.detach(),
            "L_causal": L_causal.detach(),
            "L_cons": L_cons.detach(),
            "L_sparse": L_sparse.detach(),
        }

    # ---------------- inference ---------------- #
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """通过扩散反向采样得到 x_hat；可多次采样平均（n_recon_samples）。"""
        cfg = self.cfg
        cond = None
        if self.cond_dim > 0:
            cond = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)
        n = max(cfg.n_recon_samples, 1)
        accum = torch.zeros_like(x)
        for _ in range(n):
            if cfg.sampler == "ddim":
                x_hat = self.diffusion.ddim_sample(self.unet, x, cfg.sampling_steps, cond=cond)
            else:
                x_hat = self.diffusion.ddpm_sample(self.unet, x, cond=cond)
            accum = accum + x_hat
        return accum / n

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回:
            diff_score_seq : [B, T]   per-point 扩散分数 (mean abs over channels)
            causal_score   : [B]      窗口级因果分数
            x_hat          : [B, T, C]
        """
        cfg = self.cfg
        x_hat = self.reconstruct(x)
        diff = (x - x_hat).abs()
        # 通道聚合：默认 mean；若 cfg.score_channel_topk ∈ (0, 1)，取通道 top-k 比例的均值
        topk_ratio = getattr(cfg, 'score_channel_topk', 0.0)
        if topk_ratio is not None and 0.0 < topk_ratio < 1.0:
            C = diff.size(-1)
            k = max(int(C * topk_ratio), 1)
            topk_vals, _ = diff.topk(k, dim=-1)  # [B, T, k]
            diff_score_seq = topk_vals.mean(dim=-1)  # [B, T]
        else:
            diff_score_seq = diff.mean(dim=-1)

        if cfg.ablation == "only_diffusion":
            B = x.size(0)
            zero = torch.zeros(B, device=x.device)
            return {"diff_score_seq": diff_score_seq, "causal_score": zero, "x_hat": x_hat}

            # ===== 方案 B: causal_score = masked-channel prediction error =====
        _, z_c = self.confounder(diff)
        mean_conf = (cfg.ablation == "mean_confounder")

        B_, T_, C_ = x.shape
        # 推理时也做 masking，但用确定性的 mask 集合做多次平均（更稳定）
        n_mask = max(1, min(C_ - 1, int(C_ * self.mask_ratio)))
        # 推理时采样 K 个不同的 mask 取平均
        K = 3
        all_errors = torch.zeros(B_, device=x.device)
        for _k in range(K):
            mask = torch.ones(B_, C_, device=x.device)
            for i in range(B_):
                idx = torch.randperm(C_, device=x.device)[:n_mask]
                mask[i, idx] = 0.0
            x_hat_rel, summary = self.relation(x, mask)
            h_causal, adj_summary, _ = self.causal(summary, z_c, mean_confounder=mean_conf)
            refined_seq_h = adj_summary.unsqueeze(1).expand(-1, T_, -1)
            x_hat_refined = x_hat_rel + self.refined_decoder(refined_seq_h)
            mask_inv = (1.0 - mask).unsqueeze(1).expand(-1, T_, -1)
            # per-window error: 对每个样本的所有遮蔽位置取均值
            err_per_sample = ((x_hat_refined - x) ** 2 * mask_inv).sum(dim=(1, 2)) \
                             / mask_inv.sum(dim=(1, 2)).clamp_min(1.0)
            all_errors = all_errors + err_per_sample
        causal_score = all_errors / K  # [B]

        return {"diff_score_seq": diff_score_seq,
                "causal_score": causal_score,
                "x_hat": x_hat}

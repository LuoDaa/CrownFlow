"""M1 Geometry Auto Margin —— 数据结构与配置。

**M1 只用几何，不训练任何模型，也绝不读 GT Margin 参与生成**：GT 只在
``evaluate`` 里出现（见 ``tools/run_auto_margin_m1.py``）。

输入严格限制（用户 2026-09-30 规格）：

    代型网格（modeldie）+ 就位轴（Insertion Axis）

输出：预测的 Margin 闭环。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- 状态
# 用户 2026-10-01 规格：三态语义，**生产版本不允许 silent fallback**。
STATUS_AUTO_ACCEPTED = "AUTO_ACCEPTED"      # 有可行闭环 + 闭环/无自交/表面 QC 通过
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"  # 无可行闭环 / 置信度低 / 环有歧义 / QC 异常
STATUS_FAILED = "FAILED"                    # 网格非法 / 缺轴 / 候选覆盖不足 / 数值失败
#: 兼容别名（旧代码/测试里用 STATUS_OK 表示"无异常"）
STATUS_OK = STATUS_AUTO_ACCEPTED
STATUS_REVIEW = STATUS_REVIEW_REQUIRED


@dataclass(frozen=True)
class AutoMarginConfig:
    """M1 的全部旋钮（**只允许在 val 上冻结，不许逐例调参**）。"""

    #: 角向扇区数。每个扇区出 1 个候选点 → 天然保证闭环、不自交。
    n_sectors: int = 120

    # ---------------------------------------------------------- v2（DP）旋钮
    # ---------------------------------------------------------- v3（候选生成）旋钮
    #: 候选生成方式：
    #:   "faces"  = v2（每扇区按分数排 Top-K 的**面**，top_k=24）
    #:   "ridges" = v3（每扇区取 Top ridge **簇**，ridge_top_k 个）
    #: v3 只换这一步，DP/硬门/平滑/v2 配置全部不动。
    #: **默认 = 冻结的 v2（faces/K24）** —— v3 已被数据否证（召回 1.00→0.22），
    #: 必须显式请求；默认值绝不能再漂移到未冻结的配置上（这正是 0.098 vs 0.634
    #: 那次口径混乱的根源之一）。
    candidate_mode: str = "faces"
    #: 每扇区最多取几个 ridge 簇（用户规格：K 可以回到很小）
    ridge_top_k: int = 6
    #: 一个簇至少几个支撑面（滤掉单面噪声）
    ridge_min_support: int = 2
    #: 连通 seed 分量的轴向跨度超过 factor×dh 时，按最大间隙切分（防止两条脊被并成一条）
    ridge_split_span_factor: float = 3.0
    #: **簇的角向跨度上限（度）**。实测必须要有：一条 360° 环绕特征（底盖棱这类）
    #: 会被聚成一个巨簇，在**每个扇区**都是最高分，路径退化成同一簇的 120 个拷贝
    #: （环长 0 → 重采样早退）。真脊是"短弧"，所以簇在角向上也必须局部。
    ridge_max_span_deg: float = 9.0

    #: 路径优化方法（**实测冻结：dp**）：
    #:   "argmax" = v1（每扇区独立取最高分，无整圈约束）
    #:   "dp"     = v2（扇区×Top-K 候选 格点上的**环状**最小代价路径）
    method: str = "dp"

    # ---------------------------------------------------- B/C：fail-loud 语义
    #: DP 无可行闭环时的行为：
    #:   "REVIEW"  = **生产语义**：返回 REVIEW_REQUIRED 且 margin=None，
    #:               绝不输出一条"看起来合法"的曲线（用户 2026-10-01 §B）
    #:   "ARGMAX"  = 仅用于研发诊断/消融，回退 argmax 并在 notes 里标注
    fallback_mode: str = "REVIEW"

    # ---------------------------------------------------- ② transition（v4）
    #: 表面测地搜索半径（mm）。**必须局部**：不许在整张 modeldie 上无限制最短路，
    #: 否则会出现"绕基座半圈也连通"——数学可达、牙科语义完全错误。
    surface_search_radius_mm: float = 3.0
    #: detour_ratio = d_geo / max(d_euclid, eps) 的上限。
    #: 两个点"在网格上能走到"不等于它们属于合理的相邻 Margin 段；绕太远就禁掉。
    max_detour_ratio: float = 6.0
    #: HYBRID 的**宽松** sanity 门（只防"突然跨几毫米/从预备体跳到基座"，
    #: 不承担"Margin 应该多平滑"的职责，所以比 v2 冻结门宽得多）。
    sanity_dh_mm: float = 2.50
    sanity_dr_mm: float = 1.50
    #: 测地代价权重（E_g = ρ(d_geo / d_expected)）
    lambda_geo: float = 1.0

    #: 迁移代价模式（§9）：
    #:   "V2_FROZEN" = 冻结的 h/r 常数门 + 离散度项（当前默认）
    #:   "MESH_AWARE" / "HYBRID" = 实验项（Ablation 用；未在 blind 前不得默认启用）
    transition_mode: str = "V2_FROZEN"

    #: 每扇区保留的候选数（v2）。**实测冻结值 24**（val 上扫过 3/5/8/12/16/24/32）：
    #:   K:   3     5     8     12    16    24    32
    #:   p95: 1.802 0.634 0.220 0.104 0.101 0.098 0.098
    #:   p95≤0.5 成功率: 7/19 → 9 → 13 → 15 → 16 → 19/19 → 19/19
    #: K=24 已饱和（=K=32），且 DP 复杂度 O(N·K²·K)=120·576·24 仍可忽略。
    #: **K 是"候选池大小"这个结构性参数，不是逐例调参** —— 召回诊断显示
    #: GT 最近候选的分数排名 p90=14，K=5 在一半扇区里根本够不到 Margin。
    top_k: int = 24

    #: 平滑项权重（Huber）。三项都给：只约束 Δh 时"两条高度相近的不同脊"
    #: 仍可能被跳过去，半径与 3D 步长能明显减少这类错误跳转。
    lambda_h: float = 1.0
    lambda_r: float = 1.0
    lambda_p: float = 0.6

    #: Huber 的拐点（mm）：越过它变成线性，避免单个异常扇区把整条环拖坏。
    huber_delta_mm: float = 0.35

    #: 跳变硬门（mm）。**实测冻结值**（tools/derive_m1_gates.py 在 train 上跑，
    #: 8039 个相邻扇区样本 / 67 行）：GT Margin 的环向起伏其实很小 ——
    #:   Δh: p50 0.043 / p90 0.112 / p95 0.142 / p99 0.215 / max 0.904
    #:   Δr: p50 0.031 / p90 0.089 / p95 0.110 / p99 0.157 / max 0.278
    #: 门限 = p99 + 0.5·(p99 − p50) → Δh 0.301mm、Δr 0.219mm（见 m1_gates.json）。
    #: 超门直接禁止该转移。**这是 train 上的参数设计，不是 GT leakage。**
    #: 注意：实际门限 = base + gate_disc_factor × 网格面尺度 —— 候选是"面"，
    #: 面质心高度带 ±face_scale/2 量化；只取 GT 的 p99 会让密集网格上
    #: "任何环都不可行"（实测 62% 回退 argmax，DP 完全没机会）。
    max_dh_mm: float = 0.301
    max_dr_mm: float = 0.219
    gate_disc_factor: float = 1.0

    #: 内部：由引擎按网格算出并填（面尺度 mm）。**不是用户旋钮**。
    _face_scale_mm: float = 0.0

    #: 候选打分函数名（见 geometry_score.SCORES）。M1 要在 val 上比较它们，
    #: 选出"几何上真的能指到 Margin"的那一个 —— 这是 M1 的第一项结论。
    score: str = "sharp"      # 实测冻结：sharp（median 与闭环率最好，DP 下 p95 最优）

    #: 搜索带：排除轴向两端（咬合顶盖 / 底座）各此比例的高度范围。
    #: 一个**与 GT 无关**的先验：Margin 不会落在网格的轴向极值处。
    band_trim_frac: float = 0.08

    #: 环向平滑窗口（扇区数，奇数）与重采样点数。
    smooth_window: int = 5
    resample_n: int = 240

    # ---------------------------------------------------- QC 阈值（GT-independent）
    #: 预测点投回网格表面的最大位移（mm）。**不用 GT 调**，取 0.5×面尺度量级。
    max_proj_disp_mm: float = 0.60
    #: 沿环的单个转移 3D 步长上限（mm）。实测冻结配置下 p90 0.67、max 0.82，
    #: 这里放到 1.5 只做"明显异常"的兜底（sanity，不是精度门）。
    max_transition_mm: float = 1.50

    #: QC 门限
    min_filled_sector_frac: float = 0.98   # 有候选的扇区占比
    max_radius_jump_mm: float = 1.20       # 相邻重采样点的半径突变
    min_loop_length_ratio: float = 0.50    # 预测环长 / GT 周长（仅 QC 参照，不用来生成）


def frozen_config_id(cfg) -> str:
    """稳定的**配置指纹**，写进每一行输出。

    起因（踩过）：同一套指标出现过 0.098 与 0.634 两个"v2 baseline"，根因是
    两次跑用的 `top_k` 不同（5 / 24）而输出里没有任何配置记录 —— 只靠文件名
    和记忆核对，必然出错。**以后任何对比都必须带 config_id。**
    """
    # transition_mode 必须进指纹：曾出现两次不同 transition 的运行共用同一
    # config_id（V2_FROZEN vs HYBRID），provenance 会说谎。
    return ("%s_K%d|ridge%d|%s|%s|%s|gate%.3f/%.3fx%.2f|lam%.2f/%.2f/%.2f/"
            "geo%.2f|hub%.2f|sec%d|res%d|band%.2f|sm%d"
            % (cfg.candidate_mode, cfg.top_k, cfg.ridge_top_k, cfg.score,
               cfg.method, cfg.transition_mode, cfg.max_dh_mm, cfg.max_dr_mm,
               cfg.gate_disc_factor, cfg.lambda_h, cfg.lambda_r, cfg.lambda_p,
               cfg.lambda_geo, cfg.huber_delta_mm, cfg.n_sectors,
               cfg.resample_n, cfg.band_trim_frac, cfg.smooth_window))


#: **M1 v2 冻结配置**（2026-09-30 在 val 上选定并冻结）。任何"v2 baseline"
#: 都必须用它，且输出里必须带 config_id。
M1_V2_FROZEN = dict(candidate_mode="faces", top_k=24, score="sharp",
                    method="dp", max_dh_mm=0.301, max_dr_mm=0.219,
                    gate_disc_factor=1.0, lambda_h=1.0, lambda_r=1.0,
                    lambda_p=0.6, huber_delta_mm=0.35, n_sectors=120,
                    resample_n=240)


#: **v4 冻结 transition**（用户 2026-10-02 选定）：MESH_AWARE 与 HYBRID 在
#: Hard Dev 全量上逐位一致（6/6 GT-infeasible 恢复、GOOD 140 行 0 新 catastrophic、
#: p95 中位 0.106→0.076），按用户倾向取 HYBRID —— 多一层宽松 sanity guard，
#: 适合生产。此后所有 confidence/Gate 的数值都在该 transition 下计算。
M1_V4_TRANSITION = "HYBRID"


#: **M1 生产候选配置**（用户 2026-10-01 命名）。与 M1_V2_FROZEN 的差别只有一条：
#: ``fallback_mode="REVIEW"`` —— 无可行闭环时**不输出兜底曲线**，直接 REVIEW。
#: 其余（faces/K24、sharp、冻结硬门、cyclic DP）与已验证的 v2 完全一致。
M1_V2_PRODUCTION_CANDIDATE = dict(M1_V2_FROZEN, fallback_mode="REVIEW")


@dataclass
class PredictedMargin:
    """预测的 Margin 闭环。"""

    points: list = field(default_factory=list)   # [(x,y,z), ...] 已按角度排序
    n_sectors_filled: int = 0
    n_sectors: int = 0
    closed: bool = False
    notes: list = field(default_factory=list)

    def length_mm(self) -> float:
        if len(self.points) < 2:
            return 0.0
        tot = 0.0
        for i in range(len(self.points)):
            a = self.points[i]
            b = self.points[(i + 1) % len(self.points)]
            tot += ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5
        return tot


@dataclass
class AutoMarginResult:
    """一次预测的完整结果（含 QC，不含 GT 比较 —— 那是 evaluate 的事）。"""

    status: str = STATUS_FAILED
    margin: PredictedMargin | None = None
    quality: dict = field(default_factory=dict)
    #: 置信度/歧义（用户 §12/§13/§14）：全部来自**几何量**，不看 GT
    confidence: dict = field(default_factory=dict)
    #: 求解轨迹：最优环能量与次优环能量（次优 = 起点候选不同的另一条环）
    loop_energy: float | None = None
    second_energy: float | None = None
    energy_gap: float | None = None
    #: DP 选中候选在其扇区内的分数排名统计（rank 1 = 该扇区最高分）
    candidate_rank_stats: dict = field(default_factory=dict)
    #: provenance（§D）：配置指纹 / 打分法 / 迁移法 / 求解器
    config_id: str = ""
    score_method: str = ""
    transition_method: str = ""
    solver: str = "CYCLIC_DP"
    #: 仅在 ``return_candidates=True`` 时填充：{"topk":…, "full":…, "chosen":…}
    #: 用于 M1 的候选召回诊断（GT 附近是否已在 Top-K 候选里）
    candidates: dict = field(default_factory=dict)
    reason: str = ""
    elapsed_s: float = 0.0

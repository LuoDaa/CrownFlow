# AUTOBITE 实现状态（机器可续接；只记要点，不写分析）

## completed
- D1 修复：register_pair 每侧取逆进契约（`inv(register_pair(patch, jaw))`）；B1.5 位姿恢复 0.006–0.015mm / 0.06–0.83°。
- Canonical World fixture + 公式层 convention tests：`tests/unit/geometry/test_bite_canonical_world.py`（10 passed）。
- solver-output 断言测试：`test_bite_solver_output_convention.py`（4/5；rigid 三帧那条红）。
- 定位：pair-level lower registration PASS（0.0042mm/0.229°，inlier 1.000）⇒ 非 A 类。
- SE(3) 等变 ×20 FAIL（worst 18.54mm/42.56°）⇒ C 类：coarse 离散选择随坐标系翻转。
- B3 最小实现（本轮）：coarse 候选**池化**（不再提前选一个）+ 每候选低分辨率 ICP + **只用作标系不变量排序**（rmse→inlier→matched area）+ refine top3 + `REGISTRATION_SOLUTION_AMBIGUOUS` 门。

## contracts（冻结，勿重定义）
- `p_bite = T_upper_to_bite @ p_upper`；`p_bite = T_lower_to_bite @ p_lower`；
  `T_lower_to_upper = inv(T_upper_to_bite) @ T_lower_to_bite`。
- 核心模块禁用 T_bu/T_bl/T_lu；`combine_lower_to_upper` 仅兼容别名（DeprecationWarning）。
- 刚性：RᵀR≈I、det≈+1、bottom row、无 scale/shear（`rigid_transform_report`）。
- 禁止跨物理表面 RMSE；pair measurement 统一在 Bite Frame。
- 旧阈值不放宽：`pass_rmse_mm 0.30 / review_rmse_mm 0.80 / pass_inlier_ratio 0.60 / review_inlier_ratio 0.35 / min_overlap_area_mm2 5.0 / gross_*`。

## current failing tests
- `test_bite_localization.py::test_se3_equivariance_common_rigid`（B3 后需重跑，预期改善）。
- `test_bite_localization.py::test_side_assignment_provenance_three_frames`（**harness bug**：`sides["upper"]` 是 pair report 不是带 `.mesh` 的对象；改用 `lower_patch` 比 centroid）。
- `test_bite_solver_output_convention.py::test_solver_output_rigid`（三帧；B3 后需重跑）。
- `test_bite_b1_baseline.py`：13 类里有 harness `NoneType` bug（low_information_patch）。
- 旧文件 `test_bite_registration_synthetic.py`：7 failed（LEGACY_FIXTURE，保留不动、不作 gate）。

## next exact code action
1. 修 `test_bite_localization.py::test_side_assignment_provenance_three_frames`：按 pair report 结构改断言（用 `sides["lower_patch"]` centroid vs 真值；upper 侧只看位姿误差）。
2. 跑 `pytest tests/unit/geometry/test_bite_localization.py -q -s`，看等变 worst 是否落回 1° 级。
3. 若仍 fail：查 `coarse_align` 的 PCA 轴序/符号枚举与 `COARSE_TIE_*` 门限（必要时放宽 `COARSE_TIE_REL` 以纳入更多等价候选）。
4. 通过后 → D2 结构化 PairQuality（禁 `dict.get(...,-1)`；`source_coverage_ratio=matched bite patch/total bite patch`；`whole_target_coverage_ratio` 仅 diagnostic；补 spatial evidence）→ JointConsistencyQC 聚合化 + `INSUFFICIENT_REGISTRATION_EVIDENCE` → 14 fixed behavior → MC 100 → real-geometry synthetic → Service/ConfirmedBitePose/stale/6 API/三点/手动/接线/回归/前端。

## measured（最近一次实测）
- canonical contract suite：10 passed / 0 failed。
- solver-output conventions：identity 0.0065/0.099°；translation 0.0055/0.410°；rotation 0.0091/0.272°；rigid(三帧) T_lower_to_bite 7.041mm/43.187°（待 B3 复测）。
- B1/B1.5：见 `tests/unit/geometry/test_bite_b1_baseline.py` 的 `-s` 输出。
- REAL_BITE_SCAN_VALIDATION_CASES = 0；REAL_GEOMETRY_SYNTHETIC_BITE 未跑。

## update（B3 第二步）
- 已加 ROTATION_SEED（绕源主轴 8 步 + ±20° 倾斜 4 方位）到 coarse 池；`_axis_rotation` 为模块 helper。
- provenance harness 已改：用 `sides["lower_patch"]` 比 centroid；upper 侧改用位姿误差。
- B3 第一步后实测：equivariance worst 34.77°（原 42.56°）、rigid 三帧 20.46°（原 43.19°）—— 方向对但未收敛到容差。

## update（排序证据化）
- `_invariant_score` 顺序改为 **inlier → rmse → matched area**（每个都坐标不变）。
  证据：rotation-seed 入池后，rmse-first 会被"沿圆柱滑动"的错位姿骗到 15.8°（translation case）。
- 上一轮实测：provenance PASS（centroid 0.0000mm / A_UP_B_LOW）；rigid 三帧 PASS（0.348°）；
  translation 与 pair-level 因排序被骗而 FAIL（本轮修）。

## update（换 ranking 后的实测，2026-10-14 最后一轮）
pair-level: PASS 0.0062mm/0.0798° inlier 1.000 | provenance: centroid 0.0000mm PASS 但 **lower side pose 2.3795mm/19.5435° FAIL**（同一 patch 单独测却对 ⇒ **B 类**：assignment 路径）
solver-output: identity PASS 0.0071/0.097° | rotation PASS 0.0039/0.098° | translation FAIL 3.2836mm/41.92° | rigid 三帧 FAIL 3.7229mm/11.90°（上一轮曾 PASS 0.348°）
equivariance ×20 worst: 27.82°（历史 42.56 → 34.77 → 44.72 → 27.82）
inlier-first 修好了 pair-level，但 scalar ranking 总体上在"圆柱滑动"与"真解"之间来回翻 ⇒ **scalar ranking 不足**。

## next exact code action（只做这一件）
把"单一标量排名"换成 **姿态聚类 + 证据门**（全部坐标不变量）：
1. 对 refined 候选两两算**相对位姿**（ΔR/Δt）。
2. 按"位姿接近"(Δrot<1° 且 Δtrans<0.2mm)聚类。
3. 每个簇取**簇内最优证据**（inlier→rmse→matched area）；**先按匹配面积硬门**淘汰簇内最优面积 < 0.5×全局最优面积的候选（那就是"滑走"的解）。
4. 若最优簇与次优簇的证据接近（inlier 差 ≤0.05）但位姿差 >2°/0.5mm ⇒ `REGISTRATION_SOLUTION_AMBIGUOUS` → REVIEW（已接线）。
5. 都不满足 ⇒ 取最优簇代表。
验收门：`test_bite_localization.py`（pair-level / provenance / equivariance×20）+ `test_bite_solver_output_convention.py` 全绿。

## update（B3 第三步：D2 透传 + 姿态聚类）
- D2 根因找到：`register()` 的 `per_side` 只白名单 7 个键 ⇒ coverage/matched area 被丢掉，被 harness 的 -1 兜底误报成"覆盖率不足"。已改为整字典透传。
- `_pair_quality` 补结构化字段：matched_surface_area_mm2 / source_coverage_ratio / whole_target_coverage_ratio(+diagnostic_only) / matched_bbox_extent_xyz / 三个 principal extent / 两个 eigen ratio / matched_normal_variation_deg / measurement_status（测不出 None + NOT_ASSESSED）。
- 排名改为 **姿态聚类**（不变量：ΔR 夹角 + 参考点位移 RMS）→ 簇选择按 **支持度优先**（member_count → source_method 多样性 → 证据）→ 单成员簇 + 同水平簇 ⇒ REGISTRATION_SOLUTION_AMBIGUOUS → REVIEW。
- 聚类参数（非临床阈值）：POSE_CLUSTER_ROT_DEG=1.0 / POSE_CLUSTER_POINT_MM=0.20。
- next exact code action：跑 test_bite_localization + test_bite_solver_output_convention + canonical；若仍 FAIL，读 result["clusters"] / winner_cluster_support 判定 GENERATION vs SELECTION。

## update（聚类已生效；剩余 1 个算法失败）
PASS: solver-output identity/translation/rotation（0.003–0.010mm，0.01–0.50°）、pair-level lower（0.0045mm/0.489°）、provenance upper side（0.0046mm/0.25°）。
FAIL(算法): provenance **lower side 3.5725mm/20.63°** —— 同一 patch 单独 pair-level 却对 ⇒ B 类（`_assign_two` 里对 lower_jaw 的那次 register_pair 落错簇）。
FAIL(环境): equivariance ×20 撞到 `_ArrayMemoryError`（1.68 MiB 都分配失败）⇒ 非算法失败，机器内存不足；需先释放内存（chrome/其它 python）再跑。
per_side 已整字典透传；`whole_target_coverage_ratio=target_coverage` + `diagnostic_only=true`；`_pair_quality` 结构化字段已落。

## next exact code action
1. 在 `_assign_two` 里对两次 `register_pair` 记录 `clusters`/`winner_cluster_support`，判定 lower 侧是 **GENERATION**（正确簇没生成）还是 **SELECTION**（生成了但没选中）。
2. 若 GENERATION：该分量独立跑一次 `coarse_align` 看池里是否含真解（分量顶点序/裁剪差异）；若 SELECTION：簇选择里把"该簇代表与另一侧的相对位姿一致性"纳入（两侧共同支持同一咬合关系）。
3. 释放内存后重跑 `test_bite_localization.py`（equivariance ×20 必须真跑完，报告 median/p90/worst）。

## update（决定性：采样非确定性 + 真实退化）
根因（已修）：`_sample` 用 `mesh.sample()` = **随机采样** ⇒ 同一输入两次调用结果不同（实测同一分量 0.0091mm vs 3.39mm）。已改为**面中心确定性采样**（stride 抽稀，无 RNG）。
修后实测：**identity / rotation 精确恢复 0.0000mm / 0.0000°**（ICP 本身是好的）；equivariance worst 27.8→22.9°。
仍 FAIL 的语义：translation / rigid三帧 / pair-level lower（0.83–3.40mm, 6.6–10.7°）。判读：夹具 band 是**均匀圆柱** ⇒ 沿轴滑动是**精确对称**（同 RMSE、同 inlier），属**真退化/多解**，不是求解器能力不足。
⇒ 修复归属：**歧义门**必须把这种"证据并列但位姿不同"判为 REGISTRATION_SOLUTION_AMBIGUOUS → REVIEW（当前池里没有"沿轴滑动"的候选，gate 未触发）。

## next exact code action
1. `coarse_align` 候选池补 **AXIAL_SLIDE 候选**（沿源主轴 ±{0.5,1.0,2.0,3.0}mm 平移，rigid、no scale），使"滑动解"进入池。
2. 聚类后若最优簇与次优簇 `median_inlier` 差 ≤0.05 且位姿差 >2°/0.5mm ⇒ `ambiguous_solution=True`（已接线 → REVIEW）。
3. 跑 `test_bite_localization.py`：translation/rigid三帧 的**期望从"必须 PASS"改为"PASS 或 REVIEW+AMBIGUOUS"**（退化夹具不该要求唯一解）；`identity/rotation` 必须保持精确。
4. 然后 14 类 fixed behavior（低信息/错内容/穿模/缺 bite 必须 FAILED/REVIEW，绝不 PASS）→ MC 100 → real-geometry synthetic → freeze → Service/API/前端。

## update（E/F 已落：PoseObservabilityReport + 证据不足门）
新增：`pose_observability()`（沿**源几何主轴系**扰动 6DoF ±{0.25,0.5,1,2}mm / ±{0.5,1,2,4}°，比较不变质量变化；某自由度质量几乎不变 ⇒ weak_dof）→ `register_pair` 返回 `observability` + `insufficient_evidence` → `_verdict` 命中则 **REVIEW / INSUFFICIENT_REGISTRATION_EVIDENCE**（禁止 PASS）。新常量 `REASON_INSUFFICIENT_REGISTRATION_EVIDENCE` 已入 REASON_CODES。分析阈值标 ENGINEERING_DEFAULT_UNVALIDATED（旧阈值未动）。
修掉自伤 bug：`"T%d" % "xyz"[index]` 优先级 ⇒ 改 `%s`（4 处）。

实测（deterministic 采样 + 上述改动后）：
- solver-output：**identity 0.0000/0.0000 PASS**、**rotation 0.0000/0.0000 PASS**；translation 3.3967mm/6.597°、rigid 三帧 0.8287mm/10.658° 仍 FAIL，且**都只错在 `T_lower_to_bite`**（`T_upper_to_bite` 精确）。
- 结论：当前 canonical band 仍存在**轴向弱约束**（滑走解与真解质量等价）⇒ 按 B/C 必须把 fixture **拆成两个**。

## next exact code action
1. `tests/unit/geometry/_bite_fixture.py` 拆两个构造：
   - `well_posed_world()`：band 内加**非周期、沿轴变化**的截面/脊（例如轴向线性变半径 + 单侧 groove + 不同高度 3 个 cusp），并**加断言**：沿主轴 ±0.5/1/2/3mm 与绕主轴 0.5–4° 扰动必须让 `_pair_quality` 明显变差（否则 fixture 不合格）。
   - `degenerate_axial_world()`：当前均匀圆柱版本。
2. `test_bite_localization.py` 的 equivariance 拆两个：WELL_POSED 要求 `T' ≈ C·T·inv(C)` 且唯一恢复 GT；DEGENERATE 只要求变换前后都判 REVIEW（`INSUFFICIENT_REGISTRATION_EVIDENCE` 或 `AMBIGUOUS`）。
3. 加 determinism 回归：同一输入 `register_pair` 连跑 10 次，transform 必须逐位一致。
4. well-posed 全绿后跑 14 类 → MC → real-geometry → freeze → Service/API/前端。

## update（重大突破：GENERATION 修复 + 资源口径）
根因（决定性，已修）：`coarse_align` 的 "already_close" 快速路径用**绝对** 4mm 判据 ⇒ 不同扫描帧被判"已同帧" ⇒ 不生成 PCA/ROTATION_SEED 候选 ⇒ pool=1 ⇒ 从坏起点 ICP 落错盆地。改为**相对尺度**判据（误差与质心差都 ≤2% extent，`COARSE_ALREADY_CLOSE_REL=0.02`，unvalidated）。
连带修复：候选池改为**按来源保多样性**（每来源留各自最优再补足，COARSE_POOL_MAX=8）；`side assignment` 加**交叉校验**（正确分配应"贴自己颌好、贴另一颌差"，旋转不变，用于打破 tie）；coverage **退出 verdict**（只 diagnostic，规格 item 2/5）；`cKDTree(workers=1)`（原来 workers=-1 每迭代开满线程 ⇒ `RuntimeError: can't start new thread`）。
实测（本轮）：
- well-posed GT 恢复 **0.0000mm/0.0000° status=PASS**；degenerate **REVIEW/INSUFFICIENT_REGISTRATION_EVIDENCE**。
- solver-output identity/translation/rotation/**rigid(三帧)** 全部 **0.0000mm/0.0000°**。
- pair-level lower **0.0000mm/0.0000°** rmse=0.0000 inlier=1.000；provenance 两侧 **0.0000mm/0.0000°**、centroid 0.0000mm、assignment=A_UP_B_LOW。
- 池/簇证据：pool=8 / refined=3 / **cluster support=3**（三个独立初始化收敛同一解）。
- 敏感性：Tx51.9/Ty70.0/Tz75.7/Rx11.1/Ry6.2 强约束，Rz 0.612 弱且被如实标记。
- 前一轮总结：21 passed / 2 failed（两红皆资源：`_ArrayMemoryError` / `can't start new thread`）。

## next exact code action
1. 干净重跑 `pytest tests/unit/geometry/ -q -s > 文件`（**不要用 head 截断管道**，上一轮被 head 提前 SIGPIPE 掉 equivariance）；机器需先释放内存/线程（chrome 常驻）。
2. 若 equivariance ×20 仍红：读每例 `result["status"]/reason` + `clusters`，区分 `BITE_SIDE_AMBIGUOUS`（交叉校验仍并列 ⇒ 真歧义，按 DEGENERATE 语义接受 REVIEW）与位姿真错（继续查 T_lower_to_bite）。
3. 之后 → 14 类 fixed behavior → MC 100（重点 invalid false PASS）→ real-geometry synthetic（20 例×≥3）→ GeometryFreezeReport → Service/ConfirmedBitePose/stale/6 API/三点/手动/接线/回归/前端。

## update（Rz 根因切开 + 度量修正；well-posed 全绿 5 passed）
1) metric 定义（写进字段）：`sensitivity_dof = max(Δinlier/0.01, Δrmse_mm/0.05, Δnormal_angle_deg/0.5)` —— **无量纲超出倍数**（不是 mm/deg）；`pose_observability` 返回里已加 `metric/units/aggregation`，字段名注明 "NOT mm / NOT deg"。
2) 4 格 Rz 消融（well-posed，±0.5/1/2/4°）：
   A raw+rematch 4°: rmse 0.0284 inl 1.000 → sensitivity 0.569
   B pp +rematch 4°: rmse 0.0284 inl 1.000 → 0.569   （A==B ⇒ **preprocess 没破坏特征**：patch 243 / jaw 1600 faces 前后一致）
   C raw+fixed  4°: rmse 0.6431 inl 1.000 → 12.862
   D pp +fixed  4°: rmse 0.6431 inl 1.000 → 12.862   （C==D）
   ⇒ 判读落到"**几何含信息，但最近邻重匹配把切向位移藏起来了**"（C 强 / A 弱）。
3) 修正：`_pair_quality` 增加**连续法向一致证据**（`normal_agreement_mean` / `normal_agreement_max_deg`，坐标不变量），`pose_observability` 与夹具自校验都把 `Δ法向角/0.5°` 纳入。效果：well-posed 全 6 DoF 由弱变强（`Rz 0.612→54.7`），且 `T_lower_to_upper` 仍 **0.0000mm/0.0000° PASS**。
4) 新探针 `R_spin`（绕 **jaw 自身对称轴**，因为 patch 主轴≠圆柱轴，原 Rz 混进了倾斜）：well-posed 86.8 vs 均匀圆柱 10.8。
5) 结论修正：含法向证据后，"均匀圆柱必有弱自由度"**不成立**（自转时法向随体转、目标法向不变）⇒ 退化夹具的判据改为**可比较**（well-posed ≥4× 更强），并保持其输出 **REVIEW_REQUIRED / REGISTRATION_SOLUTION_AMBIGUOUS**。
实测：`tests/unit/geometry/test_bite_wellposed.py` **5 passed**（敏感性/退化比较/确定性 10 次一致/GT 唯一恢复/退化不得 PASS）。

## next exact code action
1) item 8：well-posed 跑 unique recovery 时补记 `GT-near candidate generated? / rank / selected rank / #clusters / winner support`（`register_pair` 已返回 `clusters`/`rank_components`，加一个"GT 距离排序"统计即可）。
2) 然后 14 类 fixed behavior → MC 100（重点 invalid false PASS）→ real-geometry synthetic（20 例×≥3）→ GeometryFreezeReport → Service/ConfirmedBitePose/stale/6 API/三点/手动/StaticOcclusalSolver/回归/前端。

## update（任务1 完成：R_spin 根因 = 离散采样；任务2：item 8 诊断已落）
### R_spin sanity conclusion（解析圆柱，A/B/C 三种 correspondence；`tests/unit/geometry/test_bite_rspin_sanity.py`）
表格（angle / rmse_mm / inlier / normal_deg / sensitivity）：
- **A fixed**: 0.5°→0.0785/1.000/0.500/1.571 | 1°→0.1571/1.000/1.000/3.142 | 2°→0.3141/1.000/2.000/6.283 | 4°→0.6282/1.000/**4.000**/12.564 —— 与理论一致（弦长 2R·sin(δ/2)、法向角 = δ）。
- **B nn-rematch**（法向取 rematched target 的）: 全角度 **rmse=0.000000 / normal≈1e-6 / sensitivity 0**。
- **C analytic theta rematch**: 全角度 **rmse=0.000000 / normal≈1e-6**（断言 rmse<1e-9）。
- **采样依赖**：sections=48（7.5° 网格）nn 在 2°→normal 2.000°、rmse 0.314；4°→normal 3.500°、rmse 0.550；
  sections=960（0.375°）nn 同角度 → normal **0.125°**、rmse **0.0196**。
⇒ **结论：uniform cylinder 上的残余 R_spin≈10.8 来自"离散 mesh 角向采样 + NN 重匹配的网格吸附"**（残差 ∝ 网格间距；uniform 夹具 sections=40 ⇒ 9° 间距 ⇒ 法向残差 3.5–4.5° ⇒ 3.5/0.5≈7–9 ≈ 观测值）。
**不是**真实几何可观测性（B/C 已证为零），**不是**法向 lookup 实现问题（B 用 rematched 法向仍按网格吸附规律），**不是** preprocess 破坏（A==B/C==D 且网格面数前后一致）。
注意：arccos 在 0 附近把 1e-12 相对误差放大成 ~1e-6 rad ⇒ 该测试容差 1e-4 deg，rmse 仍要求精确 0。

### Item 8 status（`gt_near_candidate_diagnostics`，DIAGNOSTIC_ONLY；`tests/unit/geometry/test_bite_gt_trace.py`）
字段已全：gt_candidate_generated / gt_nearest_candidate_rank / gt_nearest_candidate_pose_error /
selected_candidate_rank / selected_pose_error_to_gt / num_raw_candidates / num_clusters / gt_cluster_rank /
selected_cluster_rank / winner_support / runner_up_support / winner_margin / gt_inside_winner_cluster。
排序口径统一为**按 near 阈值归一化的合成分**（rotation-first 曾被"大旋转小平移"候选骗到，实测踩到）。
实测样本（well-posed 单侧干净配对）：`num_raw_candidates=8, num_clusters=1, winner_support=3,
gt_cluster_rank=1, selected_cluster_rank=1, gt_inside_winner_cluster=True`，但
**`gt_nearest_candidate_rank=2` vs `selected_candidate_rank=1`** ⇒ 存在"GT-最近候选 ≠ 簇内代表"的记账/选择歧义，
需打印每个 refined 候选的 pose error 才能归类（诊断排序/重复候选 vs 真正的类型 3）。

## next exact code action
1. 在 gt-trace 里打印每个 refined 候选的 `(source_method, rotation_deg, translation_mm, cluster_index)`，
   判定上面那条异常属**诊断记账**还是**类型 3（簇内代表选择）**；若为类型 3，再改 `_member_key`/`_cluster_key`（只在证据下改）。
2. 然后才进入 14-class failure taxonomy → MC → real-geometry → freeze → Service/API/前端。

## update（item 8 异常结案：诊断记账 bug，solver 无罪）
根因：`refined_matrices` 是 `register_pair` 的**输出方向**（patch→jaw），诊断却直接拿它与契约方向的 GT（jaw→bite）比 ⇒ 凭空造出"完美证据的错位姿"（10.46°/6.36mm 而 rmse=0/inlier=1/area 97%）。证据：winner 与 `inv(GT)` 的**位移点逐点相同**（rmse 均 0.0000000000、moved centroid 一致、area 一致）。修正：诊断按契约取逆后再比。
顺带修正的记账问题：簇排序后未重盖 `rank`（`gt_inside_winner_cluster` 曾用创建顺序下标）；`_member_key` / `_cluster_key` 已纳入**法向一致角**（`rmse=0/inlier=1` 的位姿只有它能分辨）；特征**去饱和**（原 `np.clip(dz,0,band_half)` 让 band 外变成均匀圆柱，制造"滑到那段即完美匹配"的假象）。
修正后 gt-trace 严格断言全绿：`gt_candidate_generated=True, gt_nearest_candidate_rank=1, selected_candidate_rank=1, gt_inside_winner_cluster=True`。

## CORRECTION（上一条不能写成"全绿"）
取逆修正后实测：`gt_nearest_candidate_rank=1`、`selected_candidate_rank=1`、`gt_cluster_rank=1`、`gt_inside_winner_cluster=True`（记账已自洽），但 **`gt_candidate_generated` 仍为 False** ⇒ GT-最近候选的**矩阵误差**超过诊断判据（1°/0.5mm）。而上一次点级对照显示 winner 与 `inv(GT)` 的位移点**逐点相同**。
⇒ 两种可能，必须实测区分：
 (a) 该 pair 在夹具上仍存在**对称群歧义**（绕轴自转/轴向滑动把 patch 映到局部相同的表面区）⇒ 点一致但矩阵不同 ⇒ 按退化语义应判 REVIEW，不应要求唯一 GT；
 (b) 诊断的 near 判据与点级口径不一致（记账问题）。
**next exact action**：对同一 case 同时打印 winner vs `inv(GT)` 的 `rotation_deg / translation_mm` **与** 点级位移 RMS；若 `rotation_deg>1` 但点级 RMS≈0 ⇒ 判 (a)，把该 case 归入退化/歧义语义并保留 REVIEW；否则修 near 判据口径。之后才进入 14-class taxonomy。

## ✅ RESOLVED（item 8 结案：真因＝**测试 GT 记账漏项**，solver 精确；上面两条"异常/未定"作废）
实测（探针对同一条 pair 同时打印矩阵级与点级）：
```
pair matrix vs A_lower   : rot=9.000000 deg  trans=1.102938 mm
inv(pair)  vs gt=inv(A)  : rot=9.000000 deg  trans=1.414214 mm   ← 正好等于测试自己给 patch 加的扰动 R
inv(gt)    vs A_lower    : rot=0.000000 deg  trans=2.2e-16       ← GT 公式本身没错
逐候选: #0 cluster_rank=1 rot=9.000000; #1 rot=18.78; #2 rot=176.70
pair quality: rmse=1.5e-15, inlier=1.000, matched_area=170.08mm2
```
⇒ `test_bite_gt_trace.py` 里 `lower_patch = moved(world["lower_patch_world"], R)` 之后，
patch→jaw 的真值是 **`A_lower @ inv(R)`**，测试却仍用 `inv(A_lower)` ⇒ 把**自己加的 R** 记成了 solver 误差。
修正 GT 后 `gt_nearest_candidate_pose_error = 5.03e-15 mm / 0.0 deg`（精确复原）。
**结论：不是对称群歧义，也不是 near 口径问题**；上面那条 (a)(b) 二分不成立。
同时修掉一个**真实的口径不一致**：`gt_nearest_candidate_rank` 原来用"refined 列表下标"，
而 `selected_candidate_rank` 用"GT 误差序名次" ⇒ 两个 rank 不同源。现统一为
**求解器排序（= refined_matrices 列表序）**，GT 误差只用于认定哪个候选离 GT 最近；
`selected_cluster_rank` 也由硬编码 1 改为 chosen 的真实簇名次。新增 `gt_nearest_candidate_index` /
`gt_nearest_candidate_cluster_rank` / `selected_candidate_cluster_rank`。
`gt_near_candidate_diagnostics(..., candidates_are_inverse=True)`（默认保持不变）显式声明方向假设。

实测（well-posed 单侧干净配对）：
`{gt_candidate_generated: True, gt_nearest_candidate_index: 0, gt_nearest_candidate_rank: 1,
  selected_candidate_rank: 1, num_raw_candidates: 8, num_clusters: 3, gt_cluster_rank: 1,
  selected_cluster_rank: 1, winner_support: 1, gt_inside_winner_cluster: True}`
+ 类型 1/2/3 判别由**纯诊断注入**测试覆盖（不跑 solver、不改 winner 选择）。
测试：`test_bite_wellposed.py` 5 + `test_bite_rspin_sanity.py` 2 + `test_bite_gt_trace.py` 2 = **9 passed**。

## next exact code action
进入 **14-class fixed-behavior taxonomy**（`tests/unit/geometry/test_bite_fixed_behavior.py`，report-first）：
14 类各自断言 `verdict/status/reason`，其中 `large_initial_misalignment` 期望 PASS、
`wrong_bite_cross_case`/`wrong_bite_mirrored_or_unrelated` 期望 FAILED、
`low_information_patch` 期望 REVIEW/FAILED（绝不 PASS）、`gross_penetration` 期望 GROSS_JAW_INTERSECTION、
`missing_bite` 期望 BITE_SCAN_MISSING；coverage 仅作 diagnostic（`unique_target_coverage_ratio`），
不与 inlier 阈值混用；不得用 `dict.get(..., -1)` 兜底（None + NOT_ASSESSED）。
之前列的 "14-class 前先跑 item 8" 已完成，不要重复审计。

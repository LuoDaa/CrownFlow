# 3Shape 订单包导入

操作员在 3Shape 里做完扫描、导出订单包 ZIP，上传到 WelooAI。
这一层负责**把这个包读成结构化处方 + 可用几何 + 一份可复核的凭据**。

它**不负责**自动出牙冠。导入完成后病例停在"几何就绪、等待设计"，
设计仍要显式创建、参数仍要显式确认。

---

## 1. 包的物理形态：GUID 别名

3Shape 导出时把每个文件改名为 GUID 并去掉扩展名，原文件名记在
`AliasFilenames.xml` 里。所以真实包长这样：

```
AliasFilenames.xml                      <- 唯一的还原依据
{06274A0F-F6C2-4965-B085-AD40ED55B310}  <- 其实是 Scans/Lower/PreparationScan.dcm
{DD2C28B9-F8B9-47C0-A557-B3E17CAE3F8F}  <- 其实是 Scans/Upper/AntagonistScan.dcm
{A8134AFB-FDCA-4204-A44F-0B462ED37634}  <- 其实是 <包根名>.xml（订单主 XML）
{3A955AF9-BCC1-45F4-A97D-20E5E6103634}  <- 其实是 Materials.xml
{00113B62-457C-4157-8682-D37C91CABA59}  <- 其实是 Manufacturers.3ml
```

**「GUID 物理文件名没有扩展名」在本项目里已经咬了四次**，每次都表现为一个
完全不同的症状，值得单独记下来：

| # | 位置 | 症状 |
|---|---|---|
| 1 | 类型判定 | `path.suffix` 恒为空 → 订单 XML 认不出、扫描一个都筛不到（处方全空、scans=0） |
| 2 | 合成 fixture 写 STL | `mesh.export(path)` 报 `exporter not available` |
| 3 | companion STL 加载 | `load_mesh` 报 `NotImplementedError: file_type '...{A6}' not supported` |
| 4 | 建病例时拷几何 | 又一次 `NotImplementedError`，因为 `load_mesh(physical)` 仍靠扩展名分派 |

结论：**凡是要判断格式的地方，都必须用 AliasFilenames 还原出来的逻辑名，
或显式传 `file_type=`**。因此 `geometry/io.load_mesh` 增加了 `file_type`
参数，`package.mesh_physical_path` 负责逻辑名 → 磁盘名。

## 2. 三态扫描判定：加密 ≠ 不支持

`hps_scan.classify_scan` 只有三个状态，没有第四个：

| 状态 | 含义 |
|---|---|
| `DECODED` | HPS 11.1 明文（CC 压缩），顶点可直接解码 |
| `SUPPORTED_BUT_ENCRYPTED` | **容器格式受支持**（`CE ∈ hpsdecode.SUPPORTED_SCHEMAS`），但顶点载荷是 Blowfish 加密的 |
| `UNSUPPORTED` | 容器格式本身读不了 |

真实包实测落在中间那一档：元数据（顶点数、面数、连接性）全部可读，
只有顶点坐标拿不到。

**面向用户的措辞不可改**：必须说"顶点坐标使用 3Shape CE 加密载荷，当前无合法密钥"，
**不能说成"文件损坏 / 不支持格式"**。后者会让操作员去重新导出扫描，
而问题不在导出上。测试对这个措辞有断言（`test_encrypted_scan_reported_as_encrypted_not_unsupported`）。

同理，**绝不提取或绕过 3Shape 的 CE 密钥**。这与项目对 `.matrix4`、`.dentalCAD`
和 59 个加密 `.DCM` 的一贯处理一致：读不出就明确报错，不反向工程厂商格式。

## 3. companion STL：三档绑定，绝不位置绑定

几何拿不到时唯一的出路是**操作员从 3Shape 导出 STL 放进同一个包**。
但导入器不能"看到 STL 就用" —— 必须明确 STL 属于哪个扫描角色。

```
档 1  ORDER_REFERENCE     订单 XML 的扫描条目直接引用了该 mesh
档 2  SIBLING_SAME_STEM   与扫描同目录、同主文件名，仅扩展名不同
                          （如 Scans/Lower/PreparationScan.dcm ↔ .stl）
档 3  未绑定                宁可没有几何，也不猜
```

**明确禁止**「包里第一个 STL = Prep、第二个 = 对颌」这类位置式规则。
`test_companion_not_bound_positionally` 把它钉死：传入不符合约定的名字时，
导入器必须拒绝绑定而不是按顺序硬塞。

补一句实测事实：档 1 在真实包上**通常拿不到** —— 该包的 `TDM_Item_Scan`
的 `FileName`、`ScanName`、`ModelFilename`、`ModelBoundingBoxMin/Max`、
`ModelTransformation` 全是空串。所以实际起作用的是档 2。

### 双 provenance

同时存在 `PreparationScan.dcm`（CE 加密）与 `PreparationScan.stl`（可解码）时，
两个来源都要留，不能用一个覆盖另一个：

```
original_geometry_source  = HPS_CE_ENCRYPTED
preferred_geometry_source = COMPANION_STL
```

`ScanAsset.geometry_state` 据此区分"原件的状态"与"实际可用的几何"：
有可用几何 → `GEOMETRY_READY`，否则回落原件三态。
`ThreeShapePackage.geometry_available_for_design` 更严一层：
**预备体与对颌都要有几何**才算就绪 —— 只有对颌能解仍然不能设计。

## 4. 验收闸门与两步流程

| 检查项 | 内容 |
|---|---|
| `target_fdi` | 订单 `ToothNumber`（Universal）→ FDI |
| `restoration_type` | `toothElementTypeID` → 修复体类型，只确证了 `CROWN` |
| `preparation_scan` / `antagonist_scan` | 两种扫描都在 |
| `scan_geometry` | **按角色**看实际可用的几何 |
| `material_reference` | 订单 `MaterialID` 在 `Materials.xml` 里找得到 |
| `cad_parameters` | 有 `UNRESOLVED` 的参数就 WARN（保留记录、不参与设计） |
| `jaw_separation` | 预备体与对颌不能在同一颌 |

流程拆成两步，**顺序不能颠倒**：

```
POST /api/importers/three-shape/preview   解析 + 跑闸门，零副作用（不建病例、不写网格、不落库）
POST /api/importers/three-shape/import    闸门全过才建病例 + 写几何 + 记凭据
GET  /api/importers/three-shape/records[/{case_id}]   读回导入凭据
```

**闸门不过就 409，不建病例。** 不给"先建着、以后再补几何"的口子 ——
那种病例在库里和真正可设计的病例长得一模一样，迟早有人拿它去跑设计。

### 面板

`preview` 返回一个排好序的 `panel`，前端直接渲染，不重算判定：

```
[✓] 目标牙位    FDI 46（Universal 30）
[✓] 修复体      CROWN（Configuration_11_IDCrownZircon10）
[✓] 材料        Zirkon · Configuration_11_Material1 -> 'Crown Zircon'
[✕] 预备体扫描  颌位 lower · 胡昭/Scans/Lower/PreparationScan.dcm · 几何 SUPPORTED_BUT_ENCRYPTED
[✕] 对颌扫描    颌位 upper · 胡昭/Scans/Upper/AntagonistScan.dcm · 几何 SUPPORTED_BUT_ENCRYPTED
[✓] 颌位分离    prep=lower, antagonist=upper
[✓] 扫描格式    HPS 11.1 / CE（CC 压缩 + 顶点载荷加密）—— 格式受支持；
                顶点坐标使用加密载荷，当前无合法密钥。这不是文件损坏。
[✕] 扫描几何    以下扫描缺可用几何：… 明文可见 131304 顶点 / 260445 面 …
[!] CAD 参数    找到 24 项，其中 7 项语义未确认（保留记录、不参与自动设计）
```

注意 `扫描格式` 与 `扫描几何` 是**两行**，这是刻意的：格式受支持（✓）
和几何拿不到（✕）是两件事，混成一行就会误导。

面板文案是**纯文本**渲染的，所以不要在里面写 markdown 的 `**` 强调
（踩过一次，网页上原样显示星号；有测试守着）。

## 5. 建病例时落什么

- 病例行 + 病例目录（失败则整条回滚，删病例行 + 删目录，`assets` 走 `ON DELETE CASCADE`）
- `lower` / `upper` 两个资产：**由颌位决定**，判不出颌位就拒绝导入。
  不默认塞进 `lower` —— 对颌图出现在下颌上比缺文件更糟。
- `three_shape_import.json`：导入时间、处方、面板快照、闸门结果、扫描明细、
  CAD 参数、PHI 字段、资产清单。**"这个病例从哪来"必须可追溯。**
- `source_package.zip`：源包一并留下，凭据可被复核
- 返回里明确带 `design_created: false` —— 导入**没有**延伸出设计

`support` 资产不会在导入时生成：真实包完全没有牙位坐标信息
（`ModelTransformation`、`ModelBoundingBoxMin/Max` 实测为空），
无法自动定位预备体。仍需对预备体颌做一次 ROI 取冠。

## 6. 实测（真实「胡昭」包）

```
处方      FDI 46 / CROWN / Zirkon / Configuration_11_Material1 / prep=lower / ant=upper
材料记录  TDM_Item_ToothElementType_Crown | 'Crown Zircon'
闸门      7 PASS / 1 FAIL（仅 scan_geometry）
扫描      PreparationScan 131,304 顶点 / 260,445 面  SUPPORTED_BUT_ENCRYPTED
          AntagonistScan  286,638 顶点 / 570,241 面  SUPPORTED_BUT_ENCRYPTED
          顶点载荷熵 7.9999（加密特征）｜连通性熵 2.29（明文）
CAD 参数  24 项，其中 7 项 UNRESOLVED
```

按补救建议补上同主名 STL 后，同一份包变成 `CASE_READY` 并成功建出病例 ——
`test_operator_stl_export_makes_case_importable` 验证了这条路径真的走通，
免得"导出 STL"变成一句空话。

## 7. 现在做不到什么（要如实说）

- **没有合法密钥就读不出 HPS 顶点**。这是厂商格式，不绕过。
- **不能自动定位预备体** → 不能自动生成 `support`，需要一次 ROI 操作。
- **参数映射没接 Resolver**。3Shape 参数只按 [材质 + 修复体子类型] 两级收窄后
  记录，四组语义冲突仍标 `UNRESOLVED`：
  `CementGap`（3Shape 拆成两个）、`WallThickness`（设计目标 vs exocad 最小值）、
  `DistToPrepLine`（差一个量级，`canonical_name` 为 `None`）、
  `RemoveUndercut`（布尔 vs 模式+角度）。
- `TDM_Item_Validation`（对应 `Validation.WallThickness`）尚未提取。

## 8. fixture 与合规

患者数据不进 Git。真实「胡昭」包含 PHI —— 患者姓名同时出现在
`IntOrderID`、`Patient_LastName`、**包根目录名**、以及 `AliasFilenames.xml`
的逻辑路径里（比 exocad 侧多一层）。它只经环境变量在本地受控数据区查找：

```powershell
$env:CROWNFLOW_THREESHAPE_FIXTURE_DIR="D:\code\牙冠自动生成\数据\测试_胡昭"
```

找不到就 skip 并说明理由，而不是静默通过。

单元层用**合成 fixture**（`tests/_three_shape_fixture.py`），结构上复现了
三个真实坑（无扩展名 GUID、字段在孙节点、ItemID 不能短路 MaterialID），
可提交 Git。

## 9. 代码结构

```
backend/app/importers/three_shape/
  models.py           数据模型：三态、ScanAsset（双 provenance）、处方
  _xml.py             ET 解析（禁止正则：xsi:nil 自闭合会破坏配对）
  aliases.py          AliasFilenames 解析、包根名、颌位推断
  order.py            订单 XML → 处方 / 牙位 / PHI（用 descendants 不是 children）
  materials.py        Materials.xml → 材料记录 + CAD 参数（按材质+子类型收窄）
  hps_scan.py         HPS 三态判定（顶点载荷熵判加密，连接性熵证元数据明文）
  geometry_source.py  companion 绑定：三档显式依据，绝不位置绑定
  package.py          import_package / open_package_root / mesh_physical_path
  validator.py        验收闸门（面向用户的措辞在这里，不可改）

backend/app/services/three_shape_import.py   preview / commit / 面板 / next_steps
backend/app/api/importers.py                 HTTP 入口（只收 ZIP）
frontend/src/components/ThreeShapeImport.jsx 导入面板（只渲染，不重算判定）
```

`open_package_root` 的必要性：`import_package` 返回后 ZIP 的解包临时目录已经被删掉，
`ScanAsset.absolute_path` 全部失效。任何要在导入之后继续读网格的调用方，
必须在一个持久的包根目录上下文中工作。

## 10. 相关文档

- `docs/EXOCAD_IMPORTER.md` —— exocad 侧的同类导入，PHI 六层来源
- `docs/PARAMETERS.md` —— 参数四态模型与解析链（3Shape 参数尚未接入）
- `docs/ACCEPTANCE.md` —— 验收套件与 baseline

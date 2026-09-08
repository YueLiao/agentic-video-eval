# agentic-eval — 生成短视频的 Agentic 评测框架

设计文档 v2（first-principles 重写）。作用域：5–15s 的生成视频，多维度打分 + 可审计诊断。

---

## 1. 重新定义问题

现有做法把评测当成**打分**任务：喂 8 帧 + 一段 rubric，让 VLM 吐一个 1–5。
这是错的。评测一段生成短视频实际是**两个不同的任务**，它们需要完全不同的机器：

| | **Conformance（合规）** | **Integrity（自洽）** |
|---|---|---|
| 问题 | 视频有没有做条件要求的事？ | 不管条件是什么，它做出来的东西本身立不立得住？ |
| 性质 | 对一个结构化目标的**核对**（checklist） | 在 4 维体 (x, y, t) 里**搜索缺陷** |
| 需要 | 把 condition 解析成可判定的要求图 | 一个能定位可疑处的搜索索引 |
| 失败样式 | 少了个主体、动作顺序反了、运镜不对 | 六根手指、肢体长度在变、纹理爬行、时间跳变 |
| 典型代价 | 可预测、可并行 | 不可预测、必须自适应 |

把这两件事压成同一个 Likert 分数，正是现有 benchmark 在头部模型上饱和的原因：
合规基本都满分了，自洽的缺陷又因为看不见而没被扣分。

**本框架的核心命题**：评测 = 需求核对 + 缺陷搜索 + 证据裁决。分数是**结果**，不是被直接询问的东西。

---

## 2. 为什么裸 VLM 打不准（机制层面）

不是"prompt 不够好"。是三个可分离的物理限制：

1. **采样失败**。120 帧里取 8 帧均匀采样 = 6.7% 的盲采样。生成缺陷是**瞬态**的，
   集中在运动峰值和转换处。采样没覆盖到的缺陷，无论 prompt 怎么写都不可能被扣分。
2. **分辨率失败**。占画面宽度 6% 的人脸，在 VLM 约 700 token 的图像预算下被降采样到
   ~30px。判断所需的信息**物理上不在模型输入里**。再强的 prompt 也无法恢复。
3. **先验失败**。VLM 预训练几乎全是自然、无瑕疵视频。它对"六指手""骨长随时间变化"
   "纹理爬行"缺乏先验——它不知道该找什么。

每种失败有各自的解法，而且只有这一种解法：

| 失败 | 解法 | 框架里的位置 |
|---|---|---|
| 采样 | **自适应搜索**——先算可疑度，再决定看哪里 | Stage B 可疑度图 |
| 分辨率 | **缩放**——原生分辨率裁剪 + 时间维稠密窗口 | Stage C probe |
| 先验 | **外部先验**——专用检测器与推导不变量 | Stage B 信号 + Stage C 工具 |

Agentic 的意义正在于：**按输入决定用哪几种解法、用多深**。静物 prompt 和舞蹈特写
prompt 走的路径与开销应该完全不同。这不是一个可以用固定 pipeline 表达的东西。

---

## 3. 架构：四个阶段

```
  condition (prompt / 首帧 / 参考主体)
        │
   [A] 条件编译  ── 离线、每个 condition 一次、强 LLM、output-blind
        │           → RequirementGraph + 适用的 Invariants
        ▼
  video ─[B] 可疑度映射 ── 在线、纯信号处理、无 VLM、无学习模型
        │                  → 排序后的 SuspicionLocus 列表（搜索索引）
        ▼
   [C] Agentic 裁决 ── 在线、VLM + 工具、有预算
        │   ├─ Conformance loop（checklist 驱动，可并行）
        │   ├─ Integrity loop（搜索驱动，串行，边看边决定下一步）
        │   └─ Falsification pass（对每条指控出示反证，要求判官辩护或撤回）
        ▼
   [D] 分数合成 ── 确定性、无 VLM
            DefectInventory + SatisfactionVector → 各维分数
```

### Stage A — 条件编译（离线，一次，强 LLM）

condition 编译成 **RequirementGraph**：带类型的节点，每个节点自带判定谓词。

```jsonc
{
  "condition_id": "c0042",
  "nodes": [
    {"nid":"e1","kind":"entity","text":"白色鱼尾狮雕像","required":true,
     "predicate":{"type":"present","min_frames_frac":0.8}},
    {"nid":"a1","kind":"action","subject":"e1","text":"从口中向海面喷水柱",
     "predicate":{"type":"continuous","extent":"whole"}},
    {"nid":"a2","kind":"action","subject":"e3","text":"举起手机合影",
     "predicate":{"type":"occurs","after":null}},
    {"nid":"cam","kind":"camera","text":"缓慢拉远",
     "predicate":{"type":"trajectory","motion":"pull_out","speed":"slow"}},
    {"nid":"n1","kind":"count","subject":"e3","value":"several","tolerance":"vague"}
  ],
  "invariants": ["gravity.freefall","fluid.jet_continuity","human.anatomy","human.identity"],
  "edges": [{"from":"a1","to":"a2","rel":"concurrent"}]
}
```

两个设计点：

* **`invariants` 是推导出来的，不是列举出来的。** condition 里有人 → 自动挂上
  `human.anatomy` / `human.identity`；有自由落体物 → 挂 `gravity.freefall`；
  有刚体接触 → 挂 `contact.no_interpenetration`。这些**不依赖 condition 是否提到**，
  是这个世界本来就该成立的东西。这是"生成模型能不能骗过评测"的关键：
  **policy 无法通过回避某个要求来躲开一条它没得选的不变量。**
* **output-blind 且冻结**。编译只看输入，不看任何候选视频，一次编译供所有模型/所有 seed 复用。
  成本被摊薄到可以忽略，且保证不同模型面对**完全相同**的尺子。

### Stage B — 可疑度映射（在线，纯信号，无 VLM）

这是本设计与已有工作最不同的一环，也是让"搜索"成为可能的前提。
**没有任何 VLM 或学习模型参与**，因此便宜、确定、可在全部帧上跑。

从原始视频算一组互补信号，得到 `(t区间, bbox, 信号类型, 强度)` 的排序列表：

| 信号 | 计算 | 抓什么 |
|---|---|---|
| **运动补偿残差**（最重要） | 用光流把第 t 帧 warp 到 t+1，取残差能量 | 用运动**解释不掉**的变化 = 生成不稳定。这是最干净的"生成瑕疵"信号，与内容运动天然解耦 |
| 光流场散度/旋度异常 | ∇·v, ∇×v 的局部峰 | 物体撕裂、凭空出现/消失、非刚性崩坏 |
| 光流幅度峰 | 逐对平均幅度的局部极大 | 瞬态缺陷集中的时刻 |
| 分块锐度异常 | tile 级 Laplacian 方差，**对 tile 级光流回归后取残差** | 与运动无关的软/糊 = 生成缺陷（运动模糊往往是**对的**，不能一律罚） |
| 纹理爬行能量 | 同一 tile 在时间上的高频能量，运动补偿后 | 表面"沸腾"、AI 感 |
| 亮度/直方图跳变 | 逐帧统计的一阶差分 | 闪烁、镜头跳切 |
| 检测器置信度塌陷 | 人脸/姿态检测器的置信度骤降帧 | **检测器失效本身就是信号**——它在说"这不像人体" |

**关键性质**：这些信号是**归一化后互相竞争**的。输出不是"每种信号一张图"，而是一个
统一排序的 `SuspicionLocus` 列表。Agent 拿到的是一份**按可疑度排好序的待查清单**，
而不是一个固定的采样方案。这就是搜索索引。

### Stage C — Agentic 裁决

两个控制方式不同的 loop，加一个所有人都跳过的第三步。

**C1 Conformance loop**（checklist 驱动）
遍历 RequirementGraph 的节点。每个节点按 `kind` 走一条 probe 模板：
`entity.present` → 全局帧 + 开放词表检测；`action.continuous` → 覆盖时间轴的等距帧；
`camera.trajectory` → 单应分解出的相机轨迹（确定性，根本不需要问 VLM）；
`count` → 定向裁剪 + 计数问句。
可并行、可批量、开销可预测。

**C2 Integrity loop**（搜索驱动）
按可疑度弹出 locus，对每个 locus 做 **probe**：

```
probe(locus) = 空间裁剪(bbox, 原生分辨率, 上采样到 448) × 时间稠密窗口(t-k .. t+k)
             + 该 locus 触发的信号数值
             + 相邻时刻的同区域（作为"正常态"对照）
```

VLM 被问的不是"这段视频质量如何"，而是一个**极窄的问题**：
"这个区域在这几帧里发生了什么？是否存在缺陷？类型？严重度？"
——一个它有能力回答的问题，因为信息终于在它的输入里了。

**自适应终止**：连续 k 个 locus 判为"无缺陷"就停（缺陷是聚集的，不是均匀分布的）；
判为有缺陷则可以**加深**——更小的 bbox、更密的时间窗、更高的分辨率。
这就是"不同输入调用的 loop 不一样"的真正落点。

**C3 Falsification pass**（关键，且是现有工作的空白）
每条被指控的缺陷都要经过一次**反证质询**：把同一区域在相邻时刻的样子、以及
一块公认正常的对照区域，连同指控一起交回判官，要求它**辩护或撤回**。
问题措辞是对抗性的："以下证据是否**不足以**支持这条指控？"

理由：VLM 判官最大的误差源不是漏检，是**幻觉指控**——被要求找问题时它一定能找出问题。
Conformance 那半边可以靠 gold 兜住，Integrity 这半边没有 gold，唯一的防线就是让它面对反证。
**撤回率（retraction rate）是一等公民指标**：它同时诊断判官质量和 prompt 质量。

### Stage D — 分数合成（确定性）

**不问 VLM 要分数。** VLM 的 1–10 是出了名的不校准，且不同 condition 之间不可比。
分数由两个可数、可定位的对象机械地算出来：

```python
DefectInventory = [Defect(type, t_span, bbox, severity∈{minor,major,critical},
                          confidence, evidence_ids, survived_falsification=True)]
SatisfactionVector = {nid: pass | partial | fail}
```

```
conformance_d = Σ w_n · sat(n) / Σ w_n            # 该维度下的要求节点
integrity_d   = 1 − saturate( Σ severity_weight(dfx) · coverage(dfx) )
                 # coverage = 缺陷的时空占比,所以"一帧里的小瑕疵"≠"全程崩坏"
score_d       = combine(conformance_d, integrity_d)   # 见 §5
```

三个后果，都是设计目的：
* **可审计**——每一分的扣减都指向一个 `(帧区间, 框, 类型)`，可以直接被人复核。
* **可复现**——同样的 inventory 必然得到同样的分数，与判官的语气无关。
* **难被 hack**——分数不是 VLM 的一句话，而是一串必须挺过反证的定位化对象。

---

## 4. 工具（按"它解决哪种失败"组织，不按学科）

工具不是越多越好，而是每个都必须对应 §2 的某个物理限制。

**解决采样失败** — 给 Stage B 供料，也可被 agent 直接调用
`motion_compensated_residual` · `flow_field(mag, div, curl)` · `tile_sharpness_vs_flow`
`texture_crawl_energy` · `luma_histogram_jumps` · `detector_confidence_trace`
→ 统一出口：`suspicion_map(video) -> list[SuspicionLocus]`

**解决分辨率失败** — 把信息搬进 VLM 的输入
`crop(bbox, t_span, upscale_to=448)` · `dense_window(t0, t1, n)` ·
`contrast_pair(locus)`（可疑区 + 同区域正常时刻，并排）· `montage(frames, annotate_t=True)`

**解决先验失败** — 提供 VLM 没有的先验
`face(detect | track | identity_drift)` · `pose(keypoints | confidence)` · `hands(21kpt)`
`anatomy_invariants` → 骨长时间变异系数、左右对称性、超范围关节角、人数稳定性
（**纯几何推导，零模型**：真人骨长恒定，生成的不是——这是最便宜且最不可伪造的信号）
`open_vocab_detect(phrases)` · `track(box)` · `depth_order(a, b)`（穿模 = 深度序违反）
`ocr` · `no_reference_iqa`

**通用**
`ask(question, evidence_bundle)` — 唯一的 VLM 入口，也计预算、也进缓存
`compute(goal, arrays)` — 沙箱化的一次性 numpy/OpenCV，用于问不出来但算得出来的东西
（"这个轮子转了几圈""颗粒竖直速度是否递增"）

统一契约：

```python
@dataclass
class ToolResult:
    value: dict          # JSON 可序列化
    images: list[Path]   # 可直接注入 VLM 消息的裁剪/叠加图
    reliability: float   # 0..1,强制字段
    backend: str
    hint: str            # 数值该怎么读,给判官看
```

`reliability < 0.3` 的数值**根本不进 prompt**——展示一个不可靠的数字比不展示更糟，
因为它会被判官当成事实锚定。

---

## 5. 维度

维度不是拍脑袋列的，是 `(Conformance | Integrity) × (语义 | 空间 | 时间 | 人)` 的交叉：

| 维度 | 半边 | 主要来源 |
|---|---|---|
| `semantic_conformance` | C | entity/attribute/count/relation 节点 |
| `action_conformance` | C | action 节点 + 时序边 |
| `camera_conformance` | C | camera 节点（确定性轨迹估计，几乎不用 VLM） |
| `style_conformance` | C | style 节点 |
| `physical_integrity` | I | gravity / contact / fluid / kinematics 不变量 |
| `human_integrity` | I | anatomy / identity / hands 不变量 |
| `temporal_integrity` | I | 运动补偿残差、跳变、闪烁、身份漂移 |
| `appearance_integrity` | I | 纹理爬行、缺陷软化、结构崩坏 |
| `aesthetic_quality` | —— | 唯一诚实的纯主观维；**只用成对比较**，不给绝对分（见下） |

**校准**：绝对 Likert 在判官和人身上都不可靠；成对比较显著更稳
（这是 LLM-judge 校准文献的一致结论）。因此：
* Conformance / Integrity 维度用**可数对象**算分 → 天然可比，不需要校准。
* `aesthetic_quality` 走**成对比较 + Bradley-Terry**，只产出相对排名，不产出绝对分。
* 需要绝对分时，用一小组**锚样本**（每维 3–5 个带分数的定标视频）把 BT 分数映射到刻度上。

---

## 6. 验证（这一节决定项目成不成立）

### 6.1 合成缺陷注入 —— 零标注成本的完美 ground truth

**这是本设计最重要的验证手段，也是现有工作普遍缺的。**

拿**真实**视频（不是生成的），注入**已知类型、已知时空位置、已知强度**的缺陷：

| 注入 | 参数 | 对应真实失败 |
|---|---|---|
| 局部模糊/软化 | bbox, t区间, σ | 生成软化 |
| 时间抽帧/重复帧 | t, n | 帧间跳变、卡顿 |
| 时间扭曲 | 速度曲线 | 动作速度异常 |
| 区域时间乱序 | bbox, 打乱窗口 | 纹理爬行、沸腾 |
| 物体瞬移 | track, Δ位移, t | 物体跳变 |
| 分段人脸替换 | t区间 | 身份漂移 |
| 肢体仿射畸变 | 关键点, 强度 | 结构畸形 |
| 局部亮度脉冲 | bbox, t, 幅度 | 闪烁 |

由此**免费**得到：
* **检出率 / 误报率**（按缺陷类型分）
* **定位精度**：空间 IoU、时间 IoU
* **严重度单调性**：注入强度 ↑ 分数是否单调 ↓（这是 reward 可用性的必要条件）
* **对抗测试**：注入**零**缺陷的干净视频，测**幻觉指控率**——falsification pass 的直接考核

意义：它把"系统能不能发现缺陷"和"人是否同意这个分数"**彻底解耦**。
第一个问题现在可以在没有任何人评的情况下被严格回答，而且今天就能跑。

### 6.2 人评对齐（真实生成视频）

* **成对偏好**（比 Likert 可靠）：模型对 × condition，容平局的强制选择 → BT + bootstrap CI
* **缺陷定位一致性**：人标 `(帧, 框, 类型)`，与系统的 inventory 算 IoU 与类型一致率
  ——这比"分数相关性"细得多，能直接指出是漏检还是误判
* 逐 condition 内的 within-instance Spearman（跨模型），而不是全局相关

### 6.3 稳健性

* **判官互换**：换 VLM 骨干后排名相似度（目标 ρ ≥ 0.90）——结论不能只在一个模型上成立
* **撤回率**：falsification 撤回了多少条指控。太低 = 反证没起作用；太高 = 一轮指控是噪声
* **消融**：`裸VLM` → `+条件编译` → `+可疑度搜索` → `+缩放probe` → `+反证`
  每一步单独量化。若"+可疑度搜索"一步就吃掉大部分增益，那结论就是
  **"搜索比工具重要"**——一个完全成立、更便宜、且值得写出来的结论，
  框架必须造得让这个发现能被看见而不是被掩盖。
* **成本**：每个精度数字旁边必须报单条视频的墙钟与调用数。

---

## 7. 里程碑

| | 内容 | 可演示 |
|---|---|---|
| **M1** | 视频 IO、工具契约、证据总线、轨迹记录；**Stage B 可疑度映射全套信号**；缺陷注入器 | 在注入缺陷的视频上，可疑度图的**定位召回**曲线——纯信号、零 VLM、零标注 |
| **M2** | `ask` / VLM 客户端、probe 构造（裁剪+稠密窗+对照）、Integrity loop、falsification | 检出率/误报率/定位 IoU/幻觉指控率，全部来自注入集 |
| **M3** | Stage A 条件编译、Conformance loop、Stage D 分数合成 | 真实生成视频上的完整多维分数 + 逐视频证据报告 |
| **M4** | 成对比较与 BT、锚样本校准、判官互换、消融矩阵 | 完整消融 + 稳健性表 + 成本表 |
| **M5** | 人评采集与对齐；缺陷定位一致性 | 人锚定的对齐结论 |
| **M6** | reward 接口（inventory 本身就是稠密、定位化的信号，比标量更适合 RL） | reward server + 单调性验证 |

**M1 结束时就有一个可证伪的结论**：如果可疑度图在注入缺陷上的定位召回不明显高于均匀采样，
整个"搜索"论点就是错的，应当立刻收缩范围而不是继续往上叠八个 skill。

---

## 8. 风险

| 风险 | 缓解 |
|---|---|
| **注入缺陷 ≠ 真实生成缺陷**（分布不同，可能过拟合到注入伪影） | 注入集只用于**能力下界**与回归，绝不作为唯一验收；真实缺陷靠 §6.2 人标定位兜底；注入器刻意做多样化与参数随机化 |
| 可疑度图漏检（搜索索引错了，后面再准也没用） | 召回优先于精度：宁可多给 locus 让预算去筛；多信号互补且独立；在注入集上直接量化召回 |
| 判官幻觉指控 | falsification pass；对照证据强制并排；撤回率上报；干净视频的误报率作为硬指标 |
| 判官过度信任工具数值（锚定） | `reliability<0.3` 不进 prompt；rationale 必须描述**看到了什么**而非复述数字；诱饵数值抽检引用率 |
| 条件编译质量成为天花板 | 编译产物可 diff、可版本化、可人工复核；节点级人评一致性作为独立验收 |
| 成本与延迟 | Stage B 全是便宜信号；Stage C 自适应终止；跨候选复用同一编译产物；内容寻址缓存 |
| 静默降级（缺权重/判官挂掉 → 分数退化成常数） | 启动期可用性检查**失败要吵**；每份报告记录实际后端；工具可用集变化在 diff 中告警 |
| 过度工程 | M1 就要求给出可证伪结论；不通过就收缩 |

---

## 附：与既有资产的关系

多数团队都已有一套既存的评测工具链与题库。本框架**不以任何既有工具链为地基**，
以免继承其假设。但两处**可选**互操作是划算的，放在 `agenteval/interop/` 里，缺失不影响主干：

* 旧维度分数的导出适配（让新结果能进已有的区分度/报表脚本做横向参考）
* 已有的生成视频语料与 condition 集可直接当作输入语料

互操作是单向的、可选的、不进主路径。

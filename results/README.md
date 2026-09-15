# results

每个数字都能追到产生它的那一行。指标算法是 VideoAlign `METRICS.md` 的两个口径
(移植在 `src/agenteval/meta/videoalign.py`,与原版逐位对拍过)。

## 主表 `leaderboard.json`

| 运行 | §1 acc(无平局) | acc* | 说明 |
|---|---|---|---|
| `wp_valac` | **67.1%** | 42.1% | video + 生成提示词条件 —— 当前最高 |
| `vp_valac` | 65.9% | 42.0% | video 顺序呈现,温度 0,单次(基线) |
| `vp_vote`  | 57.4% | **44.5%** | 每序 3 次采样共 6 票 —— §1 掉 8.5 点、acc* 最高 |
| `sp_valac` | 44.6% | 42.1% | 上下分屏 —— 分辨率减半跌破 448px 底线 |

对照(同一批 998 对、同一口径):cy 微调的 RM **65.5** · 官方 VideoReward **46.3**。
n=587 非平局,95% 区间约 ±4,所以 67.1 与 65.9 之间的差**不算显著**。

**acc* 都在 42 左右的原因**:我们的输出是三值(A/B/平局),ε 扫描只有两个有意义的
取值,而 val_ac 的平局率是 41.2%——"一律判平局"就有 41.2%。投票版的票差有 7 个取值,
所以它的 acc* 是我们里面最高的。**缺的是连续的分数,不是更好的判断。**

## 逐对结果 `pairs_*.csv`

`pair_id, MQ, family, n_annotations, winner, margin, raw, order_consistent,
path_A, path_B` —— `margin` 是该运行**自己的**聚合量(投票的是票差,单次的是 ±1),
可以直接重算任何口径。

## 验证集与标定

| 文件 | 内容 |
|---|---|
| `detector_matrix.json` | 8 种合成缺陷 × 6 路信号的检出/定位矩阵(5% 误报下,n=32) |
| `freeze_debug_set.json` | 40 例固定卡顿调试集,每例带从渲染像素量出的 band 对比度 |
| `routed_typed_reports.json` | 分流链路的逐片报告(findings / checked / unresolved / score) |
| `signals.json` | 六路信号在 40 条干净片段上的语料标定 |
| `gemma-4-31b-it.json` | 11 项能力探针的结果与由此导出的证据策略 |

## 分层(基线配置)

```
强偏好 AA/BB  n=164   80.5%
弱偏好 A/B    n=423   60.3%   ← 占 72%,总分由它决定
family seed   n=110   69.8%   ← 对照:微调 RM 39.7%(低于随机)
upset 爆冷    n=529   64.1%   ← 档位先验在此为 0,平凡基线约 50%
```

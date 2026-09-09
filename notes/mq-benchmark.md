# MQ pairwise benchmark:能验证什么,不能验证什么

数据位置(只读,不属于本项目):
`.../mq_promptgen/pairing/review_results/rounds/R012_20260908/`,视频根 `cy/rm_videos`。

| 文件 | 对数 | 用途 |
|---|---|---|
| `train.csv` | 14,711 | 训练 |
| `dev.csv` | 2,000 | **选型集**,标定阈值/挑配置用,不报数 |
| `val_ac.csv` | 998 | **报数集**,AC 终判(每对 3–5 人共识) |

三者的视频、prompt、pair_id **零重叠**。视频齐全率 100%。

## 只标了 MQ 一个维度

`VQ` 和 `TA` 列全部是 `invalid`。所以这个 benchmark 能验证的是**运动质量**,
对应本框架的 `运动合理性` 复合视图(motion_magnitude / smoothness / naturalness /
gravity / rigidity)。**人物、语义、画质那几组无法用它验证。**

另一个失望之处:`incoming/benchmark` 里的逐视频绝对缺陷标签
(flicker / identity_drift / limb_instability / motion_not_smooth 等 11 类,
本可与 aspect 逐项对齐)**实际上没有标**——5996 个视频里只有 8 个被标了 `fake_motion`,
其余全零。所以只能做 pairwise 排序对齐,做不了逐维度对齐。

## 三个决定怎么读数的事实

**1. `same` 占 41.2%(val_ac)。**
连续分数永远能分出高低,所以准确率**完全由 tie 阈值 τ 决定**。
τ 必须在 `dev.csv` 上标定后再用于 `val_ac`——在报数集上挑 τ 就是在挑答案。

**2. 人类上限约 72.8%。**
在 `incoming/benchmark` 的单人初标上测:499 个多标注员 pair 中,
完全一致(含强弱档)仅 6.4%,方向一致 25.3%,
**两人都非 tie 时方向一致 72.8%**——后者才是合理的对标值。
6.4% 那个数会误导人:分歧主要来自强弱档和一方判 tie 这两处。
val_ac 是 3–5 人 AC 共识,上限会更高,但仍非 100%。
**任何显著超过 ~80% 的结果应先怀疑评估泄漏**(最常见是排除了 same 对却不声明)。

**3. 两个准确率必须一起报。**
全体准确率(含 same)与非平局方向准确率不可混用;
人类那个 72.8% 是在「两人都非 tie」条件下算的,只能和后者比。

## 成本

998 对 = 1996 个视频。全 8 skill 时每视频约 200s → 约 110 小时,不可行。
只跑 motion 相关 2 个 skill + 单相位,每视频约 50s;
6 并发下 200 个视频约 30–40 分钟,故 100 对可行,全量约需 5–6 小时。

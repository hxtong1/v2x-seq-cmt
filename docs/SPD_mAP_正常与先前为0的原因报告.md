# SPD 上 mAP 正常 vs 先前为 0 的详细原因报告

## 一、结论摘要

| 状态 | mAP | 直接原因 |
|------|-----|----------|
| **之前** | **0** | 评估没有真正跑 nuScenes 的 DetectionEval，而是走了 **fallback**，直接返回全 0 的占位结果。 |
| **现在** | **0.2120**（NDS 0.1419） | 用「按预测的 sample_tokens 从 DB 加载 GT」+「短类别名兼容」后，pred 与 GT 成功对齐，**正常执行** DetectionEval。 |

**根本原因（一句话）**：SPD 的 val 划分与 nuScenes 官方的 val 划分**不是同一套**；原逻辑只认官方 val，导致 GT 与预测的 sample 对不上，代码用“无交集”触发 fallback，从而 mAP 被置 0。修复后改为“用预测的 token 去 DB 里取 GT”，不再依赖官方 val，因此能正确算 mAP。

---

## 二、评估流程（数据到指标）

### 2.1 整体链路

```
测试/评估入口
  → 模型对 val 每帧推理，得到 outputs（list of dict，含 pts_bbox）
  → format_results(outputs)  →  写入 results_nusc.json（key = 每帧的 sample_token）
  → evaluate(outputs)  →  _evaluate_single(result_path)
      ① 加载 NuScenes(version, dataroot)  →  nusc
      ② load_prediction(result_path)  →  pred_boxes（sample_tokens = json 的 key）
      ③ load_gt(nusc, 'val')  →  gt_boxes_full（仅“官方 val”的 sample）
      ④ common = pred_tokens ∩ gt_tokens
      ⑤ 若 common 为空 → 尝试 _load_gt_for_sample_tokens(nusc, pred_tokens)
      ⑥ 若仍无 GT → _evaluate_single_fallback()  →  返回全 0（mAP=0, NDS=0）
      ⑦ 若有 common → 过滤 pred/gt 到 common → add_center_dist / filter_eval_boxes → DetectionEval → 真实 mAP/NDS
```

- **pred_tokens**：来自 val 的 pkl（`spd_infos_temporal_val.pkl`）里每条 `info['token']`，与 `results_nusc.json` 的 key 一致。
- **gt_tokens（官方）**：来自 `load_gt(nusc, 'val')`，只包含「scene 名在 nuScenes 官方 val 名单里」的 sample。

### 2.2 为何 `load_gt(nusc, 'val')` 在 SPD 上等于“没有 GT”

`load_gt` 的内部逻辑（nuscenes 官方）大致为：

1. 取 `create_splits_scenes()['val']`，得到**固定**的 val scene 名列表，例如：  
   `['scene-0003', 'scene-0012', 'scene-0013', ...]`（约 150 个 scene）。
2. 只保留「该 sample 所在 scene 的 `scene_record['name']` 属于上述列表」的 sample。
3. 只对这些 sample 加载标注，得到 `gt_boxes_full`。

而 SPD 的 v1.0-trainval 里：

- **scene 名**是自定义的，如 `"0000"`, `"0001"`, `"0002"` 等，**不在** `create_splits_scenes()['val']` 的名单里。
- 因此 `load_gt(nusc, 'val')` 过滤后，**没有任何** sample 被保留 → `gt_boxes_full.sample_tokens` 为空 → `gt_tokens = ∅`。

于是：

- `common = pred_tokens ∩ gt_tokens = pred_tokens ∩ ∅ = ∅`。
- 原代码在「common 为空」时没有任何补救，直接走 **fallback** → 返回占位全 0 → **这就是之前 mAP 为 0 的真正原因**。

---

## 三、之前 mAP 为 0 的“真正原因”归纳

1. **直接原因**  
   评估走到了 `_evaluate_single_fallback()`，没有跑 nuScenes 的 DetectionEval，所以打印出来的 mAP/NDS 是代码里写死的 0。

2. **触发条件**  
   `common = pred_tokens ∩ gt_tokens` 为空。  
   因为 gt_tokens 来自 `load_gt(nusc, 'val')`，在 SPD 上该函数返回的 sample 数为 0，所以 common 一定为空。

3. **根本原因**  
   - 数据集是 **SPD（V2X-Seq-SPD）**，val 划分和 scene 命名是**自定义**的（如 scene 名 `0000`, `0001`）。  
   - 评估逻辑却**只认 nuScenes 官方的 val 划分**（scene 名如 `scene-0003`, `scene-0012` 等）。  
   - 两套划分不一致 → 官方 load_gt 在 SPD 上得到 0 条 GT → pred 与 GT 无法对齐 → 用“无交集”触发 fallback → **mAP 被置 0**。  

因此：**之前 mAP 为 0，不是因为“模型在 SPD 上检测全错”，而是“根本没有用 SPD 的 GT 做 nuScenes 那套评估”。**

---

## 四、当前 mAP 正常的两个改动

### 4.1 改动一：无交集时用「预测的 sample_tokens」从 DB 加载 GT

**位置**：`custom_nuscenes_dataset.py` 中 `_evaluate_single`。

**逻辑**：

- 当 `common = pred_tokens ∩ gt_tokens` **为空**时，不再直接 fallback。
- 调用 **`_load_gt_for_sample_tokens(nusc, list(pred_tokens), ...)`**：  
  不按 scene 划分过滤，而是**按“预测结果里出现的 sample_token”**，逐个在 nusc 里取 sample，再取该 sample 的标注，组成 GT。
- 若这样得到的 `gt_boxes_by_tokens.sample_tokens` 非空：
  - 用这批 token 作为新的 `common`，用 `gt_boxes_by_tokens` 作为 `gt_boxes_full`；
  - 后续照常：过滤 pred/gt 到 common → add_center_dist → filter_eval_boxes → **DetectionEval** → 得到真实 mAP/NDS。

**效果**：

- SPD 的 val pkl 与 v1.0-trainval 的 `sample.json` 使用**同一套 token**（如 `000870`, `000871` 等）。
- pred_tokens 来自 val pkl，在 `sample.json` 里都能找到 → `_load_gt_for_sample_tokens` 能为 3316 个 val 帧都取到 GT。
- common 非空，评估真正执行 → **mAP 从 0 变为 0.2120，NDS 0.1419**。

### 4.2 改动二：短类别名兼容（SPD 的 category 名）

**位置**：`_load_gt_for_sample_tokens` 内，对 `category_name` 的处理。

**背景**：

- nuScenes 官方的 `category_to_detection_name()` 只认识**长格式**类别名，如 `vehicle.car`, `human.pedestrian.adult`，映射到 `car`, `pedestrian` 等。
- SPD 的 v1.0-trainval 里，category 表是**短名**：`car`, `pedestrian`, `bicycle`, `traffic_cone`。
- NuScenes 在加载时会把 `sample_annotation` 的 category 填成 `category_name`（通过 instance → category），所以在 SPD 上这里拿到的是 `"car"` 等短名。
- `category_to_detection_name("car")` 返回 **None**（因为映射表里 key 是 `"vehicle.car"` 不是 `"car"`），原先逻辑会 **continue**，该框被跳过；若所有框都被跳过，该 sample 就没有 GT，可能导致有效 GT 变少甚至再次触发“无 GT”的 fallback。

**修改**：

- 当 `category_to_detection_name(cat_name)` 为 None 时，**若** `cat_name` 属于检测用的 10 个短名（如 `car`, `pedestrian`, `bicycle`, `traffic_cone`, `truck`, ...），则**直接**把 `cat_name` 当作 `detection_name` 使用，不再丢弃该框。

**效果**：

- SPD 的 4 类（car, pedestrian, bicycle, traffic_cone）都能被正确当成检测类参与评估，GT 框不会被误过滤，mAP 计算完整。

---

## 五、前后对比小结

| 项目 | 之前（mAP=0） | 现在（mAP=0.2120） |
|------|----------------|--------------------|
| **GT 来源** | 仅 `load_gt(nusc, 'val')`，按官方 val scene 过滤 | common 为空时改用 `_load_gt_for_sample_tokens(nusc, pred_tokens)`，按预测的 token 从 DB 取 GT |
| **SPD 上 gt_tokens** | 官方 val 名单与 SPD scene 名不匹配 → **0 条** | 不依赖官方 val，用 pred 的 token 在 sample 表里取 → **3316 条** |
| **common** | 恒为 ∅ | = pred_tokens（全部有 GT） |
| **类别名** | 若走 DB 路径，短名会被 `category_to_detection_name` 判为 None 而丢弃 | 短名显式当作 detection 类，不丢弃 |
| **是否跑 DetectionEval** | 否，走 fallback | 是 |
| **打印的 mAP/NDS** | 占位 0 | 真实 0.2120 / 0.1419 |

---

## 六、总结

- **之前 mAP 为 0 的真正原因**：  
  评估依赖「官方 nuScenes val 划分」取 GT，而 SPD 使用自定义 scene 名与划分，导致 `load_gt(nusc, 'val')` 在 SPD 上返回 0 条 GT；pred 与 GT 的 sample_tokens 无交集，代码进入 fallback，直接返回全 0，**从未在 SPD 上真正跑 DetectionEval**。

- **现在 mAP 正常的原因**：  
  （1）在 common 为空时，用 **`_load_gt_for_sample_tokens`** 按**预测的 sample_tokens** 从同一套 v1.0-trainval DB 里取 GT，使 pred 与 GT 对齐；  
  （2）对 SPD 的**短类别名**做兼容，避免 GT 框被错误过滤。  
  这样在 SPD 上会正常执行 DetectionEval，得到真实 mAP（如 0.2120）和 NDS（如 0.1419）。

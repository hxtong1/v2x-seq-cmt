# AP 为 0 的精确原因定位

## 结论（一句话）

**AP=0 当且仅当评估走到了 `_evaluate_single_fallback()`**，即：没有用 nuScenes 的 DetectionEval 真正算 mAP，而是直接返回了全 0 的占位结果。触发 fallback 的**唯一三条路径**见下。

---

## 1. 评估代码路径（数据 → 代码 → 指标）

```
评估入口: NuScenesDataset.evaluate()
  → format_results(results)  →  result_path = xxx/pts_bbox/results_nusc.json
  → _evaluate_single(result_path)   [CustomNuScenesDataset 重写]
```

### 1.1 `_evaluate_single` 决策树（精确到行）

| 步骤 | 代码位置 | 若失败/条件 | 结果 |
|------|----------|-------------|------|
| 1 | `nusc = NuScenes(version, dataroot=self.data_root)` | **Exception**（如 table 缺失、map log_tokens 断言） | → **fallback，AP=0** |
| 2 | `load_prediction(result_path, ...)` | 文件不存在或格式错误 | → **fallback，AP=0** |
| 3 | `load_gt(nusc, eval_set='val', ...)` | 抛错（如 version 断言） | → **fallback，AP=0** |
| 4 | `common = pred_tokens ∩ gt_tokens` | 官方 val 的 GT 与 pred 无交集（SPD scene 名 ≠ 官方 val） | common 为空 |
| 5 | `_load_gt_for_sample_tokens(nusc, list(pred_tokens))` | 仅当 common 为空时调用 | 用「预测的 token」从 nusc 再取 GT |
| 6 | 若 `gt_boxes_by_tokens.sample_tokens` 仍为空 | pred 的 token 在 `nusc.sample` 里**一个都不存在** | → **fallback，AP=0** |
| 7 | 否则 | 用 common 过滤 pred/gt，走 add_center_dist → filter_eval_boxes → DetectionEval | 正常算 mAP/NDS |

因此：

- **AP=0 的精确原因** = 上述 1、2、3 任一抛错，**或** 6：`_load_gt_for_sample_tokens` 返回的 GT 为空（即 pred 的 sample_tokens 与 v1.0-trainval 的 sample 表无交集）。

---

## 2. 数据流对应关系（为何会“无交集”）

| 数据来源 | 含义 | 来源文件/代码 |
|----------|------|----------------|
| **pred_tokens** | 预测结果里每个 sample 的 key | `results_nusc.json` 的 `results` 的 key；来自 `_format_bbox()` 里 `sample_token = self.data_infos[sample_id]['token']` |
| **data_infos**（val） | 验证集每条 info | `spd_infos_temporal_val.pkl` 的 `data['infos']`，每条有 `info['token']` |
| **gt_tokens（官方）** | 官方 val 的 sample | `load_gt(nusc,'val')` → 只保留 `scene_record['name'] in create_splits_scenes()['val']` 的 sample；SPD 的 scene 名为 `0000` 等，**不在**官方 val 名单 → 恒为空 |
| **nusc.sample 的 token** | DB 里所有 sample | `v1.0-trainval/sample.json` 的 `token` 字段 |

因此：

- **pred_tokens** = val pkl 里 `info['token']` 的集合（与 result json 的 key 一致）。
- **_load_gt_for_sample_tokens 有 GT** 当且仅当：这些 token 在 **v1.0-trainval/sample.json** 里存在，且 nusc 能 `get('sample', token)` 并拿到 `sample['anns']`。

所以：

- **若 val pkl 与 v1.0-trainval 不是同一套 token 体系**（例如 pkl 用 frame_id/别的 id，而 sample.json 用另一套），则 `_load_gt_for_sample_tokens` 会得到 0 个 sample → 走 fallback → **AP=0**。
- **若 NuScenes() 初始化失败**（例如 v1.0-trainval 缺失、map 表缺 `log_tokens` 等），也会在步骤 1 直接 fallback → **AP=0**。

---

## 3. 如何精确定位到是“哪一步”导致的 0

运行仓库里的诊断脚本（见下），会依次检查：

1. **val pkl 的 token 与 v1.0-trainval sample.json 的交集**  
   - 若交集为 0 → 根因就是 **pred 与 DB 的 token 不一致**（数据/生成流程问题）。
2. **NuScenes(version, dataroot) 是否成功**  
   - 若抛错 → 根因是 **DB 加载失败**（路径/表结构/断言）。
3. **用 val 的前若干 token 调用 _load_gt_for_sample_tokens**  
   - 若返回 0 个 sample → 要么 token 不在 sample 表，要么 nusc 内部（如 category_name 等）导致全部被跳过；脚本会区分。

根据脚本输出即可把 AP=0 对应到上述**唯一三条**原因之一。

**说明**：若诊断脚本显示「有交集、NuScenes 成功、能加载 GT」，而历史某次 run 仍得到 AP=0，则很可能是**该次 run 使用的是未合入「按 sample_tokens 加载 GT」+「短类别名兼容」的旧代码**。用当前代码重新跑一次验证集评估即可得到非 0 mAP。

---

## 4. 诊断脚本用法

在项目根目录执行（使用当前环境的 Python，例如 conda 的 cmt）：

```bash
# 示例：指定 val pkl 和 v1.0-trainval 的 data_root
python tools/diagnose_ap_zero.py \
  --val-pkl data/infos/V2X-Seq-SPD-New/vehicle-side/spd_infos_temporal_val.pkl \
  --data-root datasets/V2X-Seq-SPD-New/vehicle-side \
  --version v1.0-trainval
```

脚本会打印：pred 与 DB 的 token 交集数量、NuScenes 是否加载成功、按 pred token 能加载到的 GT 样本数，并给出**结论：AP=0 的精确原因**。

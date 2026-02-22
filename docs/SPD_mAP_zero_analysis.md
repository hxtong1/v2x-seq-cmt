# SPD 数据集上 mAP 为 0 的原因分析

## 结论摘要

**mAP=0 的直接原因**：评估阶段没有在 SPD 验证集上真正跑 nuScenes 指标，而是因为「预测的 sample_tokens 与官方 nuScenes val 的 GT sample_tokens 无交集」，触发了回退逻辑，直接返回了全 0 的占位指标。

根本原因来自：**SPD 是自定义划分 + 自定义目录结构，而当前评估仍依赖“nuScenes DB + 官方 val 划分”**，二者不兼容。

---

## 1. 评估流程与为何得到 0

### 1.1 当前评估逻辑（CustomNuScenesDataset._evaluate_single）

1. 用 `data_root`（SPD 路径）构造 `NuScenes(version=..., dataroot=data_root)`。
2. 用 `load_gt(nusc, eval_set='val')` 从 **nuScenes DB** 里取官方 val 的 GT，得到 `gt_boxes_full`（其 `sample_tokens` 来自 nuScenes 的 `sample.json` + `create_splits_scenes()['val']`）。
3. 预测结果来自 **SPD val 的 pkl**（`spd_infos_temporal_val.pkl`），`sample_tokens` 是 SPD 里每条 `info['token']`（如 `frame_id` 或 uuid）。
4. 计算 `common = pred_tokens ∩ gt_tokens`：
   - 若 **无交集**：打 warning 并调用 `_evaluate_single_fallback()`，返回 **全 0 的 mAP/NDS 等**（即你现在看到的 0）。
   - 若有交集：才用交集的 pred/gt 跑 nuScenes 的 NDS/mAP。

### 1.2 为何 pred 与 gt 的 sample_tokens 无交集？

- **GT 的 sample_tokens**：来自 `nusc = NuScenes(version, dataroot=data_root)`。  
  nuScenes 要求 `dataroot` 下存在 `{version}` 目录（如 `v1.0-trainval/`），且其中有 `sample.json`、`sample_annotation.json` 等。  
  SPD 的 `data_root`（如 `V2X-Seq-SPD-New/vehicle-side`）**没有**这套 nuScenes 表结构，因此通常会在 `NuScenes()` 的 `assert osp.exists(self.table_root)` 处失败；即使不失败，`load_gt(..., 'val')` 也是按 nuScenes 官方 val 的 scene 名单过滤，和 SPD 的 token 不是同一套。
- **Pred 的 sample_tokens**：来自 SPD pkl 的 `info['token']`（SPD 自己的 frame_id / token），与 nuScenes 的 sample token 完全不是同一命名空间。

因此：**在 SPD 上测试时，pred_tokens 与 gt_tokens 的交集为空 → 走 fallback → mAP/NDS 被置为 0。**  
也就是说，**并不是“模型在 SPD 上真的检测全错”，而是“根本没有用 SPD 的 GT 做 nuScenes 那套评估”。**

---

## 2. 数据集与 info 格式（SPD vs NuScenes）

### 2.1 NuScenes 原始设计

- **标注来源**：`dataroot/{version}/sample.json`、`sample_annotation.json` 等表；每个 sample 有唯一 `token`；val 划分由 `create_splits_scenes()` 的 scene 名单决定。
- **单条 info（mmdet3d 从 pkl 里用的）**：通常包含 `token`、`lidar_path`、`sweeps`、`timestamp`、`gt_boxes`、`gt_names`、`num_lidar_pts`、（可选）`valid_flag` 等；`gt_boxes` 为 (N, 7) 或 (N, 9)，LiDAR 系下 (x,y,z,l,w,h,yaw) 或带速度；`gt_names` 为 NuScenes 的类别名（如 `'car'`）。

### 2.2 SPD 的 pkl 格式（spd_to_uniad 生成）

- **结构**：`data = dict(infos=..., metadata=dict(version='v1.0-trainval'))`，与 mmdet3d 的 `load_annotations` 期望的 `data['infos']`、`data['metadata']` 一致。
- **单条 info**（与 NuScenes 对齐的部分）：
  - `token`：来自 `sample_info['token']`（SPD 自己的 id，不是 nuScenes DB 的 token）。
  - `lidar_path`：`pointcloud_path` 的 .pcd 换成 .bin（你这边又通过 `custom_nuscenes_dataset` 改成 .pcd）。
  - `gt_boxes`：`np.concatenate([locs, dims, -rots - np.pi/2], axis=1)` → (N, 7)，格式与 NuScenes 的 LiDAR 框一致。
  - `gt_names`：来自 `annotation['type']`，并经过 `class_names_nuscenes_mappings` 映射成 NuScenes 名（如 'Car'→'car'），因此 **类别名与 config 里 CLASSES 一致**，训练/验证时不会因为类别名不对而全 0。
- **可能差异**：
  - `num_lidar_pts`：SPD 里目前是占位 `np.array([1 for b in boxes])`，若后续用 `use_valid_flag` 或按点数过滤，需要与 NuScenes 的语义一致。
  - **sweeps**：SPD 的 `info['sweeps']` 若结构或 key 与 NuScenes 不完全一致，只影响多 sweep 的加载，不直接导致 mAP 被置 0。

结论：**从“训练/验证用的 info 和 gt 格式”看，SPD pkl 已经按 NuScenes 那套对齐，mAP=0 不是由“info 里缺字段或 gt_names 不对”直接造成的，而是评估根本没用到 SPD 的 GT。**

---

## 3. 数据处理管线中需留意的点

- **LoadPointsFromFile_E2E**：已改为 5 维 (x,y,z,intensity,timestamp)，与 LoadPointsFromMultiSweeps 的 time_dim=4 一致，避免越界。
- **pts_filename**：在 CustomNuScenesDataset 里已从 `lidar_path` 改为 `.pcd` 后缀，与当前数据一致。
- **评估用 GT**：当前唯一来源是 `load_gt(nusc, 'val')`，依赖 nuScenes DB；SPD 的 `data_root` 下没有这套 DB，所以评估端拿不到 SPD 的 GT，只能回退成 0。

---

## 4. 为何说“从数据集本身”导致 0？

可以归纳为两点：

1. **数据组织形式**  
   SPD 没有 nuScenes 的 `{version}/sample.json` 等表结构，也没有“官方 val 划分”的 scene 名单。评估脚本却假定“在 data_root 上能建 NuScenes DB 并取官方 val”，所以一旦用 SPD 的 `data_root`，要么建库失败，要么得到的 GT 与 SPD 的 token 完全不重叠，从而必然走“无交集 → mAP=0”的分支。

2. **sample_tokens 命名空间**  
   预测的 token 来自 SPD pkl 的 `info['token']`，GT 的 token 来自 nuScenes 的 sample 表；两者来自两套体系，没有做“用 SPD 的 token 去 nuScenes 里查 GT”的桥接，因此从设计上就不会有交集，mAP 被置 0 是预期行为。

所以：**“在 SPD 上 mAP 为 0”主要是评估流程与数据集（SPD）不匹配导致的，而不是 SPD 的 info/标注格式在训练阶段有致命错误。**

---

## 5. 建议的修复方向（让 SPD 上真正算 mAP）

要在 SPD 上得到非 0 的 mAP，必须**用 SPD 验证集的 GT 做评估**，而不是依赖 nuScenes 的 val 划分。可选两种思路：

### 方案 A：用 SPD 的 data_infos 构建 GT，再跑 nuScenes 指标（推荐）

- 在 `CustomNuScenesDataset._evaluate_single` 中：
  - 若检测到“pred 与 load_gt 的 token 无交集”（或 NuScenes 建库失败），则**不再直接 fallback 0**。
  - 改为：用当前 dataset 的 `self.data_infos`（即 SPD val）为 **sample_tokens**，按 nuScenes 的 DetectionBox 格式，从 `info['gt_boxes']`、`info['gt_names']` 等构造 `gt_boxes`（EvalBoxes），并保证与 pred 的 sample_tokens 一致。
- 对 `add_center_dist` 所需 ego_pose：可从 `info['ego2global_rotation'/'translation']` 或现有字段构造，或对 SPD 实现一个不依赖 nuScenes 的简化版 center_dist。
- 用现有的 `_DetectionEvalFromBoxes`（或等价逻辑）对 **pred_boxes + 上述 SPD GT** 跑 nuScenes 的 NDS/mAP 计算，即可得到 SPD 上的真实 mAP。

### 方案 B：为 SPD 生成 nuScenes 式 DB 再评估

- 用项目里 `tools/spd_data_converter/spd_to_nuscenes.py` 等脚本，在 SPD 的 `data_root` 下生成 `v1.0-trainval/`（或 val 用到的 version），包含 `sample.json`、`sample_annotation.json` 等，且 **sample token 与当前 pkl 的 `info['token']` 一致**。
- 这样 `NuScenes(version, dataroot=data_root)` 和 `load_gt(..., 'val')` 会得到与 pred 同源的 token，`common` 非空，即可正常跑现有 nuScenes 评估。  
  （需要保证 val 的 scene 划分与 `spd_infos_temporal_val.pkl` 一致。）

---

## 6. 小结表

| 可能原因 | 是否导致当前 mAP=0 | 说明 |
|----------|--------------------|------|
| 评估时 pred 与 GT 的 sample_tokens 无交集 | **是** | 使用 SPD 的 data_root 时无 nuScenes DB 或 val 与 SPD 不一致，走 fallback，直接返回 0。 |
| SPD pkl 缺 NuScenes 所需 info 字段 | **否** | infos/metadata、gt_boxes、gt_names、token 等已按 NuScenes 约定生成。 |
| gt_names 与 CLASSES 不一致 | **否** | 已通过 class_names_nuscenes_mappings 映射为 NuScenes 类名。 |
| gt_boxes 格式不对 | **否** | (N,7) LiDAR 格式与 NuScenes 一致。 |
| 未用 SPD 的 GT 做评估 | **是** | 当前只用了“官方 nuScenes val”的 GT，SPD 没有这套 val，导致必然 0。 |

实施 **方案 A**（在 _evaluate_single 里用 `self.data_infos` 建 SPD 的 GT 并跑 nuScenes 指标）后，SPD 上的 mAP 会反映模型在 SPD 验证集上的真实表现，而不再被固定为 0。

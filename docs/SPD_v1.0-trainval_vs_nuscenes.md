# SPD v1.0-trainval 与 nuScenes 数据集差异分析

## 1. 目录与文件概览

路径：`datasets/V2X-Seq-SPD-New/vehicle-side/v1.0-trainval/`


| 文件                     | 说明                  | 规模/备注                 |
| ---------------------- | ------------------- | --------------------- |
| scene.json             | 场景元数据               | 67 个 scene            |
| sample.json            | 帧（key frame）索引      | 10,426 个 sample       |
| sample_annotation.json | 3D 框标注              | 147,906 条             |
| sample_data.json       | 传感器数据索引（含 LiDAR/相机） | ~145,965 条            |
| instance.json          | 实例（跨帧同一物体）          | ~42,000 条             |
| category.json          | 类别定义                | 4 类                   |
| attribute.json         | 属性定义                | **空 []**              |
| visibility.json        | 可见度等级               | 4 档 (v0-10 ~ v90-100) |
| calibrated_sensor.json | 传感器外参/内参            | 多传感器                  |
| ego_pose.json          | 自车位姿                | 与 sample 一一对应         |
| log.json               | 采集日志                | 若干 log                |
| map.json               | 地图引用                | 占位（filename 为空）       |
| sensor.json            | 传感器类型               | LIDAR_TOP、相机等         |


---

## 2. 与 nuScenes 的主要差异

### 2.1 场景与划分


| 项目       | nuScenes                                         | SPD v1.0-trainval                                     |
| -------- | ------------------------------------------------ | ----------------------------------------------------- |
| scene 命名 | `scene-0003`, `scene-0012`, ...（固定 val/train 名单） | `0000`, `0001`, ...（自定义）                              |
| 划分       | `create_splits_scenes()` 硬编码 train/val/test      | 无官方划分；val 由 pkl 的 `infos` 决定                          |
| 影响       | `load_gt(nusc, 'val')` 只认官方 val scene 名单         | 官方 val 名单与 SPD scene 名无交集 → 需用「按 sample_tokens 加载 GT」 |


### 2.2 类别（category）


| 项目  | nuScenes                                                                   | SPD v1.0-trainval                                      |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| 类别名 | 长格式，如 `vehicle.car`, `human.pedestrian.adult`                              | **短格式**：`car`, `pedestrian`, `bicycle`, `traffic_cone` |
| 类别数 | 10 类检测（含 barrier, bus, construction_vehicle, trailer, truck, motorcycle 等） | **仅 4 类**：car, pedestrian, bicycle, traffic_cone       |
| 影响  | `category_to_detection_name(long_name)` 有映射                                | 短名直接传入会得到 **None**，需在加载 GT 时兼容短名                       |


### 2.3 属性（attribute）


| 项目                                 | nuScenes              | SPD v1.0-trainval                  |
| ---------------------------------- | --------------------- | ---------------------------------- |
| attribute.json                     | 非空，多种属性               | **[] 空**                           |
| sample_annotation.attribute_tokens | 常有 1 个 token          | **[] 空**                           |
| 影响                                 | 检测评估用 attribute 做部分指标 | 纯 LiDAR 检测/跟踪不依赖 attribute，**无影响** |


### 2.4 可见度（visibility）


| 项目                  | nuScenes                                                             | SPD v1.0-trainval                                                                |
| ------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| visibility_token 类型 | 字符串 token，如 `"1"`, `"4"`                                             | **整数**：`2`, `4` 等                                                                |
| visibility 表        | token 为字符串                                                           | token 为 `"1"`,`"2"`,`"3"`,`"4"`                                                  |
| 影响                  | 若代码 `nusc.get('visibility', record['visibility_token'])` 会因类型/键不一致报错 | 当前 **eval 的 load_gt / DetectionEval 未使用 visibility**，纯 LiDAR 检测无影响；2D/可视化脚本可能受影响 |


### 2.5 坐标系与数据格式


| 项目                | nuScenes                            | SPD v1.0-trainval                                                                    |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| sample_annotation | translation/size/rotation 为全局坐标     | 同左，格式一致                                                                              |
| 点云路径              | sample_data.filename 多为 .pcd 或 .bin | filename 写 **velodyne/xxx.bin**，实际项目用 .pcd（在 CustomNuScenesDataset 中重写 pts_filename） |
| ego_pose          | 与 sample 通过 sample_data 关联          | ego_pose.token 与 sample.token **一致**（如 `000009`），一一对应                                |


### 2.6 地图与 log


| 项目  | nuScenes        | SPD v1.0-trainval                        |
| --- | --------------- | ---------------------------------------- |
| map | 有实际地图文件引用       | map.filename 为 **空**，仅 semantic_prior 占位 |
| log | 与真实采集日志对应       | 占位（如 yizhuang06, n015）                   |
| 影响  | 若用 map 做地图先验会失败 | 纯 LiDAR 检测/跟踪 **不依赖 map/log**，无影响        |


---

## 3. 对纯 LiDAR 目标检测的影响


| 差异                     | 影响程度  | 说明                                                                                                                                |
| ---------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------- |
| scene 名与官方 val 不一致     | **高** | 已通过「按 pred 的 sample_tokens 从 nusc 加载 GT」解决，评估可正常出 mAP/NDS                                                                         |
| 类别为短名 + 仅 4 类          | **高** | 需在 `_load_gt_for_sample_tokens` 中兼容：短名直接当 detection_name，否则仍用 `category_to_detection_name`；config 的 CLASSES 需包含这 4 类（当前 10 类包含它们） |
| attribute 为空           | 无     | 检测评估不依赖                                                                                                                           |
| visibility_token 为 int | 无     | load_gt / DetectionEval 未使用                                                                                                       |
| 点云写 .bin 实际用 .pcd      | 无     | 已在数据集与 pipeline 中统一为 .pcd                                                                                                         |
| map/log 占位             | 无     | 检测不依赖                                                                                                                             |


**结论（检测）**：在兼容「按 sample_tokens 加载 GT」+ 「短类别名」后，纯 LiDAR 检测可正常训练与评估。

---

## 4. 对跟踪（Tracking）的影响


| 差异                           | 影响程度 | 说明                                                            |
| ---------------------------- | ---- | ------------------------------------------------------------- |
| instance 与 sample_annotation | 中    | instance 存在且含 category_token，跨帧同一物体有 instance_token，**可支持跟踪** |
| visibility_token 类型          | 低    | 若跟踪脚本用 `nusc.get('visibility', ...)` 可能报错，需转成字符串或兼容 int       |
| attribute 为空                 | 低    | 若跟踪用 attribute 做显示/过滤需默认值                                     |
| 类别仅 4 类                      | 低    | 跟踪类别映射与检测类似，需兼容短名                                             |


**结论（跟踪）**：数据结构上可做跟踪；若直接用 nuScenes 官方跟踪评估脚本，需注意 visibility/attribute/类别名兼容。

---

## 5. 建议修改汇总

1. **评估 GT 加载**（已做）：无官方 val 交集时用 `_load_gt_for_sample_tokens(nusc, pred_boxes.sample_tokens)`。
2. **类别名**：在 `_load_gt_for_sample_tokens` 中，若 `category_to_detection_name(category_name)` 为 None，且 `category_name` 属于 `{'car','truck','bus',...}` 等检测类短名，则直接使用 `category_name` 作为 detection_name。
3. **可选**：若后续用 2D/visibility 相关脚本，将 SPD 的 `visibility_token` 转为字符串写入或读取时做兼容。


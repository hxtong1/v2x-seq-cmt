#!/usr/bin/env python3
"""
诊断 SPD 上 AP=0 的精确原因：检查 pred token 与 v1.0-trainval 的 token 是否一致、
NuScenes 是否加载成功、按 pred token 能否加载到 GT。
"""
import argparse
import json
import os
import sys

import mmcv


def main():
    parser = argparse.ArgumentParser(description='Diagnose AP=0 on SPD')
    parser.add_argument('--val-pkl', type=str,
                        default='data/infos/V2X-Seq-SPD-New/vehicle-side/spd_infos_temporal_val.pkl',
                        help='Path to val pkl (spd_infos_temporal_val.pkl)')
    parser.add_argument('--data-root', type=str,
                        default='datasets/V2X-Seq-SPD-New/vehicle-side',
                        help='Data root (vehicle-side), containing v1.0-trainval/')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--max-tokens', type=int, default=20,
                        help='Max sample tokens to try when loading GT from nusc')
    args = parser.parse_args()

    val_pkl = args.val_pkl
    data_root = os.path.abspath(args.data_root)
    version = args.version
    sample_json_path = os.path.join(data_root, version, 'sample.json')

    print('=' * 60)
    print('AP=0 诊断')
    print('=' * 60)
    print('val_pkl:', val_pkl)
    print('data_root:', data_root)
    print('sample.json:', sample_json_path)
    print()

    # 1) Tokens in val pkl
    if not os.path.isfile(val_pkl):
        print('[FAIL] val pkl 不存在:', val_pkl)
        print('结论: 无法加载 pred 侧 token，评估会失败或使用错误 ann_file。')
        sys.exit(1)
    data = mmcv.load(val_pkl)
    infos = data.get('infos', [])
    if not infos:
        print('[FAIL] val pkl 中 infos 为空。')
        print('结论: 验证集为空或格式错误，评估可能未在预期 val 上运行。')
        sys.exit(1)
    pred_tokens = [info['token'] for info in infos]
    pred_set = set(pred_tokens)
    print('[OK] val pkl 样本数:', len(pred_tokens))
    print('    前 5 个 token 示例:', pred_tokens[:5])
    print()

    # 2) Tokens in v1.0-trainval/sample.json
    if not os.path.isfile(sample_json_path):
        print('[FAIL] sample.json 不存在:', sample_json_path)
        print('结论: NuScenes() 会失败 → 评估走 fallback → AP=0。')
        sys.exit(1)
    with open(sample_json_path) as f:
        samples = json.load(f)
    db_tokens = [s['token'] for s in samples]
    db_set = set(db_tokens)
    print('[OK] v1.0-trainval sample 数:', len(db_tokens))
    print('    前 5 个 token 示例:', db_tokens[:5])
    print()

    # 3) Intersection
    common = pred_set & db_set
    print('pred_tokens 与 DB sample.json 的交集:', len(common), '/', len(pred_set))
    if not common:
        print('[ROOT CAUSE] pred 与 DB 的 token 无交集。')
        print('             _load_gt_for_sample_tokens 会返回 0 个样本 → fallback → AP=0。')
        print('建议: 检查 val pkl 与 v1.0-trainval 是否同源生成、token 是否一致。')
        sys.exit(0)
    print('[OK] 有交集，可继续检查 NuScenes 与 GT 加载。')
    print()

    # 4) NuScenes() load
    try:
        from nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        print('[OK] NuScenes(version=%s, dataroot=...) 加载成功。' % version)
    except Exception as e:
        print('[FAIL] NuScenes() 异常:', e)
        print('结论: 评估在 _evaluate_single 第一步即 fallback → AP=0。')
        sys.exit(0)
    print()

    # 5) _load_gt_for_sample_tokens (use project helper)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from projects.mmdet3d_plugin.datasets.custom_nuscenes_dataset import _load_gt_for_sample_tokens
        from nuscenes.eval.detection.data_classes import DetectionBox
    except Exception as e:
        print('[WARN] 无法导入 _load_gt_for_sample_tokens:', e)
        print('       已确认 token 有交集且 NuScenes 可加载，AP=0 更可能为 fallback 因“无交集”的旧 run 或别处异常。')
        sys.exit(0)

    try_token_list = list(common)[:args.max_tokens]
    gt_boxes = _load_gt_for_sample_tokens(nusc, try_token_list, box_cls=DetectionBox, verbose=False)
    n_loaded = len(gt_boxes.sample_tokens)
    print('_load_gt_for_sample_tokens(前 %d 个共同 token) 加载到 GT 的样本数:' % len(try_token_list), n_loaded)
    if n_loaded == 0:
        print('[ROOT CAUSE] 即使 token 在 sample 表存在，GT 仍为 0（例如 category 名被过滤）。')
        print('             → fallback → AP=0。')
    else:
        print('[OK] 能加载到 GT，评估应能正常算 mAP（若仍为 0，请检查 result 文件与坐标系）。')
    print()
    print('诊断结束。')


if __name__ == '__main__':
    main()

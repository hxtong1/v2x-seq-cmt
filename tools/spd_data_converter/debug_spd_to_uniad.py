#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script for spd_to_uniad.py
Usage:
    PYTHONPATH=. python tools/spd_data_converter/debug_spd_to_uniad.py \
        --data-root ./datasets/V2X-Seq-SPD-New \
        --save-root ./data/infos/V2X-Seq-SPD-New \
        --v2x-side vehicle-side \
        --split-file ./data/split_datas/cooperative-split-data-spd.json \
        --limit-frames 50 \
        --max-workers 1
"""
import argparse
import sys
import traceback
import os.path as osp

# Add project root for imports
sys.path.insert(0, osp.join(osp.dirname(__file__), '../..'))

from tools.spd_data_converter.spd_to_uniad import (
    load_json,
    create_spd_infos,
    _init_create_spd_worker,
    _process_single_frame_worker,
    _generate_sample_infos,
    _get_secene_frame_mappings,
    _get_total_annotations,
    _get_instance_token_mappings,
    get_lidar_ego_global_infos,
    _generate_unvisible_annotations,
    _add_annotation_velocity_prev_next,
    _build_sweep_data_mapping,
    visibility_mappings,
)
from tools.spd_data_converter.spd_to_uniad import to_remove_list_veh  # noqa: F401


def debug_run_single_frame(data_info, worker_state, frame_idx, total):
    """Run single frame with detailed error reporting."""
    try:
        result = _process_single_frame_worker(data_info)
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  [DEBUG] OK frame {frame_idx + 1}/{total} token={data_info['frame_id']}")
        return result
    except Exception as e:
        print(f"\n[DEBUG] ERROR at frame {frame_idx + 1}/{total}")
        print(f"  frame_id: {data_info.get('frame_id', '?')}")
        print(f"  exception: {type(e).__name__}: {e}")
        traceback.print_exc()

        # Inspect annotation 'rotation' format (common cause of 'float' not subscriptable)
        total_annos = worker_state['total_annotations']
        sample_token = data_info['frame_id']
        annos = total_annos.get(sample_token, {})
        print(f"\n  [DEBUG] annotations count: {len(annos)}")
        for i, (k, v) in enumerate(annos.items()):
            rot = v.get('rotation', 'MISSING')
            rot_type = type(rot).__name__
            if rot_type == 'float':
                print(f"    anno[{i}] token={k}: rotation is float = {rot}  <-- likely cause")
            elif rot_type in ('list', 'tuple', 'ndarray'):
                print(f"    anno[{i}] token={k}: rotation = {rot} (len={len(rot)})")
            else:
                print(f"    anno[{i}] token={k}: rotation type={rot_type} value={rot}")
            if i >= 3:
                print(f"    ... (showing first 4)")
                break

        raise


def main():
    parser = argparse.ArgumentParser(description='Debug spd_to_uniad conversion')
    parser.add_argument('--data-root', type=str, default='./datasets/V2X-Seq-SPD-New')
    parser.add_argument('--save-root', type=str, default='./data/infos/V2X-Seq-SPD-New')
    parser.add_argument('--split-file', type=str,
                        default='./data/split_datas/cooperative-split-data-spd.json')
    parser.add_argument('--v2x-side', type=str, default='vehicle-side')
    parser.add_argument('--limit-frames', type=int, default=None,
                        help='Process only first N frames for quick debug')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='1 = sequential, clearer traceback')
    parser.add_argument('--no-save', action='store_true', help='Do not save pkl')
    args = parser.parse_args()

    root_path = osp.join(args.data_root, args.v2x_side)
    data_info_path = osp.join(root_path, 'data_info.json')
    split_data = load_json(args.split_file)
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    data_infos = load_json(data_info_path)
    if args.v2x_side == 'vehicle-side':
        from tools.spd_data_converter.spd_to_uniad import to_remove_list_veh
        data_infos = [x for x in data_infos if x['frame_id'] not in to_remove_list_veh]

    if args.limit_frames:
        data_infos = data_infos[: args.limit_frames]
        print(f"[DEBUG] Limiting to first {args.limit_frames} frames")

    sample_infos, sample_info_mappings = _generate_sample_infos(data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations(
        root_path, data_infos, sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)
    lidar_ego_global_infos = get_lidar_ego_global_infos(root_path, data_infos, args.v2x_side)
    total_annotations = _generate_unvisible_annotations(
        args.v2x_side, sample_info_mappings, secene_frame_mappings,
        instance_token_mappings, total_annotations, lidar_ego_global_infos)
    instance_token_mappings = _get_instance_token_mappings(
        total_annotations, sample_info_mappings)
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(
        total_annotations, instance_token_mappings, lidar_ego_global_infos)
    data_infos_mapping = _build_sweep_data_mapping(data_infos, lidar_ego_global_infos)

    worker_state = {
        'root_path': root_path,
        'v2x_side': args.v2x_side,
        'sample_info_mappings': sample_info_mappings,
        'lidar_ego_global_infos': lidar_ego_global_infos,
        'data_infos_mapping': data_infos_mapping,
        'total_annotations': total_annotations,
        'instance_token_mappings': instance_token_mappings,
        'visibility_mappings': visibility_mappings,
        'max_sweeps': 10,
        'forecasting': False,
        'forecasting_length': 13,
    }

    _init_create_spd_worker(worker_state)
    spd_infos = []
    total = len(data_infos)
    print(f"[DEBUG] Processing {total} frames (max_workers={args.max_workers})...")
    for i, data_info in enumerate(data_infos):
        info = debug_run_single_frame(data_info, worker_state, i, total)
        spd_infos.append(info)

    print(f"\n[DEBUG] Successfully processed {len(spd_infos)} frames")
    if not args.no_save and spd_infos:
        out_path = osp.join(args.save_root, args.v2x_side)
        import mmcv
        os.makedirs(out_path, exist_ok=True)
        train_infos = [x for x in spd_infos
                       if sample_info_mappings[x['token']]['scene_token'] in train_scenes]
        val_infos = [x for x in spd_infos
                     if sample_info_mappings[x['token']]['scene_token'] in val_scenes]
        mmcv.dump(dict(infos=train_infos, metadata=dict(version='v1.0-trainval')),
                  osp.join(out_path, 'spd_infos_temporal_train.pkl'))
        mmcv.dump(dict(infos=val_infos, metadata=dict(version='v1.0-trainval')),
                  osp.join(out_path, 'spd_infos_temporal_val.pkl'))
        print(f"[DEBUG] Saved pkl to {out_path}")


if __name__ == '__main__':
    import os
    main()

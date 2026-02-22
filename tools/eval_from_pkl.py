#!/usr/bin/env python3
"""从已保存的 results pkl 只跑评估并打印 mAP/NDS。"""
import argparse
import os
import sys

import mmcv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config path')
    parser.add_argument('pkl', help='path to results pkl (e.g. work_dirs/.../epoch_60_eval.pkl)')
    parser.add_argument('--jsonfile-prefix', default=None,
                        help='prefix for results_nusc.json; default temp')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False) and getattr(cfg, 'plugin_dir', None):
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_path = _module_dir.replace('/', '.')
        importlib.import_module(_module_path)
    dataset = build_dataset(cfg.data.test)

    outputs = mmcv.load(args.pkl)
    if not isinstance(outputs, list):
        print('pkl root is not a list:', type(outputs))
        sys.exit(1)
    print('Loaded %d results from %s' % (len(outputs), args.pkl))

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.setdefault('metric', 'bbox')
    if args.jsonfile_prefix:
        eval_kwargs['jsonfile_prefix'] = args.jsonfile_prefix

    result_dict = dataset.evaluate(outputs, **eval_kwargs)
    print('\n============ Evaluation ============')
    for k, v in sorted(result_dict.items()):
        if 'NuScenes' in k and ('mAP' in k or 'NDS' in k or 'AP_dist' in k):
            print('%s: %.4f' % (k, v))
    print('====================================\n')
    return result_dict


if __name__ == '__main__':
    main()

import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import os
import sys


def single_gpu_test_tracking(model,
                             data_loader,
                             show=False,
                             out_dir=None,
                             show_score_thr=0.3):
    """Test tracking model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset

    # Default: avoid per-step progress spam in redirected log files.
    # Set COOPTRACK_SHOW_PROGRESS=1 to enable live progress bar in TTY.
    show_progress = os.environ.get(
        "COOPTRACK_SHOW_PROGRESS", "0") == "1" and sys.stdout.isatty()
    prog_bar = mmcv.ProgressBar(len(dataset)) if show_progress else None
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        if prog_bar is not None:
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
    if prog_bar is None:
        print(f"[eval] processed {len(results)}/{len(dataset)} samples.")
    return results

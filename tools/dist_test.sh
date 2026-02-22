#!/usr/bin/env bash
# 用法: dist_test.sh <CONFIG> <CHECKPOINT> <GPUS> [额外参数...]
# 示例: CUDA_VISIBLE_DEVICES=0 tools/dist_test.sh config.py ckpt.pth 1

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
if [[ -z "$GPUS" || ! "$GPUS" =~ ^[0-9]+$ ]]; then
  echo "错误: 第3个参数必须是 GPU 数量（整数），当前为: '$GPUS'" >&2
  echo "用法: $0 <CONFIG> <CHECKPOINT> <GPUS> [额外参数...]" >&2
  exit 1
fi
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 避免 numba 在分布式子进程中初始化失败（mmdet3d 的 kitti_utils 会 import numba）
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}

# 未指定 --out/--eval/--format-only/--show/--show-dir 时默认做 bbox 评估
EXTRA="${@:4}"
if [[ -z "$EXTRA" ]]; then
  EXTRA="--eval bbox"
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    $EXTRA

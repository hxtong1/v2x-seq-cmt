#!/usr/bin/env bash
# 用法: dist_train.sh <CONFIG> <GPUS> [CHECKPOINT] [额外参数...]
# 示例: tools/dist_train.sh config.py 4
# 加载 checkpoint 继续训练: tools/dist_train.sh config.py 4 ckps/epoch_20.pth

CONFIG=$1
GPUS=$2
if [[ -z "$GPUS" || ! "$GPUS" =~ ^[0-9]+$ ]]; then
  echo "错误: 第2个参数必须是 GPU 数量（整数），当前为: '$GPUS'" >&2
  echo "用法: $0 <CONFIG> <GPUS> [CHECKPOINT] [--work-dir ...]" >&2
  exit 1
fi

# 第3个参数：若存在且不以 - 开头则视为 checkpoint，自动加 --resume-from
EXTRA="${@:3}"
if [[ -n "$3" && "$3" != -* ]]; then
  RESUME_FROM="--resume-from $3"
  EXTRA="${@:4}"
else
  RESUME_FROM=""
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch \
    $RESUME_FROM \
    $EXTRA

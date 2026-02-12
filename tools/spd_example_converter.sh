v2x_side=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/V2X-Seq-SPD-New \
    --save-root ./data/infos/V2X-Seq-SPD-New \
    --v2x-side ${v2x_side} \
    --forecasting
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/V2X-Seq-SPD-New \
    --save-root ./datasets/V2X-Seq-SPD-New \
    --v2x-side ${v2x_side}
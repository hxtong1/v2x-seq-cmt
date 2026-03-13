_base_ = ["../_base_/default_runtime.py", "../_base_/datasets/nus-3d.py"]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Infrastructure-side pc_range (keep as set; do not modify)
point_cloud_range = [0, -46.08, -3.0, 102.4, 46.08, 1.0]
post_center_range = [-10.0, -61.2, -10.0, 112.4, 61.2, 10.0]

# voxel_size aligned with lidar track; grid computed from pc_range
voxel_size = [0.16, 0.16, 0.1]
# grid: x=(102.4-0)/0.16=640, y=(46.08-(-46.08))/0.16=576, z=(1-(-3))/0.1=40
grid_size = [640, 576, 40]
sparse_shape = [41, 576, 640]
out_size_factor = 8

class_names = ['car', 'pedestrian', 'bicycle']
class_range = {
    "car": 100,
    "bicycle": 80,
    "pedestrian": 80,
}
new_range_100 = True

# inf_query storage: set save_track_query=True to save track query for coop stage2
save_track_query = False
save_track_query_file_root = 'data/infos/inf_query'

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=True)

file_client_args = dict(backend="disk")

dataset_type = "SPDDataset"
data_root = "./datasets/V2X-Seq-SPD-New/infrastructure-side/"
info_path = "./data/infos/V2X-Seq-SPD-New/infrastructure-side/"
split_datas_file = "./data/split_datas/cooperative-split-data-spd.json"

num_gpus = 4
batch_size = 1
queue_length = 5
num_iters_per_epoch = 7110 // (num_gpus * batch_size)
evaluation = dict(interval=20 * num_iters_per_epoch)

train_pipeline = [
    dict(
        type='LoadPointsFromFile_E2E',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pts_root=data_root),
    dict(
        type='LoadPointsFromMultiSweeps_E2E',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pts_root=data_root,
    ),
    dict(type='LoadAnnotations3D_E2E',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False,
         with_future_anns=False,
         with_ins_inds_3d=True,
         ins_inds_add_1=True,
         with_forecasting=False,
         ),
    dict(
        type='SeqGlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "points",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
        ],
    ),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile_E2E',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pts_root=data_root),
    dict(
        type='LoadPointsFromMultiSweeps_E2E',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pts_root=data_root,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type="CustomCollect3D",
                keys=[
                    "points",
                    "timestamp",
                    "l2g_r_mat",
                    "l2g_t",
                ]
            ),
        ])
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        forecasting=False,
        data_root=data_root,
        ann_file=info_path + 'spd_infos_temporal_train.pkl',
        seq_mode=True,
        load_interval=1,
        queue_length=queue_length,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        split_datas_file=split_datas_file,
        v2x_side='infrastructure_side',
        class_range=class_range,
        new_range_100=new_range_100,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=info_path + 'spd_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        v2x_side='infrastructure_side',
        box_type_3d='LiDAR',
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        class_range=class_range,
        new_range_100=new_range_100,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=info_path + 'spd_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        v2x_side='infrastructure_side',
        box_type_3d='LiDAR',
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        class_range=class_range,
        new_range_100=new_range_100,
    ),
    shuffler_sampler=dict(type="InfiniteGroupEachSampleInBatchSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

model = dict(
    type='CMTCoopTracker',
    pc_range=point_cloud_range,
    queue_length=queue_length,
    save_track_query=save_track_query,
    save_track_query_file_root=save_track_query_file_root,
    class_birth_thresholds=[0.50, 0.40, 0.40],
    train_track=True,
    seq_mode=True,
    batch_size=1,
    video_test_mode=True,
    spatial_temporal_reason=dict(
        history_reasoning=True,
        future_reasoning=False,
        embed_dims=256,
        hist_len=4,
        fut_len=4,
        num_reg_fcs=2,
        code_size=10,
        num_classes=3,
        pc_range=point_cloud_range,
        is_motion=False,
        is_cooperation=False,
        learn_match=False,
        veh_thre=0.10,
        hist_temporal_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        spatial_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
    ),
    runtime_tracker=dict(
        score_threshold=0.5,
        predict_score_thresh=0.05,
        record_threshold=0.4,
        output_threshold=0.1,
        max_age_since_update=5,
        iou_threshold=0.25,
        min_active=1,
        num_track_instance=300,
    ),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(160000, 200000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=sparse_shape,
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64),
                          (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=512,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtLidarHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(
            1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=3, class_names=['car', 'bicycle', 'pedestrian'])
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxTrackCoder',
            post_center_range=[-61.2, -61.2, -10.0, 122.4, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=3,
            score_threshold=0.2,
            iou_thres=0.25,
            with_nms=True),
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=3),
        transformer=dict(
            type='CmtLidarTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0),
        loss_bbox=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=0.5),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='spd',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='spd',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=80
        )))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1/3,
    min_lr_ratio=1e-2
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 48
runner = dict(type='IterBasedRunner',
              max_iters=total_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch*2, max_keep_ckpts=3)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
load_from = None

work_dir = "/home/david/src/S21/aiodrive/mmdetection3d-1/outputs/04_22/overnight/0"

'''
Class Car, Size [4.70046957 1.7028659  2.12782547]
Class Pedestrian, Size [0.5307366  1.72344168 0.45728498]
Class Cyclist, Size [1.80065348 1.91574763 0.65743065]
Class Motorcycle, Size [2.31313721 1.67277009 1.11653532]

NOTE: The location is the "bottom center" of the box
--- CAM COORDS ---
Class Car, Location [-15.35111244   2.9476469  -21.17868254]
Class Pedestrian, Location [-23.19587715   2.4570023   -5.35376136]
Class Cyclist, Location [-22.31175596   3.22564977 -28.96918515]
Class Motorcycle, Location [-21.1882527    2.88725123 -25.43608163]

--- VELO COORDS ---
VELO COORDS: Class Car, Location [-18.87337728  15.3511127   -3.44764689]
VELO COORDS: Class Pedestrian, Location [-3.03752739 23.19587742 -2.95700229]
VELO COORDS: Class Cyclist, Location [-26.66815835  22.31175618  -3.72564976]
VELO COORDS: Class Motorcycle, Location [-23.13713322  21.18825292  -3.38725122]
'''
# Model #############################################################################################################
# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.08, 0.08, 0.1]
point_cloud_range = [-81.92, -64, -4, 81.92, 64, 2] # likely change z-axis range for SECOND-based.
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(80000, 200000)),
    voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[61, 1600, 2048],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=384,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4, #TODO: Important
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator', # the z axis one is the bottom of the box.
            ranges=[
                [-81.92, -64, -3.448, 81.92, 64, -3.448], # Car 
                [-81.92, -64, -2.957, 81.92, 64, -2.957], # Pedestrian
                [-81.92, -64, -3.726, 81.92, 64, -3.726], # Cyclist
                [-81.92, -64, -3.387, 81.92, 64, -3.387]  # Motorcycle
            ],
            sizes=[
                [2.13, 4.70, 1.70], # Car
                [0.46, 0.53, 1.72], # Pedestrian
                [0.66, 1.80, 1.92], # Cyclist
                [1.12, 2.31, 1.67]  # Motorcycle
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[ # always remember to order these correctly
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # motorcycle
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.25,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=500))

# Data #############################################################################################################

# dataset settings
dataset_type = 'AIODriveDataset'
data_root = 'data/aiodrive/data/'
gen_root = 'data/aiodrive/generated/'

class_names = ['Car', 'Pedestrian', 'Cyclist', 'Motorcycle'] # appears like order is important.
input_modality = dict(use_lidar=True, use_camera=False)

# Load one every this many frames for training & validation & testing.
train_load_interval = 5 # ends up with 5k
val_load_interval = 10 # ends up with 3k

db_sampler = dict(
    data_root=gen_root,
    info_path=gen_root + 'aiodrive_velodyne_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10, Motorcycle=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10, Motorcycle=10))

file_client_args = dict(backend='disk')


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=gen_root + 'aiodrive_velodyne_infos_train.pkl',
            split='trainval',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            load_interval=train_load_interval)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=gen_root + 'aiodrive_velodyne_infos_val.pkl',
        split='trainval',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=val_load_interval),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=gen_root + 'aiodrive_velodyne_infos_val.pkl',
        split='trainval',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=val_load_interval))


# Training #############################################################################################################

# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.001 / 2 * 3
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[21, 27])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)

checkpoint_config = dict(interval=1)
evaluation = dict(interval=6)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = "/home/david/src/S21/aiodrive/mmdetection3d-1/outputs/04_22/overnight/0/epoch_6.pth"
workflow = [('train', 1)]
cudnn_benchmark = False
fp16 = dict(loss_scale=512.)
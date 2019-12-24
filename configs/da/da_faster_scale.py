# model settings
model = dict(
    type='DA_FasterRCNN',
    # pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='VGG',
        depth=16,
        num_stages=5,
        out_indices=(4,),
        frozen_stages=2,
        style='pytorch'),
    rpn_head=dict(
        type='RPNHead',
        in_channels=512,
        feat_channels=512,
        anchor_scales=[4, 8, 16],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    da_scale=dict(
        type='DAScaleHead',
        in_channels=512*7*7,
        feat_channels=512,
        grl_weight=-0.1,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=512,
        featmap_strides=[16]),
    bbox_head=dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=512,
            fc_out_channels=4096,
            roi_feat_size=7,
            num_classes=9,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=300,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=300))
# dataset settings
train_dataset_type = 'DADataset'
test_dataset_type = "CityscapesDataset"
s_data_root = '../data/cityscapes/'
t_data_root = '../data/foggy_cityscapes/'
test_data_root = '../data/cityscapes/'
img_norm_cfg = dict(mean=[102.9801, 115.9465, 122.7717], std=[1, 1, 1], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1200, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=train_dataset_type,
        ann_file=s_data_root + 'cityscapes_train.json',
        t_ann_file=t_data_root + 'foggy_cityscapes_train.json',
        img_prefix=s_data_root,
        t_img_prefix=t_data_root,
        s_pipeline=train_pipeline,
        t_pipeline=target_pipeline),
    val=dict(
        type=test_dataset_type,
        ann_file=test_data_root + 'cityscapes_val.json',
        img_prefix=test_data_root,
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        ann_file=test_data_root + 'cityscapes_val.json',
        img_prefix=test_data_root,
        pipeline=test_pipeline))
    # test=dict(
    #     type=test_dataset_type,
    #     ann_file="../data/foggy_cityscapes/" + 'foggy_cityscapes_val.json',
    #     img_prefix="../data/foggy_cityscapes/",
    #     pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/da_faster'
load_from = "../data/vgg16_caffe.pth"
resume_from = None
workflow = [('train', 1)]

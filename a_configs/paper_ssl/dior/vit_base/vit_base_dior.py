_base_ = './2_dataset_schedule_runtime.py'
model = dict(
    type='VIT_ATSS',
    pretrained=None,
    backbone=dict(
        type='MaskedAutoencoderViT',img_size=800,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/airstudio/code/mae/checkpoint/root/log/checkpoint-40.pth')
    ),
    neck=None,
    bbox_head=dict(
        type='PAAHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=20,
        in_channels=768,
        stacked_convs=4,
        feat_channels=768,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
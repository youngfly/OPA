_base_ = './retina_r101_voc.py'
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='GHMC',
            bins=30,
            momentum=0.75,
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='GHMR',
            mu=0.02,
            bins=10,
            momentum=0.7,
            loss_weight=10.0)))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12
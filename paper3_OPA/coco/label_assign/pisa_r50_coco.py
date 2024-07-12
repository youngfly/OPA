_base_ = './retinanet50_coco.py'

model = dict(
    bbox_head=dict(
        type='PISARetinaHead',
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(isr=dict(k=2., bias=0.), carl=dict(k=1., bias=0.2)))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
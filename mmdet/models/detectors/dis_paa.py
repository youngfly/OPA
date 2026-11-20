import torch
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS, LOSSES
from .paa import PAA
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mmcv.cnn import normal_init
import torch.nn as nn
import mmcv
import numpy as np
import os

EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@DETECTORS.register_module()
class Distill_PAA(PAA):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None,
                 init_cfg=None):
        super(Distill_PAA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()
        # todo 采用giou作为回归的损失函数
        loss_dis_reg_cfg = dict(type='NIoULoss', loss_weight=2.0)
        self.loss_distill_reg = build_loss(loss_dis_reg_cfg)
        # todo 针对学生的fpn的特征层的尺度变为原来的一半，此处从configs获取学生fpn通道数
        self.in_channels_adapt = bbox_head.in_channels
        self.out_channels = 256
        self.adapt_conv = nn.Conv2d(self.in_channels_adapt, self.out_channels, 1, padding=0)
        # todo 初始化适应层的权重
        self._init_adapt_weights()

        # todo 是否关系
        self.dis_relation = True
        self.if_cate = True
        self.cate_wei = 0.01
        self.if_geo = False
        self.cate_geo = 1
        # todo 是否响应
        self.dis_respone = False

        # todo 是否蒸馏最终的回归头
        self.dis_reg = False

    def _init_adapt_weights(self):
        normal_init(self.adapt_conv, std=0.01)

    def freeze_models(self):

        self.tea_model.eval()
        for param in self.tea_model.parameters():
            param.requires_grad = False

    def load_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self.tea_model, pretrained, strict=False, logger=logger)
            print("load teacher model success")
        else:
            raise TypeError('pretrained must be a str')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # todo 获取教师网络的输出
        self.tea_model.eval()
        with torch.no_grad():
            tea_x = self.tea_model.extract_feat(img)  # 教师网络的输出
            tea_outs = self.tea_model.bbox_head(tea_x)  # tea_outs[0]分类 tea_outs[1]回归
            tea_loss_inputs = tea_outs + (gt_bboxes, gt_labels, img_metas)  # 获取教师网络的输入，方便后面的获取最后的输出

        # todo 获取学生的网络的输出
        stu_x = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_x)

        # todo 添加适应层，因为采用的学生网络的fpn层通道数发生了改变，因此采用适应层为学生网络的通道数进行调整。
        stu_adapt = self.adapt(stu_x)

        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 导出结果为了教师和学生的结果对齐，导出学生的pos_inds
        pos_cls_stu, pos_reg_stu, pos_labels, iou_bbox, \
        pos_index, pos_gt, pos_inds_flatten, \
        cls_loss_stu, reg_loss_stu = self.bbox_head.loss_fea(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 导出教师组的分类和回归结果平滑组
        pos_cls_tea, pos_reg_tea, \
        cls_loss_tea, reg_loss_tea = self.bbox_head.loss_tea(*tea_loss_inputs, pos_inds_flatten,
                                                             gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 样本集A
        mask_A = cls_loss_tea < cls_loss_stu
        mask_B = reg_loss_tea < reg_loss_stu

        # todo 关系
        if self.dis_relation:
            sup_cate = self.dis_cate(mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea) * self.cate_wei
            sup_loss_cls = torch.clamp(sup_cate, 0, 10)
            losses.update(dict(loss_cate=sup_loss_cls))

            # sup_geo = self.dis_geo(mask_B, pos_reg_stu, pos_reg_tea) * self.cate_geo
            # sup_loss_reg = torch.clamp(sup_geo, 0, 10)
            # losses.update(dict(loss_cate=sup_loss_reg))
        # todo 1 s3选择模块，输出，正样本的分类和回归分数

        # todo 2 cate and geo 模块

        # todo 3 Fea cls reg  模块

        # todo ************************绘图函数*******************************
        # todo 可视化展示不同的mask的结果, 需要选择path, 以及不同的mask_batch
        # path = "/home/airstudio/code/yyr/masks/mask0/"
        # path1 = "/home/airstudio/code/yyr/masks/mask1/"
        # path2 = "/home/airstudio/code/yyr/masks/mask2/"
        # path3 = "/home/airstudio/code/yyr/masks/mask3/"
        # path4 = "/home/airstudio/code/yyr/masks/mask4/"

        # todo 热力图区域 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # todo 热力图的存储路径
        # path_tea_fpn = "/home/airstudio/code/yyr/heatmap/tea_fpn/"
        # path_stu_fpn = "/home/airstudio/code/yyr/heatmap/stu_fpn/"

        # todo 可视化热力图
        # self.show_heatmap(img_metas, tea_x, path_tea_fpn)
        # todo 同时绘制教师和学生的热力图
        # self.show_heatmap_all(img_metas, stu_x, path_stu_fpn, tea_x, path_tea_fpn)

        # todo 为每一个图片的存储一个文件夹
        # self.heatmap_level_t(img_metas, tea_x, path_tea_fpn)
        # self.heatmap_level_all(img_metas, stu_x, path_stu_fpn, tea_x, path_tea_fpn)

        # todo 检测框区域 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # todo 将带有检测框的结果可视化
        # self.show_bboxes(gt_bboxes, img_metas, path4, thickness=3)
        # todo ************************绘图函数*******************************

        # todo 对于fpn层特征图进行蒸馏
        return losses

    # todo cate
    def dis_cate(self, mask_A, pos_cls_stu, pos_labels, pos_index, pos_gt, pos_cls_tea):
        sup_loss = 0.0
        for i in range(len(pos_cls_stu)):
            # todo 计算每一个样本与其他所有的样本之间的关系
            s_feats = pos_cls_stu[i].repeat(len(pos_cls_stu), 1)
            loss_stu = (torch.pow(s_feats - pos_cls_stu, 2) * mask_A.unsqueeze(1)).sum()
            t_feats = pos_cls_tea[i].repeat(len(pos_cls_tea), 1)
            loss_tea = (torch.pow(t_feats - pos_cls_tea, 2) * mask_A.unsqueeze(1)).sum()
            # todo 计算 smooth L1 loss
            loss = self.sml1(loss_stu, loss_tea)
            loss_norm = loss/len(pos_cls_stu)
            sup_loss = sup_loss + loss_norm
        return sup_loss

    def dis_geo(self, mask_B, pos_reg_stu, pos_reg_tea):
        sup_loss = 0.0
        # todo 计算每个样本的蕴含的几何知识
        print(pos_reg_stu)
        s_dia = pos_reg_stu[:, ]
        s_di = pos_reg_stu[:, 2] - pos_reg_stu[:, 2]
        print(s_di)
        # s_angle = torch.cos()
        # s_dia = []
        for i in range(len(pos_reg_stu)):

            pass
        return []

    def sml1(self, a, b):
        x = a - b
        if x.abs() < 1:
            y = 0.5 * x * x
        else:
            y = x.abs() - 0.5
        return y

    # todo 小工具**************************************************************
    # todo 生成文件夹
    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")

    # todo 适应层的网络函数
    def adapt(self, x):
        stu_adapt = []
        for i in range(len(x)):
            stu_ada = []
            stu_ada = self.adapt_conv(x[i])
            stu_adapt.append(stu_ada)
        return stu_adapt

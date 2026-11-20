import torch
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmdet.core import bbox_overlaps
from ..builder import DETECTORS, LOSSES
from .retinanet import RetinaNet
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mmcv.cnn import normal_init
import torch.nn as nn
import time
import mmcv
import numpy as np
import os


@DETECTORS.register_module()
class Prune_Retina(RetinaNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(Prune_Retina, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.if_p = True
        self.percent = 0.5

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        # todo 获取学生的网络的输出
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # todo stu_x=[5,[n,c,h,w]]
        # todo stu_out=[3,[5,[n,c,h,w]]]
        # todo stu_out[0]分类， stu_outs[1]回归, stu_outs[2]centness
        # todo tupe (3, 5, n, c, h ,w), 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size,
        # todo 分类头的c=15, 代表类别， 检测头c=4, 代表4个偏移量， 点头c=1代表， 代表一个中心点
        # todo [h,w] 128, 64, 32 ,16, 8

        # todo 稀疏化训练的

        # todo 对backbone进行剪枝
        # self.prune(self.percent)

        # todo 对FPN进行剪枝
        # self.prune_fpn(self.percent)

        return losses

    # todo 对backbone进行剪枝
    def prune(self, percent):
        # todo 获取Backbone  BN层的参数
        total = 0
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0

        # todo 获取Backbone  BN层的权重
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size

        # todo 权重排序， 获取阈值
        y, i = torch.sort(bn)
        thr_index = int(total * percent)
        thr = y[thr_index]
        # print(len(y))
        pruned = 0
        cfg = []
        cfg_mask = []
        layer = 0
        # todo 不能用K作为层数的索引， 因为对于每一个卷积后面接了norm和relu层
        for k, m in enumerate(self.backbone.modules()):
            # todo BN 层
            if isinstance(m, nn.BatchNorm2d):
                # todo 拷贝BN层的权重
                weight_copy = m.weight.data.abs().clone()
                # todo 小于阈值的权重为0， 大于为1
                mask = weight_copy.gt(thr).float().cuda()
                # todo 被剪枝权重的总数
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                # todo 权重对应的相乘
                m.weight.data.mul_(mask)
                # todo 偏置也对应相乘
                m.bias.data.mul_(mask)

                # todo 第几个BN保留多少。
                cfg.append(int(torch.sum(mask)))
                # todo 第几个BN 保留的weight
                cfg_mask.append(mask.clone())  # 第几个batchnorm 保留的weight
                layer = layer + 1
                # print(self.backbone.res_layers)
                # exit()
                if layer == 1:
                    print('**************block1*************')
                elif layer == 2:
                    print('**************block2*************')
                elif layer == 12:
                    print('**************block3*************')
                elif layer == 25:
                    print('**************block4*************')
                elif layer == 44:
                    print('**************block5*************')

                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(layer, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')

        # todo 剪枝比例
        pruned_ratio = pruned / total
        # print('Pre-processing Successful!' + str(pruned_ratio))

        # todo 剪枝的配置文件, 开始剪枝
        # print(cfg)
        # print(cfg_mask)
        # todo 如果是resnet 50， cfg=54, 相当于在2-5每个Block的后面的第一组卷积后面多了一个升维的卷积，等效于多了四个卷积，
        # 或者说多了4个bn层
        # print(len(cfg))
        # exit()

        # newbackbone = self.backbone.copy()
        # layer_id_in_cfg = 0 # 第几层
        # start_mask = torch.ones(3)
        # end_mask = cfg_mask[layer_id_in_cfg]
        # for [m0, m1] in zip(self.backbone.modules(), newbackbone.modules()):
        #     if isinstance(m0, nn.BatchNorm2d):
        #         # todo np.where 返回的是所有满足条件的数的索引，有多少个满足条件的数就有多少个索引，绝对的索引
        #         # todo 大于0的所有数据的索引，squeeze变成向量
        #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        #         # todo 只要有一个就变成数组的1个
        #         if idx1.size == 1:
        #             idx1 = np.resize(idx1, (1,))
        #         # todo 用经过剪枝的替换原来的权重和偏置
        #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        #         # todo 克隆方差和均值
        #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        #         m1.running_var = m0.running_var[idx1.tolist()].clone()
        #         # todo 下一层
        #         layer_id_in_cfg += 1
        #         # todo 当前要处理的一层
        #         start_mask = end_mask.clone()  # 当前在处理的层的mask
        #         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
        #             end_mask = cfg_mask[layer_id_in_cfg]
        #     elif isinstance(m0, nn.Conv2d):  # 对卷积层进行剪枝
        #         # todo 卷积后面会接bn
        #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        #         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        #         if idx0.size == 1:
        #             idx0 = np.resize(idx0, (1,))
        #         if idx1.size == 1:
        #             idx1 = np.resize(idx1, (1,))
        #         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # 这个剪枝牛B了。。
        #         w1 = w1[idx1.tolist(), :, :, :].clone()  # 最终的权重矩阵
        #         m1.weight.data = w1.clone()
        #     elif isinstance(m0, nn.Linear):
        #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #         if idx0.size == 1:
        #             idx0 = np.resize(idx0, (1,))
        #         m1.weight.data = m0.weight.data[:, idx0].clone()
        #         m1.bias.data = m0.bias.data.clone()

    def prune_fpn(self, percent):
        # todo 获取CONV2d  BN层的参数
        total = 0
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.shape[0]
        # todo m.weight 卷积核的权重样式 是一个四维数组[输入通道， 输出通道， 卷积核第一维， 卷积核第二维]
        bn = torch.zeros(total) # bn =[2048]
        index = 0

        # todo 获取Backbone  BN层的权重
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size

        # todo 权重排序， 获取阈值
        y, i = torch.sort(bn)
        thr_index = int(total * percent)
        # print(thr_index)
        # print(len(y))
        # exit(0)
        thr = y[thr_index]

        pruned = 0
        cfg = []
        cfg_mask = []
        layer = 0
        # todo 不能用K作为层数的索引， 因为对于每一个卷积后面接了norm和relu层
        for k, m in enumerate(self.neck.modules()):
            # todo BN 层
            if isinstance(m, nn.BatchNorm2d):
                # todo 拷贝BN层的权重
                weight_copy = m.weight.data.abs().clone()
                # todo 小于阈值的权重为0， 大于为1
                mask = weight_copy.gt(thr).float().cuda()
                # todo 被剪枝权重的总数
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                # todo 权重对应的相乘
                m.weight.data.mul_(mask)
                # todo 偏置也对应相乘
                m.bias.data.mul_(mask)

                # todo 第几个BN保留多少。
                cfg.append(int(torch.sum(mask)))
                # todo 第几个BN 保留的weight
                cfg_mask.append(mask.clone())  # 第几个batchnorm 保留的weight
                layer = layer + 1
                # print(self.backbone.res_layers)
                # exit()
                if layer == 1:
                    print('**************block1*************')
                elif layer == 2:
                    print('**************block2*************')
                elif layer == 12:
                    print('**************block3*************')
                elif layer == 25:
                    print('**************block4*************')
                elif layer == 44:
                    print('**************block5*************')

                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(layer, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')

        pass

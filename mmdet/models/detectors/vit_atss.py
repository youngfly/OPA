# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import warnings, os
import cv2
import torch
import matplotlib.pyplot as plt
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

@DETECTORS.register_module()
class VIT_ATSS(SingleStageDetector):
    """Implementation of `PAA <https://arxiv.org/pdf/2007.08103.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VIT_ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        if pretrained:
            # print(pretrained)
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        # todo  2, 512, 50, 50
        xs = self.extract_feat(img)
        # todo x=[,[n,c,h,w]]
        # todo xs = [3,[n,c,h,w]]
        losses = self.bbox_head.forward_train(xs, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        # todo ************************绘图函数*******************************
        # todo 热力图的存储路径
        # path_fpn = "/home/airstudio/code/2.17/heatmap/res/"
        # # todo 为每一个图片的存储一个文件夹
        # self.heatmap_level_t(img_metas, xs, path_fpn)
        # todo ************************绘图函数*******************************

        return losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        x = [x]
        outs = self.bbox_head(x)
        return outs

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feats = self.extract_feat(img)
        # feat = [feat]
        results_list = self.bbox_head.simple_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    # todo 绘制为每一张图片生成一个文件夹
    def heatmap_level_t(self, img_metas, heatmaps, path):
        levels = len(heatmaps)
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            self.mkdir(path_pic)
            flip = img_meta['flip']
            img = cv2.imread(filename)
            if flip:
                img = cv2.flip(img, 1)
            plt.imshow(img)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(path_pic + "/image.png")
            for j in range(levels):
                heat = torch.sum(heatmaps[j][i], dim=0).detach().cpu().numpy()
                plt.imshow(heat, interpolation='bilinear', cmap='jet')
                # cm 可选参数 cool hot, gray, cmap=plt.cm.cool。 没有的话默认是绿蓝的图
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def mkdir(self,path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")
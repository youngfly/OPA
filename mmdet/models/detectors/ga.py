from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .paa import PAA
import mmcv

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.visualization import imshow_det_bboxes

@DETECTORS.register_module()
class GA(PAA):
    """Implementation of `PAA <https://arxiv.org/pdf/2007.08103.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      ):
        # todo fpn之后的特征图
        fea_x = self.extract_feat(img)
        # todo 最终分类回归的结果
        outs = self.bbox_head(fea_x)

        # todo regnet_3.2GF_f0.5
        # todo x = [4,[n,c,h,w]]
        # todo c=[96, 192, 432, 1008] [h,w] = ?
        # todo fea_x=[5,[n,c,h,w]]
        # todo c= [128], [h,w] = ?
        # todo outs=[3,[5,[n,c,h,w]]]
        # todo iou 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size
        # todo outs[0]分类， outs[1]回归, outs[2]
        # todo 分类头的c=20, 代表类别， 检测头c=4, 代表4个偏移量， 点头c=1代表， 代表一个中心点

        # todo 计算普通的损失
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)


        return losses


    def show_result(self,
                    img,
                    result,
                    score_thr=0.6,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes

        score_thr = 0.7
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

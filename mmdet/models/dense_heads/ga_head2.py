import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, multiclass_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads import ATSSHead

import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import geatpy as ea
from scipy.optimize import minimize
EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


# class myproblem(ea.Problem):
#     def __init__(self):
#         name = 'MyProblem'
#         M = 1  # todo 初始化Dim
#         maxormins = [1]  # todo 1:最小化， -1最大化
#         Dim = 2  # todo 决策变量维度，
#         varTypes = [0] * Dim
#         lb = [0, -10]  # todo 决策变量下界
#         ub = [5, 0]  # todo 决策变量上界
#         lbin = [0, 1]  # todo 决策变量下边界 0不包含
#         ubin = [1, 0]  # todo 决策变量下边界 1包含
#         ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
#         self.x1 = x1
#         self.x2 = x2
#
#     def aimFunc(self, pop):
#         Vars = pop.Phen
#         b = Vars[:, [0]]
#         c = Vars[:, [1]]
#         pop.ObjV = (np.abs(x1 + b * y1 + c) / np.sqrt(1 + b ** 2)).sum(axis=1, keepdims=True)

def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class GAHead2(ATSSHead):
    def __init__(self,
                 *args,
                 topk=18,
                 score_voting=True,
                 covariance_type='diag',
                 **kwargs):
        self.topk = topk
        self.with_score_voting = score_voting
        self.covariance_type = covariance_type
        super(GAHead2, self).__init__(*args, **kwargs)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when are computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        # todo 将按照层级转换为到图片上的
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]

        pos_losses_list, pll_all, lc, lr = multi_apply(self.get_pos_loss, anchor_list,
                                                       cls_scores, bbox_preds, labels,
                                                       labels_weight, bboxes_target,
                                                       bboxes_weight, pos_inds)

        # todo ga的reassign
        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
            reassign_bbox_weights, num_pos = multi_apply(
                self.ga_reassign,
                pos_losses_list,
                labels,
                labels_weight,
                bboxes_weight,
                pos_inds,
                pos_gt_index,
                anchor_list,
                pll_all,
                lc, lr)
            num_pos = sum(num_pos)

        # todo paa的reassign
        # with torch.no_grad():
        #     reassign_labels, reassign_label_weight, \
        #     reassign_bbox_weights, num_pos = multi_apply(
        #         self.paa_reassign,
        #         pos_losses_list,
        #         labels,
        #         labels_weight,
        #         bboxes_weight,
        #         pos_inds,
        #         pos_gt_index,
        #         anchor_list)
        #     num_pos = sum(num_pos)

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))

        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)
        # todo 计算分类损失
        losses_cls = self.loss_cls(
            cls_scores,
            labels,
            labels_weight,
            avg_factor=max(num_pos, len(img_metas)))  # avoid num_pos=0
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten],
                bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]

            # todo 为 预测的iou生成一个训练目标
            iou_target = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)

            # todo 为 预测的GIOU生成一个训练目标的
            giou_target = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_target, mode='giou', is_aligned=True)

            # todo 预测一个完全分支，包含了分类和回归联合置信度。
            p_cls_score = cls_scores[pos_inds_flatten]
            c_r_targets = p_cls_score * giou_target[..., None]
            joint_targets, _ = torch.max(c_r_targets, 1)

            """  辅助性分支采用的 """
            # todo 采用IOU分支来计算的***********************************
            # todo 计算iou损失，采用交叉熵损失函数
            # losses_iou = self.loss_centerness(
            #     iou_preds[pos_inds_flatten],
            #     iou_target.unsqueeze(-1),
            #     avg_factor=num_pos)
            #
            # # todo 计算box的损失， 采用GIOU, 权重采用iou的权重
            # losses_bbox = self.loss_bbox(
            #     pos_bbox_pred,
            #     pos_bbox_target,
            #     iou_target.clamp(min=EPS),
            #     avg_factor=iou_target.sum())

            # todo 采用GIOU分支来计算 ***********************************
            # todo 计算giou损失，采用交叉熵损失函数
            losses_iou = self.loss_centerness(
                iou_preds[pos_inds_flatten],
                giou_target.unsqueeze(-1),
                avg_factor=num_pos)

            # todo 计算box的损失， 采用GIOU, 权重采用iou的权重
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_target,
                giou_target.clamp(min=EPS),
                avg_factor=giou_target.sum())

            # todo 采用联合分支来计算 ***********************************
            # todo 计算联合损失，采用交叉熵损失函数
            # losses_iou = self.loss_centerness(
            #     iou_preds[pos_inds_flatten],
            #     joint_targets.unsqueeze(-1),
            #     avg_factor=num_pos)
            #
            # # todo 计算box的损失， 采用GIOU, 权重采用iou的权重
            # losses_bbox = self.loss_bbox(
            #     pos_bbox_pred,
            #     pos_bbox_target,
            #     joint_targets.clamp(min=EPS),
            #     avg_factor=joint_targets.sum())




        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou)

    def get_pos_loss(self, anchors, cls_score, bbox_pred, label, label_weight,
                     bbox_target, bbox_weight, pos_inds):
        """Calculate loss of all potential positive samples obtained from first
        match process.

        Args:
            anchors (list[Tensor]): Anchors of each scale.
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            bbox_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_target (dict): Regression target of each anchor with
                shape (num_anchors, 4).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.

        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        if not len(pos_inds):
            return cls_score.new([]),
        anchors_all_level = torch.cat(anchors, 0)
        pos_scores = cls_score[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_target = bbox_target[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]
        pos_anchors = anchors_all_level[pos_inds]
        pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
        # print(pos_label.shape)
        # print(pos_label_weight.shape)
        # exit()
        # to keep loss dimension
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        # todo 将分类和回归的分数分开两个
        pos_all = torch.stack((loss_cls, loss_bbox), 1)
        return pos_loss, pos_all, loss_cls, loss_bbox

    def paa_reassign(self, pos_losses, label, label_weight, bbox_weight,
                     pos_inds, pos_gt_inds, anchors):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                # todo 这个地方应该是每个层级挑选了前9个样本
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            # todo 此处获得了 pos_loss_gmm 重新按照顺序进行排序
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type)
            gmm.fit(pos_loss_gmm)
            gmm_assignment = gmm.predict(pos_loss_gmm)
            scores = gmm.score_samples(pos_loss_gmm)
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)

            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)

        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        # todo 先把所有的label全部置为10
        label[reassign_ids] = self.num_classes
        # todo 被忽略的样本的权重设为0， 即不参与训练
        label_weight[ignore_inds_after_paa] = 0
        # todo 将重新分配后的与之前的进行对比，凡是没有被分配的pos，这次权重全部重新置0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, bbox_weight, num_pos

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper instead of (b).
        # todo 这个地方说明，没有被忽略的样本， 样本被分为正样本和负样本
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores[fgs].topk(1)
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp

    # todo 采用优化算法***************************************************
    def ga_reassign(self, pos_losses, label, label_weight, bbox_weight,
                      pos_inds, pos_gt_inds, anchors, pll_all, lc, lr):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        # todo 分层构建的 mask 筛选出的每一层上的真值
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        # todo pos_level_mask 用来指示每一层属于这一层上的物体 [pos, 5]
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        # todo 为每一个 gt 计算属于自己的正样本。 每个的gt的多个样本被用来的分离
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            # todo 自己设计的储存列表
            pos_inds_svm = []
            pos_loss_svm = []
            pos_lc_svm = []
            pos_lr_svm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                # todo 过滤出属于每一个真值的每一个层级的正样本
                level_gt_mask = level_mask & gt_mask
                # todo 这个地方应该是每个层级挑选了前9个样本（最少的为属于的它的正样本）
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
                # todo 设计自己的二元列表
                pos_inds_svm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_svm.append(pll_all[level_gt_mask][topk_inds])
                pos_lc_svm.append(lc[level_gt_mask][topk_inds])
                pos_lr_svm.append(lr[level_gt_mask][topk_inds])
            # todo 对于的一个gt来说最多只有45个样本
            # pos_inds_gmm = torch.cat(pos_inds_gmm)
            # pos_loss_gmm = torch.cat(pos_loss_gmm)
            # todo 拼接在一起计算
            # pos_loss_svm = torch.cat(pos_loss_svm)
            pos_inds_svm = torch.cat(pos_inds_svm)
            pos_lc_svm = torch.cat(pos_lc_svm)
            pos_lr_svm = torch.cat(pos_lr_svm)
            # todo 看一下的可视化的结果
            # x1 = pos_lc_svm.detach().cpu().numpy()
            # y1 = pos_lr_svm.detach().cpu().numpy()
            x1 = pos_lc_svm
            y1 = pos_lr_svm

            """ 最优化算法 """
            # todo 此处使用了最优化算法进行计算************************************************
            # mean_x = x1.mean()
            # mean_y = y1.mean()
            # b_init = 1
            # c_init = -(mean_x + mean_y)
            #
            # # todo 定义一个最优化的目标
            # def disloss(p):
            #     b, c = p
            #     dis = (np.abs(x1 + b * y1 + c) / np.sqrt(1 + b ** 2)).sum(0)
            #     return dis
            #
            # # todo 调用优化算法求解的
            # # todo Nelder-Mead BFGS TNC CG Powell
            # mymin = minimize(disloss, (b_init, c_init), method='Powell')
            # # todo 优化后的最小值,以及解析解
            # outs = mymin.fun
            # results = mymin.x
            # todo 此处使用了最优化算法进行计算************************************************
            """ 最优化算法 """

            """ 遗传学算法 """
            # todo 此处使用了遗传学算法进行计算************************************************
            class myproblem(ea.Problem):
                def __init__(self):
                    name = 'MyProblem'
                    M = 1 # todo 初始化Dim
                    maxormins = [1] # todo 1:最小化， -1最大化
                    Dim = 2 # todo 决策变量维度，
                    varTypes = [0] * Dim
                    lb = [0, -10] # todo 决策变量下界
                    ub = [5, 0] # todo 决策变量上界
                    lbin = [0, 1] # todo 决策变量下边界 0不包含
                    ubin = [1, 0] # todo 决策变量下边界 1包含
                    ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

                def aimFunc(self, pop):
                    Vars = pop.Phen
                    b = torch.tensor(Vars[:, [0]]).cuda()
                    c = torch.tensor(Vars[:, [1]]).cuda()
                    pop.ObjV = (torch.abs(x1 + b * y1 + c) / torch.sqrt(1 + b ** 2)).sum(axis=1, keepdims=True)
                    pop.ObjV = pop.ObjV.detach().cpu().numpy()

            problem = myproblem()
            Encoding = 'RI' # todo 编码方式
            NIND = 150 # todo 种群规模
            Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
            population = ea.Population(Encoding, Field, NIND)

            myalgorathm = ea.soea_DE_rand_1_bin_templet(problem, population)
            myalgorathm.MAXGEN = 100 # todo 最大进化代数
            myalgorathm.mutOper.F = 0.5 # todo 差分进化中的参数F
            myalgorathm.recOper.XOVR = 0.7 # todo 重组概率
            myalgorathm.logTras = 0
            myalgorathm.verbose = False
            myalgorathm.drawing = 0

            [BestIndi, population] = myalgorathm.run()
            outs = BestIndi.ObjV[0][0]
            results = BestIndi.Phen[0]
            # todo 此处使用了遗传学算法进行计算************************************************
            """ 遗传学算法 """
            # print(results)
            # plt.scatter(x1, y1)
            # plt.plot([0, -float(results[1]/results[0])], [-results[1], 0])
            # plt.savefig('result2.jpg')
            # plt.close()
            # exit()
            # todo 为所有的点计算到经过原点的但是的斜率和分界线的斜率相同的直线
            device = pos_inds_svm.device
            b1 = torch.tensor(results[0])
            c1 = torch.tensor(0)
            distance = torch.abs(x1 + b1 * y1 + c1)/torch.sqrt(1 + b1 **2)
            # todo 计算的两条直线的距离
            line_dis = torch.abs(torch.tensor(results[1]))/ torch.sqrt(1 + b1 ** 2)
            # todo 边界判断， 处于两条平行线中间的是正样本的，否则就为负样本。
            cate = (distance > line_dis) + 0
            # todo 修复ga
            if len(pos_inds_svm) < 2:
                continue
            # todo 重新分配标签
            device = pos_inds_svm.device
            # cate = torch.from_numpy(cate).to(device)
            ga_pos_inds_temp, ga_ignore_inds_temp = self.ga_separation_scheme(
                cate, pos_inds_svm)
            pos_inds_after_paa.append(ga_pos_inds_temp)
            ignore_inds_after_paa.append(ga_ignore_inds_temp)
            # todo *****************************************************
            # todo 修复gmm
            # fix gmm need at least two sample
            # if len(pos_inds_gmm) < 2:
            #     continue

            # todo 此处获得了 pos_loss_gmm 重新按照顺序进行排序, 将损失的按照升序进行排序
            # device = pos_inds_gmm.device
            # pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            # pos_inds_gmm = pos_inds_gmm[sort_inds]
            # pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            # # todo 初始化数据准备
            # min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            # means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            # weights_init = np.array([0.5, 0.5])
            # precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            # if self.covariance_type == 'spherical':
            #     precisions_init = precisions_init.reshape(2)
            # elif self.covariance_type == 'diag':
            #     precisions_init = precisions_init.reshape(2, 1)
            # elif self.covariance_type == 'tied':
            #     precisions_init = np.array([[1.0]])
            # if skm is None:
            #     raise ImportError('Please run "pip install sklearn" '
            #                       'to install sklearn first.')
            # gmm = skm.GaussianMixture(
            #     2,
            #     weights_init=weights_init,
            #     means_init=means_init,
            #     precisions_init=precisions_init,
            #     covariance_type=self.covariance_type)
            # gmm.fit(pos_loss_gmm)
            # gmm_assignment = gmm.predict(pos_loss_gmm)
            # scores = gmm.score_samples(pos_loss_gmm)
            # gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            # scores = torch.from_numpy(scores).to(device)
            #
            # pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
            #     gmm_assignment, scores, pos_inds_gmm)
            # pos_inds_after_paa.append(pos_inds_temp)
            # ignore_inds_after_paa.append(ignore_inds_temp)
            # todo *****************************************************
            # print(gmm_assignment)
            # print(pos_inds_gmm)
            # print(pos_inds_temp)
            # print(ignore_inds_temp)

        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        # todo 先把所有的label全部置为10
        label[reassign_ids] = self.num_classes
        # todo 被忽略的样本的权重设为0， 即不参与训练
        label_weight[ignore_inds_after_paa] = 0
        # todo 将重新分配后的与之前的进行对比，凡是没有被分配的pos，这次权重全部重新置0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, bbox_weight, num_pos

    def ga_separation_scheme(self, ga_assignment, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper instead of (b).
        # todo 这个地方说明，没有被忽略的样本， 样本被分为正样本和负样本
        # todo gmm_ass = [num] scores = [num]
        # todo 首先把assignment=0 的找出来 fgs应该是前景, 最多是45个
        fgs = ga_assignment == 0
        # todo 设置等待分配的正样本
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        # todo numel() 返回一个矩阵中元素的个数 nonzero()返回的是数组中非零元素的位置
        # todo 当有正样本的时候进行分配
        if fgs.nonzero().numel():
            # todo 只要fgs存在东西就是正样本
            pos_inds_temp = pos_inds_gmm[fgs]
            # _, pos_thr_ind = scores[fgs].topk(1)
            # # todo 将前景的列表设置 ，pos_inds_gmm:正样本的编号， 先从正样本的中筛选出前景， 再从所有的前景中筛选出从零到阈值的样本
            # # todo 由于是按照损失的排序，损失小的都在前面，即为正样本。
            # pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            # todo 将忽略的样本的列表设置为空
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp
    # todo ************************************************************

    def get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=None,
            label_channels=1,
            unmap_outputs=True,
    ):
        """Get targets for PAA head.

        This method is almost the same as `AnchorHead.get_targets()`. We direct
        return the results from _get_targets_single instead map it to levels
        by images_to_levels function.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels (list[Tensor]): Labels of all anchors, each with
                    shape (num_anchors,).
                - label_weights (list[Tensor]): Label weights of all anchor.
                    each with shape (num_anchors,).
                - bbox_targets (list[Tensor]): BBox targets of all anchors.
                    each with shape (num_anchors, 4).
                - bbox_weights (list[Tensor]): BBox weights of all anchors.
                    each with shape (num_anchors, 4).
                - pos_inds (list[Tensor]): Contains all index of positive
                    sample in all anchor.
                - gt_inds (list[Tensor]): Contains all gt_index of positive
                    sample in all anchor.
        """

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (labels, label_weights, bbox_targets, bbox_weights, valid_pos_inds,
         valid_neg_inds, sampling_result) = results

        # Due to valid flag of anchors, we have to calculate the real pos_inds
        # in origin anchor set.
        pos_inds = []
        for i, single_labels in enumerate(labels):
            pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero().view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                gt_inds)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        This method is same as `AnchorHead._get_targets_single()`.
        """
        assert unmap_outputs, 'We must map outputs back to the original' \
                              'set of anchors in PAAhead'
        return super(ATSSHead, self)._get_targets_single(
            flat_anchors,
            valid_flags,
            gt_bboxes,
            gt_bboxes_ignore,
            gt_labels,
            img_meta,
            label_channels=1,
            unmap_outputs=True)

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    iou_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        This method is almost same as `ATSSHead._get_bboxes()`.
        We use sqrt(iou_preds * cls_scores) in NMS process instead of just
        cls_scores. Besides, score voting is used when `` score_voting``
        is set to True.
        """
        assert with_nms, 'PAA only supports "with_nms=True" now'
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        batch_size = cls_scores[0].shape[0]

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_preds = []
        for cls_score, bbox_pred, iou_preds, anchors in zip(
                cls_scores, bbox_preds, iou_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            iou_preds = iou_preds.permute(0, 2, 3, 1).reshape(batch_size,
                                                              -1).sigmoid()
            # todo scores = [anhcors, cls_num], iou_pred.shape = [anchors, 1]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[1] > nms_pre:
                max_scores, _ = (scores * iou_preds[..., None]).sqrt().max(-1)
                # max_scores, _ = iou_preds.sqrt().max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                iou_preds = iou_preds[batch_inds, topk_inds]
            else:
                anchors = anchors.expand_as(bbox_pred)

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_preds.append(iou_preds)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
        batch_mlvl_iou_preds = torch.cat(mlvl_iou_preds, dim=1)
        batch_mlvl_nms_scores = (batch_mlvl_scores *
                                 batch_mlvl_iou_preds[..., None]).sqrt()

        det_results = []
        for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                              batch_mlvl_nms_scores):
            det_bbox, det_label = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=None)
            if self.with_score_voting and len(det_bbox) > 0:
                det_bbox, det_label = self.score_voting(
                    det_bbox, det_label, mlvl_bboxes, mlvl_scores,
                    cfg.score_thr)
            det_results.append(tuple([det_bbox, det_label]))

        return det_results

    def score_voting(self, det_bboxes, det_labels, mlvl_bboxes,
                     mlvl_nms_scores, score_thr):
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 5), each dimension means
                (x1, y1, x2, y2, score).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            mlvl_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            mlvl_nms_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            mlvl_iou_preds (Tensor): The predictions of IOU of all boxes
                before the NMS procedure, with shape (num_anchors, 1)
            score_thr (float): The score threshold of bboxes.

        Returns:
            tuple: Usually returns a tuple containing voting results.

                - det_bboxes_voted (Tensor): Remaining boxes after
                    score voting procedure, with shape (k, 5), each
                    dimension means (x1, y1, x2, y2, score).
                - det_labels_voted (Tensor): Label of remaining bboxes
                    after voting, with shape (num_anchors,).
        """
        candidate_mask = mlvl_nms_scores > score_thr
        candidate_mask_nonzeros = candidate_mask.nonzero()
        candidate_inds = candidate_mask_nonzeros[:, 0]
        candidate_labels = candidate_mask_nonzeros[:, 1]
        candidate_bboxes = mlvl_bboxes[candidate_inds]
        candidate_scores = mlvl_nms_scores[candidate_mask]
        det_bboxes_voted = []
        det_labels_voted = []
        for cls in range(self.cls_out_channels):
            candidate_cls_mask = candidate_labels == cls
            if not candidate_cls_mask.any():
                continue
            candidate_cls_scores = candidate_scores[candidate_cls_mask]
            candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
            det_cls_mask = det_labels == cls
            det_cls_bboxes = det_bboxes[det_cls_mask].view(
                -1, det_bboxes.size(-1))
            det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4],
                                               candidate_cls_bboxes)
            for det_ind in range(len(det_cls_bboxes)):
                single_det_ious = det_candidate_ious[det_ind]
                pos_ious_mask = single_det_ious > 0.01
                pos_ious = single_det_ious[pos_ious_mask]
                pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                pos_scores = candidate_cls_scores[pos_ious_mask]
                pis = (torch.exp(-(1 - pos_ious) ** 2 / 0.025) *
                       pos_scores)[:, None]
                voted_box = torch.sum(
                    pis * pos_bboxes, dim=0) / torch.sum(
                    pis, dim=0)
                voted_score = det_cls_bboxes[det_ind][-1:][None, :]
                det_bboxes_voted.append(
                    torch.cat((voted_box[None, :], voted_score), dim=1))
                det_labels_voted.append(cls)

        det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
        det_labels_voted = det_labels.new_tensor(det_labels_voted)
        return det_bboxes_voted, det_labels_voted

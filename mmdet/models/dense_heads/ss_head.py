import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, multiclass_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads import ATSSHead

EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


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
class SSHead(ATSSHead):
    """Head of PAAAssignment: Probabilistic Anchor Assignment with IoU
    Prediction for Object Detection.

    Code is modified from the `official github repo
    <https://github.com/kkhoot/PAA/blob/master/paa_core
    /modeling/rpn/paa/loss.py>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2007.08103>`_ .

    Args:
        topk (int): Select topk samples with smallest loss in
            each level.
        score_voting (bool): Whether to use score voting in post-process.
        covariance_type : String describing the type of covariance parameters
            to be used in :class:`sklearn.mixture.GaussianMixture`.
            It must be one of:

            - 'full': each component has its own general covariance matrix
            - 'tied': all components share the same general covariance matrix
            - 'diag': each component has its own diagonal covariance matrix
            - 'spherical': each component has its own single variance
            Default: 'diag'. From 'full' to 'spherical', the gmm fitting
            process is faster yet the performance could be influenced. For most
            cases, 'diag' should be a good choice.
    """

    def __init__(self,
                 *args,
                 topk=9,
                 score_voting=True,
                 covariance_type='diag',
                 **kwargs):
        # topk used in paa reassign process
        self.topk = topk
        self.with_score_voting = score_voting
        self.covariance_type = covariance_type
        super(SSHead, self).__init__(*args, **kwargs)

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
        # todo 获取训练的目标 由于在设置中 pos的iou阈值设置为0.1，采用maxiou先挑选一大批的anchor，但是pos_anchor基本位于gt附近
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
        # todo 将每个level上面的的分数转换为图片上
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        # todo 计算所有正样本的损失， 利用cls和reg分别计算得分
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)
        # todo 重新分配样本
        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
            reassign_bbox_weights, num_pos = multi_apply(
                self.paa_reassign,
                pos_losses_list,
                labels,
                labels_weight,
                bboxes_weight,
                pos_inds,
                pos_gt_index,
                anchor_list)
            num_pos = sum(num_pos)
        # convert all tensor list to a flatten tensor
        # todo 将所有的结果的全部展平, 此处将所有的batch的结果拼接在一起
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
        # todo 只有正样本时才计算回归部分的损失
        if num_pos:
            pos_bbox_pred = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten],
                bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            # todo 为 预测的iou生成一个训练目标，目标就是直接计算预测 box和真值之间的iou重叠度
            iou_target = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            # todo 计算iou损失，采用交叉熵损失函数
            losses_iou = self.loss_centerness(
                iou_preds[pos_inds_flatten],
                iou_target.unsqueeze(-1),
                avg_factor=num_pos)
            # todo 计算box的损失， 采用GIOU, 权重采用iou的权重
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_target,
                iou_target.clamp(min=EPS),
                avg_factor=iou_target.sum())
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
        # todo 获取所有的正样本的分数
        pos_scores = cls_score[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_target = bbox_target[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]
        # todo 获取所有的正样本的anchor
        pos_anchors = anchors_all_level[pos_inds]
        pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
        # to keep loss dimension
        # todo 计算所有的分类损失
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # todo 计算所有的回归的损失
        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return pos_loss,

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
        # todo 前期数量统计
        label = label.clone()
        label_weight = label_weight.clone()
        bbox_weight = bbox_weight.clone()
        # todo 真值的数量
        num_gt = pos_gt_inds.max() + 1
        # todo 层级的数量
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        # todo 不断叠加，返回元素的梯形累计和, 等效于不断返回anchor的数量叠加
        inds_level_interval = np.cumsum(num_anchors_each_level)
        # todo 准备每个层级的正样本mask, 根据上一笔划分的每个层级的anchor数量，利用mask将所有的正样本序号分配
        # todo
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        # todo paa之后的pos_inds
        pos_inds_after_paa = [label.new_tensor([])]
        # todo paa之后的忽略的样本
        ignore_inds_after_paa = [label.new_tensor([])]
        # todo 根据每个真值来计算样本的分配, 由于topk设置为9，所以说为每个gt在每个层级匹配9个样本
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            # todo pos_gt_inds = 所有的样本中的为正样本的位置为对应真值的号码， gt_ind真值的编号
            # todo gt_mask = 从所有的真值中筛选出和每一个真值对应的所有正样本
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                # todo 提取每个层级中某个真值对应的所有
                #  真样本
                level_gt_mask = level_mask & gt_mask
                # todo 这个地方应该是每个层级挑选了前9个样本（得分最小的，损失最少，说明匹配最好）
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                # todo 为每个层级添加参与gmm的正样本的编号
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                # todo 为每个层级添加参与gmm的正样本的值
                pos_loss_gmm.append(value)
            # todo 把一个list拼接起来
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            # todo gmm至少需要两个样本
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            # todo 此处获得了 pos_loss_gmm 重新按照顺序进行排序， 默认是按照升序排列，从小到大，
            # todo 原本是分数越高，约为正样本，在原始论文中的是按照损失来计算，损失越小约为正样本,
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            # todo 所以按照重新排序后来说，排在前面的为正样本
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            # todo pos_loss_gmm的数量不会超过45, level 5 * top 9  = 45
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            # todo 获取其中损失的最小值以及最大值
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            # todo 初始化高斯混合估计的参数
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            # todo 设置斜方差矩阵 默认为diag
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            # todo 设置高斯混合估计模型, 分为两类，正负样本
            # todo 收敛阈值为0.001， 最大迭代次数为100
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type,
                reg_covar=1e-5
            )
            # todo 使用EM算法估算模型参数
            gmm.fit(pos_loss_gmm)
            # todo 使用估计好的模型来预测 数据样本的标签, 0和1分别代表两个类别
            gmm_assignment = gmm.predict(pos_loss_gmm)
            # todo 计算每个样本的加权对数的概率
            scores = gmm.score_samples(pos_loss_gmm)
            # todo 将numpy转换为tensor
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)
            # todo 根据高斯分布将数据分为两个个部分，正样本和忽略的赝本
            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm)
            # todo 添加新的已经分配完成的样本
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)
        # todo 获取一个图片上面所有的正样本，以及不参与训练的样本
        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        # todo 对比原本正样本的序号和重新分配之后的序号，如果不相同，则得到剔除的正样本
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        # todo 获取重新分配的样本的id
        reassign_ids = pos_inds[reassign_mask]
        # todo 将所有的剔除的正样本的标签设置为背景
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
        # todo score 是一个根据高斯模型预测出来的每个anchor的概率
        # todo gmm_ass = [num] scores = [num]
        # todo 首先把assignment=0 的找出来 fgs应该是前景, 最多是45个
        fgs = gmm_assignment == 0
        # todo 设置等待分配的正样本
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        # todo numel() 返回一个矩阵中元素的个数 nonzero()返回的是数组中非零元素的位置
        # todo 当有正样本的时候进行分配
        if fgs.nonzero().numel():
            # todo 从前景中选择得分最高的一个作为阈值,并返回它的索引。得分最高的
            _, pos_thr_ind = scores[fgs].topk(1)
            # todo 将前景的列表设置 ，pos_inds_gmm:正样本的编号， 先从正样本的中筛选出前景， 再从所有的前景中筛选出从零到阈值的样本
            # todo 由于是按照损失的排序，损失小的都在前面，即为正样本。
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            # todo 将忽略的样本的列表设置为空
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp

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

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[1] > nms_pre:
                max_scores, _ = (scores * iou_preds[..., None]).sqrt().max(-1)
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

    # todo 此处是为了计算cls特征图每个特征点所在的损失
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss_fea(self,
                 cls_scores,
                 bbox_preds,
                 iou_preds,
                 gt_bboxes,
                 gt_labels,
                 img_metas,
                 gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # todo 获取训练的目标 由于在设置中 pos的iou阈值设置为0.1，采用maxiou先挑选一大批的anchor，但是pos_anchor基本位于gt附近
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        # todo pos_inds 和 pos_gt_index是对应的
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets

        # todo 将每个level上面的的分数转换为图片上, 此处拿到的还是按照每个batch的分批次的结果，此处仍然是一个的list
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        # todo 获取每个bath的里面的点数
        test = cls_scores
        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        # todo 计算所有正样本的损失， 利用cls和reg分别计算得分，
        # todo 此处拿到的仍然是一个list，按照batch
        pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)
        # todo 重新分配样本
        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
            reassign_bbox_weights, num_pos = multi_apply(
                self.paa_reassign,
                pos_losses_list,
                labels,
                labels_weight,
                bboxes_weight,
                pos_inds,
                pos_gt_index,
                anchor_list)
            num_pos_lists = num_pos
            num_pos = sum(num_pos)

        # todo 由于重新分配的样本
        # todo reassign_labels, reassign_label_weight, reassign_bbox_weights
        # todo labels,          labesl_weight,       ,
        # todo 得到所需要的正样本的分类分数以及回归分数,从这里开始的所有的bath的样本的全部被拼接在一起！！！！！！！
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        # todo 通过重新分配之后, 得到新分配的labels，以及label的权重， 用此可以筛选出来所有正样本。
        labels = torch.cat(reassign_labels, 0).view(-1)
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))
        # todo 最后的一个类为背景, 将所有的类别的标签进行展开
        # todo 这个就是最终拿到的正样本的序号！！！！！！！
        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)
        # todo 为每一个样本的生成一个序号的，代表他是某一张图片的
        batch_index_all = []
        for j in range(len(test)):
            batch_index = torch.ones([test[j].size(0)], dtype=torch.int64) * (j + 1)
            batch_index_all.append(batch_index)
        batch_index_all = torch.cat(batch_index_all, 0).view(-1)
        # todo 获取所有的样本对应的真值序号
        batch_gt_all = []
        for i in range(len(test)):
            batch_gt_index = (torch.ones([test[j].size(0)], dtype=torch.int64) * 1000) .cuda()
            batch_gt_index[pos_inds[i]] = pos_gt_index[i]
            batch_gt_all.append(batch_gt_index)
        batch_gt_all = torch.cat(batch_gt_all, 0).view(-1)
        # todo 获得所有的正样本的标签，正样本的分类的结果, 对应的图片，对应的真值的索引
        pos_labels = labels[pos_inds_flatten]
        pos_cls = cls_scores[pos_inds_flatten]
        pos_index = batch_index_all[pos_inds_flatten]
        pos_gt = batch_gt_all[pos_inds_flatten]

        # todo 只有正样本时才得到回归的样本的值
        if num_pos:
            # todo 得到解码后的正样本
            pos_bbox_pred = self.bbox_coder.decode(
                flatten_anchors[pos_inds_flatten],
                bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            # todo 计算的所有的正样本与真值的之间的iou值
            iou_bbox = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
        else:
            # todo 如果一个样本都没有就虚构一个，不参与计算
            pos_bbox_pred = torch.zeros([1, 4], dtype=torch.int64)
            iou_bbox = torch.zeros([1], dtype=torch.int64)

        # todo 获取分离的 cls 和 reg 分数
        pos_cls_loss, pos_reg_loss = self.get_cr_loss(pos_cls, pos_bbox_pred, pos_labels,
                                                      pos_bbox_target, pos_inds_flatten, flatten_anchors)

        # todo SS架构的返回的结果
        # return pos_losses_list, pos_inds, pos_gt_index, pos_cls, pos_bbox_pred, pos_labels, iou_bbox, \
        #        pos_index, pos_gt

        # todo TS架构返回的结果
        # todo 1\ 真样本的分类结果， 回归结果， 每个样本的label,
        return pos_cls, pos_bbox_pred, pos_labels, iou_bbox, pos_index, pos_gt, pos_inds_flatten, pos_cls_loss, pos_reg_loss

    # todo 导出教师
    def loss_tea(self,
                 cls_scores,
                 bbox_preds,
                 iou_preds,
                 gt_bboxes,
                 gt_labels,
                 img_metas,
                 pos_inds_flatten,
                 gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # todo 获取训练的目标 由于在设置中 pos的iou阈值设置为0.1，采用maxiou先挑选一大批的anchor，但是pos_anchor基本位于gt附近
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        # todo pos_inds 和 pos_gt_index是对应的
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets

        # todo 将每个level上面的的分数转换为图片上, 此处拿到的还是按照每个batch的分批次的结果，此处仍然是一个的list
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]

        # todo 由于重新分配的样本
        # todo reassign_labels, reassign_label_weight, reassign_bbox_weights
        # todo labels,          labesl_weight,       ,
        # todo 得到所需要的正样本的分类分数以及回归分数,从这里开始的所有的bath的样本的全部被拼接在一起！！！！！！！
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))

        labels = torch.cat(labels, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target, 0).view(-1, bboxes_target[0].size(-1))

        cls_scores = cls_scores[pos_inds_flatten]
        bbox_preds = bbox_preds[pos_inds_flatten]
        labels = labels[pos_inds_flatten]
        bboxes_target = bboxes_target[pos_inds_flatten]
        flatten_anchors = torch.cat(
            [torch.cat(item, 0) for item in anchor_list])
        pos_cls_loss, pos_reg_loss = self.get_cr_loss(cls_scores, bbox_preds, labels,
                                                      bboxes_target, pos_inds_flatten, flatten_anchors)

        # todo 1\ 真样本的分类结果， 回归结果， 每个样本的label,
        return cls_scores, bbox_preds, pos_cls_loss, pos_reg_loss

    def get_cr_loss(self, cls_score, bbox_pred, label,
                     bbox_target, pos_inds, flatten_anchors):
        if not len(pos_inds):
            return cls_score.new([]),
        # todo 获取所有的正样本的分数
        pos_scores = cls_score
        pos_bbox_pred = bbox_pred
        pos_label = label
        pos_label_weight = torch.ones((len(cls_score), 1)).cuda()
        pos_bbox_target = bbox_target
        pos_bbox_weight = torch.ones((len(bbox_pred), 4)).cuda()

        # todo 获取所有的正样本的anchor
        pos_anchors = flatten_anchors[pos_inds]
        pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

        # to keep loss dimension
        # todo 计算所有的分类损失
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # todo 计算所有的回归的损失
        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return loss_cls, loss_bbox
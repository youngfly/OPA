from ..builder import DETECTORS, LOSSES
from .paa import PAA
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.models import build_detector
from mmdet.models import build_loss
import matplotlib.pyplot as plt
from mmcv.cnn import normal_init

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


@DETECTORS.register_module()
class SS(PAA):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                 test_cfg, pretrained, init_cfg)
        # todo 自监督
        self.if_gmm_bk = False
        self.gmm_bk_weight = 0.01
        # todo 交叉监督
        self.if_gmm_fc = False
        self.gmm_fc_weight = 0.001
        # todo 分类的正样本的监督
        self.if_pos_cls = False
        self.c_weight = 0.01
        # todo 回归的正样本的监督
        self.if_pos_reg = True
        self.r_weight = 1

        # todo gmm特征选取的规则
        # todo 每个层选取一定的特征数量
        # self.percent_topk = 0.01
        # todo 每层选取前100个特征
        # self.fea_topk = 100
        # self.fea_topks = [100, 100, 200, 200]
        # self.fea_topks_cls = [100, 100, 100, 100, 10]
        # self.covariance_type = 'diag'

        # todo 监督的通路选择
        # todo 0=1->2, 1=2->3, 2=3->4
        # self.ss_path = [2]

        # todo 滞后监督
        self.if_delay = True
        # todo dior训练集的图片数
        self.num_images_dior = 11725
        self.num_images_voc = 16552
        # todo 当前的iter以及epoch数
        self.iter = 0
        self.epoch = 0
        # todo 从第 n+1 个的epoch开始蒸馏
        self.ss_epoch = 3

        # todo 定义一个giou loss
        loss_dis_reg_cfg = dict(type='NIoULoss', loss_weight=2.0)
        self.loss_distill_reg = build_loss(loss_dis_reg_cfg)

        # todo 初始化参数
        self._init_adapt_weights()

    def _init_adapt_weights(self):
        normal_init(self.adapt_conv, std=0.01)

    def extract_feats_from_bk(self, img):
        """Directly extract features from the backbone."""
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      ):
        # todo 获取从backbone拿到的特征图
        bk_x = self.extract_feats_from_bk(img)
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

        # todo 全时域监督, 默认为Ture
        flag = True

        # todo 计算得到所有的特征点的损失
        # todo 应该是计算出每个层级上面的正样本的序号，计算出每个层级上面的每个地方正样本的特征值得分
        # todo 利用已经计算好的分数，利用反向回退回去计算在层级上面的位置。可能需要确定原始特征图的尺寸
        # todo 直接最后的扁平后的特征拉出来
        pos_lists, pos_inds, pos_gt_index, cls_feat, bbox_feat, pos_labels, iou_bbox, pos_index, pos_gt\
            = self.bbox_head.loss_fea(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # todo 采用延迟监督策略
        # todo 获取 batchsize 计算当前的iter轮数， 以及epoch数,
        if self.if_delay:
            batch_nums = len(gt_bboxes)
            self.iter += 1
            # todo num_images = ['dior', 'voc']
            self.epoch = int((self.iter * batch_nums) / self.num_images_voc) + 1
            flag = self.epoch > self.ss_epoch

        # todo 高斯混合密度估计，特征强化模块, 对应bk部分的各种通路选择蒸馏
        if self.if_gmm_bk and flag:
            # todo 高斯混合特征提取
            mask = self.feat_enhance(bk_x, gt_bboxes, gt_labels, self.fea_topks)
            # todo 全局特征监督
            # mask_a = self.feat_all(bk_x, gt_bboxes)
            # todo gt 区域特征
            # mask_gt = self.feat_gt(bk_x, gt_bboxes, img_metas, img)

            # todo 针对强化特征区域进行自监督
            sup_loss_self = self.ss_up(bk_x, mask) * self.gmm_bk_weight
            # todo 限制损失在一定的区域
            sup_loss_self = torch.clamp(sup_loss_self, 0, 10)
            # todo 更新损失函数
            losses.update(dict(loss_sup_self=sup_loss_self))

        # todo 高斯混合估计，交叉监督， 使用最后分类的特征去监督fpn的特征图。
        if self.if_gmm_fc and flag:
            # todo 高斯混合特征提取
            # mask_cross = self.feat_enhance(outs[0], gt_bboxes, gt_labels, self.fea_topks_cls)
            # todo 将对应一个图片所有得分组转换回每个层级上面
            # mask_score = self.tran_score(pos_lists, pos_inds, featmap_sizes)

            # todo 交叉监督--高效率版本
            sup_loss_cross = self.ss_cross_h(fea_x, cls_feat, pos_lists, pos_inds) * self.gmm_fc_weight
            # todo 交叉监督
            # sup_loss_cross = self.ss_cross(fea_x, outs, mask_cross) * self.gmm_fc_weight
            # todo 限制损失在一定的区域
            sup_loss_cross = torch.clamp(sup_loss_cross, 0, 10)
            # todo 更新损失函数
            losses.update(dict(loss_sup_cross=sup_loss_cross))

        # todo 分类的正样本的监督
        if self.if_pos_cls and flag:
            sup_loss_cls = self.ss_cls(cls_feat, pos_labels, pos_index, pos_gt) * self.c_weight
            sup_loss_cls = torch.clamp(sup_loss_cls, 0, 10)
            losses.update(dict(sup_loss_cls=sup_loss_cls))

        # todo 回归的正样本的监督
        if self.if_pos_reg and flag:
            sup_loss_reg = self.ss_reg(bbox_feat, pos_index, iou_bbox, pos_gt) * self.r_weight
            sup_loss_reg = torch.clamp(sup_loss_reg, 0, 10)
            losses.update(dict(sup_loss_reg=sup_loss_reg))

        # todo 绘图
        path_vis = "/home/airstudio/code/2.12/heatmap/"
        # self.heatmap_level(img_metas, outs[0], path_vis, name='cls')
        # self.heatmap_level(img_metas, fea_x, path_vis, name='fpn')
        # self.heatmap_level(img_metas, bk_x, path_vis, name='bk')
        # self.show_bboxes(gt_bboxes, img_metas, path_vis, thickness=3)
        # self.mask_level(img_metas, mask_score, path_vis)
        # self.gt_mask_level(img_metas, mask_gt, path_vis)
        # print('a batch is finish!')

        return losses

    # todo 特征强化监督模块
    # todo 1、2、利用混合估计挑选前景特征
    def feat_enhance(self, features, gt_bboxes, gt_label, _top):
        # todo features=[4,[n,c,h,w] 其中n为 batchsize
        # todo 特征图尺寸
        feat_size = [feature.size()[-2:] for feature in features]
        # todo batch以及level的参数
        batch_size = len(gt_bboxes)
        level = len(features)
        # todo 1、先计算特征图每个点的特征得分，得分高的特征点作为正样本,采用attention来计算每个点的优劣
        heat_score_all = []
        for j in range(level):
            # todo 两种计算attentiona得分的方式，求和以及平均值 [n,h,w]
            # heatmap_sum = torch.sum(features[j].detach(), dim=1)
            heatmap_ave = torch.mean(features[j].detach(), dim=1)
            # todo 得分归一化
            # sum_max = heatmap_sum.max()
            # sum_min = heatmap_sum.min()
            # ave_max = heatmap_ave.max()
            # ave_min = heatmap_ave.min()
            # todo 采用加和得分
            # heat_score.append(heatmap_sum)
            # todo 采用平均得分
            heat_score_all.append(heatmap_ave)

        # todo heat_score_all = [level,[n,h,w]]
        # todo 2、利用attention的得分来进行混合高斯建模估计，重新选择用来监督的区域，通过生成一个对应尺度的mask
        # todo 增加样本的挑选，每个特征图上选择topk个样本，防止参与拟合的样本数量太多了
        device = features[0].device
        # todo 最终返回的是一个指定区域的mask, 为后续的自监督指定. mask_all=[level,[n,h,w]]
        mask_all = []
        for j in range(level):
            mask_per_level = []
            for i in range(batch_size):
                # todo 准备初始化参数
                min_heat, max_heat = heat_score_all[j][i].min(), heat_score_all[j][i].max()
                means_init = np.array([min_heat, max_heat]).reshape(2, 1)
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
                # todo 构建高斯混合模型,将所有特征分为两组，一组为前景，一组为背景
                gmm = skm.GaussianMixture(
                    2,
                    weights_init=weights_init,
                    means_init=means_init,
                    precisions_init=precisions_init,
                    covariance_type=self.covariance_type,
                    reg_covar=1e-4)
                # todo 构建贝叶斯高斯混合模型，
                bgm = skm.BayesianGaussianMixture(
                    n_components=2,
                    init_params='kmeans',
                    covariance_type=self.covariance_type)
                # todo 为每一个层级的特征图挑选出前k个特征点,heat_score_all = [level,[n,h,w]]
                # todo 直接把tensor展成一个一维向量,从每一层特征的选取出最大的topk作为估计初始化参数
                heat_score_1 = heat_score_all[j][i].reshape(-1)
                # todo value的排序的是按照从大到小的顺序，但是inds的排序为对应的位置
                # todo 给出三个topk的选择方法，一个固定数量，一个百分比确立
                # todo 1) 百分比
                # nums = int(heat_score_1.size(0) * self.percent_topk)
                # value_fea, fea_topk_inds = heat_score_1.topk(nums, largest=True)
                # todo 2) 所有的层级采用相同的数量的样本
                # value_fea, fea_topk_inds = heat_score_1.topk(self.fea_topk, largest=True)
                # todo 3) 每个层级采用的不同数量的样本
                value_fea, fea_topk_inds = heat_score_1.topk(_top[j], largest=True)
                value_fea = value_fea.view(-1, 1).cpu().numpy()
                device = fea_topk_inds.device
                # todo 使用EM算法估算高斯混合模型参数
                gmm.fit(value_fea)
                # todo 使用估计好的模型来预测 数据样本的标签, 0-gt_nums个标签
                gmm_assignment = gmm.predict(value_fea)
                # todo 计算每个样本的加权对数的概率
                # scores = gmm.score_samples(value)
                # todo 将gmm_assignment和score全部转换为tensor
                gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
                # scores = torch.from_numpy(scores).to(device)
                # todo 将分配为前景的点筛选出来
                mask = fea_topk_inds * gmm_assignment
                mask_level = torch.zeros([heat_score_all[j][i].size(0), heat_score_all[j][i].size(1)]).reshape(-1)
                # todo 将筛选出来的设置为有值
                mask_level[mask] = 1
                mask_level[0] = 0
                # todo 将mask转化为特征图的尺寸
                mask_level = mask_level.reshape(heat_score_all[j][i].size(0), heat_score_all[j][i].size(1))

                mask_per_level.append(mask_level)
            # todo trans [h,w,n]
            mask_per_level = torch.stack(mask_per_level, 0)
            # todo trans [n,h,w]
            mask_per_level.permute(2, 0, 1)
            # todo trans [level,[n,h,w]]
            mask_all.append(mask_per_level)
        return mask_all

    # todo 全局特征的挑选
    def feat_all(self, features, gt_bboxes):
        mask_all = []
        batch_size = len(gt_bboxes)
        level = len(features)
        for j in range(level):
            mask_per_level = torch.ones((batch_size, features[j].size()[2], features[j].size()[3]))
            mask_all.append(mask_per_level)
        return mask_all

    # todo gt区域特征挑选
    def feat_gt(self, features, gt_bboxes, img_metas, img):
        # todo features = [4, [n, c, h, w]]
        level = len(features)
        batch_size = len(gt_bboxes)
        featmap_sizes = [featmap.size()[-2:] for featmap in features]
        # features = features[0].device
        # todo 获取一个批次图片的尺寸, 并计算其面积  img_ori=[1000,600] ，area_image = 不固定
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])
        area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()

        # todo gt_mask
        gt_mask_batch = []

        for i in range(batch_size):
            mask_per_im = []
            # todo gt_bboxes=[gt_num,4] 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作,
            # todo [0,1/16]->0, [1/16,1/8]->0, [1/8,1/4]->1,[1/4, 1/2]->2, [1/2, 1]->3
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制4个backbone层级里面
            target_lvls = target_lvls.clamp(min=0, max=3).long()
            # todo target_lvls=[gt_num], 值=[0~3]
            for j in range(level):
                # todo 每个层级的特征图的尺度
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                fea_scale = torch.tensor([featmap_sizes[j]])
                # todo 计算出每一层特征图相对于原始图片的缩放比率。
                # todo l1= 1/4, l2=1/8, l3=1/16, l4= 1/32
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # percent = torch.tensor(1/(4*(2 ** j))).cuda()
                # todo 构建按比例缩放后的gt坐标组
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                gt_trans[:, :2] = gt_bboxes[i][:, :2] * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:] * percent
                # todo gt_mask尺寸为和gt_bboxes尺寸相同，gt_mask=[gt_num,4]
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # todo 根据当前图片里每所有的gt，生成对应当前层级的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果第k个gt分配的level层级等于当前的level层级，则将该层级的gt的gt_mask置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # todo 将mask和gt_trans相乘，得到真正的gt
                gt_trans = gt_trans * gt_mask
                # todo 根据不同的层级得到对应的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # todo 为该层级生成特征图尺寸的mask
                    mask_per_gt = torch.zeros(height, width, dtype=torch.int64).cuda()
                    # todo 将mask里面gt位置的值置为1
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                    gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                    # todo 将这个level里面所有的gt位置叠加起来，得到一个层级特征mask
                    mask_per_level += mask_per_gt
                # todo 得到一张图片的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            gt_mask_batch.append(mask_per_im)
        return gt_mask_batch

    # todo 层级的转换函数
    def tran_score(self, pos_lists, pos_inds, featmap_sizes):
        # todo pos_lists = [n,[?]] ?:所有正样本的得分列表，
        # todo pos_inds= [n, [?]] ?:代表所有正样本序号的列表
        # todo featmap_sizes = [level,[h,w]]
        # todo 需要生成一个这样的 mask [level,[n,h,w]], 内容为pos_lists, 位置为pos_inds, 大小为featmaps_sizes
        mask_score = []
        # todo 每个层级的该有的特征数, 使用这个列表把样本分开到每个层级上
        fea_nums_each_level = [fea[0] * fea[1] for fea in featmap_sizes]
        # todo consum 实现了每个层级的特征数量的叠加
        fea_nums = np.cumsum(fea_nums_each_level)
        # todo 建立一个新的映射 club = [l,y,x,value]
        # todo  全部batch
        club_all =[]
        for i in range(len(pos_lists)):
            club = torch.zeros(len(pos_lists[i]), 4).cuda()
            # todo 将一个batch里面的正样本的score赋予进去 (value)
            club[:, 3] = pos_lists[i]
            # todo 根据pos_inds将正样本的序号的分配给各个层级 (l)
            level0 = pos_inds[i] < fea_nums[0]
            level1 = ((fea_nums[0] - 1) < pos_inds[i]) & (pos_inds[i] < (fea_nums[1]))
            level2 = ((fea_nums[1] - 1) < pos_inds[i]) & (pos_inds[i] < (fea_nums[2]))
            level3 = ((fea_nums[2] - 1) < pos_inds[i]) & (pos_inds[i] < (fea_nums[3]))
            level4 = ((fea_nums[3] - 1) < pos_inds[i]) & (pos_inds[i] < (fea_nums[4]))
            mask_one = torch.ones(len(pos_lists[i])).cuda()
            club[:, 0] = level0 * mask_one
            club[:, 0] = club[:, 0] + level1 * mask_one * 2
            club[:, 0] = club[:, 0] + level2 * mask_one * 3
            club[:, 0] = club[:, 0] + level3 * mask_one * 4
            club[:, 0] = club[:, 0] + level4 * mask_one * 5
            # todo 为每个样本修正在原有层级上面的坐标 (y,x),首先根据层级计算的位于哪个层级
            # todo 从第二个层级开始首先应该减去前面层级的特征数量
            for j in range(len(pos_inds[i])):
                if club[j, 0] == 1:
                    club[j, 1] = pos_inds[i][j] / featmap_sizes[0][1]
                    club[j, 2] = (pos_inds[i][j] % featmap_sizes[0][1])-1
                if club[j, 0] == 2:
                    ind_new = pos_inds[i][j] - fea_nums[0]
                    club[j, 1] = ind_new / featmap_sizes[1][1]
                    club[j, 2] = (ind_new % featmap_sizes[1][1])-1
                if club[j, 0] == 3:
                    ind_new = pos_inds[i][j] - fea_nums[1]
                    club[j, 1] = ind_new / featmap_sizes[2][1]
                    club[j, 2] = (ind_new % featmap_sizes[2][1])-1
                if club[j, 0] == 4:
                    ind_new = pos_inds[i][j] - fea_nums[2]
                    club[j, 1] = ind_new / featmap_sizes[3][1]
                    club[j, 2] = (ind_new % featmap_sizes[3][1])-1
                if club[j, 0] == 5:
                    ind_new = pos_inds[i][j] - fea_nums[3]
                    club[j, 1] = ind_new / featmap_sizes[4][1]
                    club[j, 2] = (ind_new % featmap_sizes[4][1])-1
            club_all.append(club)
        # todo club_all=[n, [nums, 4]] 4=[l, y, x, value]
        # todo 所有的level
        for j in range(len(featmap_sizes)):
            mask_per_level = []
            # todo 所有的batch featuremap = [h, w]
            for i in range(len(pos_lists)):
                mask_level_batch = torch.zeros([featmap_sizes[j][0], featmap_sizes[j][1]])
                # todo 所有的正样本的
                for n in range(len(club_all[i])):
                    # todo 根据娥
                    if club_all[i][n][0] == j+1:

                        yy = club_all[i][n, 1].detach().type(torch.long)
                        xx = club_all[i][n, 2].detach().type(torch.long)
                        vv = club_all[i][n, 3].detach().type(torch.long)
                        mask_level_batch[yy, xx] = vv
                mask_per_level.append(mask_level_batch)
            mask_per_level = torch.stack(mask_per_level, 0)
            mask_per_level.permute(2, 0, 1)
            mask_score.append(mask_per_level)
        # todo mask_score =  [level,[n,h,w]]
        return mask_score

    # todo 自监督模块
    # todo 3、构建自监督模块，根据之前选择的特征区域，利用特征图直接计算L2损失?或者采用其他的计算损失方式
    # todo 由于四个block出来的特征图的尺寸完全不同,将高尺度的特征图上采样，通道减少
    # todo Loss = F[|T(B1)-B2|*mask]+ ....  B = [l,[n,c,h,w]] features mask=[l[n,h,w]]
    def ss_down(self, features, mask):
        sup_loss_self = 1e-6
        norm = 0
        # todo 四个level
        for j, feature in enumerate(features):
            # todo 将后续level的特征图进行上采样，1——>2(up), 2-->3(up), 3-->4(up)
            # todo 将所有batch的特征图进行上采样, 从第二个level开始
            if j > 0:
                loss = 1e-6
                # todo 所有操作相当于把高级别的特征图去匹配低级别的特征图
                # todo 尺度上采样
                feature_up = F.interpolate(feature, size=features[j - 1].size()[-2:], scale_factor=None, mode='nearest')
                # todo 1、将所有的batch的特征图进行通道匹配, 逐通道计算差异性 [4,[n,c,h,w]] features channel机制
                # todo 将高尺度的通道数转换为低尺度的通道数，通道减少
                feature_expand = self.adapt_conv(feature_up, feature_up.size()[1], features[j - 1].size(1))
                # todo 获取每一个层级上面的mask
                mask_per_level = (mask[j - 1] > 0).float().cuda()
                # todo 获得归一化的参数,每个层级的上面的归一化参数
                norm = mask_per_level.sum() * 2
                # todo 计算得到高层级和低层级的特征图之间的loss_map
                loss_map = torch.pow(feature_expand - features[j - 1], 2)

                # todo 1)计算得到attention
                # loss_map = torch.sum(loss_map, dim=1)
                # for i in range(batch_size):
                #     sup_loss_self += (loss_map[i] * mask_per_level[i]).sum()

                # todo 2）逐通道与mask对应点进行乘, 首先将mask的维度增加一维， 并对损失进行归一化
                mask_per_level = mask_per_level.unsqueeze(1)
                loss = (loss_map * mask_per_level).sum()
                loss = loss/(norm + 1)

                # todo 3) 通路选择，只有选择的通路在添加自监督
                if j not in self.ss_path:
                    loss = 0

                # todo 4）损失计算
                sup_loss_self += loss
        return sup_loss_self

    def ss_up(self, features, mask):
        sup_loss_self = 1e-6
        norm = 0
        # todo 四个level
        for j, feature in enumerate(features):
            # todo 将所有batch的特征图进行下采样, 从第二个level开始
            if j < 3:
                loss = 1e-6
                # todo 所有操作相当于把高级别的特征图去匹配低级别的特征图
                # todo 尺度下采样
                feature_down = F.interpolate(feature, size=features[j + 1].size()[-2:], scale_factor=None, mode='nearest')
                # todo 1、将所有的batch的特征图进行通道匹配, 逐通道计算差异性 [4,[n,c,h,w]] features channel机制
                # todo 将低尺度的通道数转换为高尺度的通道数，通道增加
                feature_expand = self.adapt_conv(feature_down, feature_down.size()[1], features[j + 1].size(1))
                # todo 获取每一个层级上面的mask
                mask_per_level = (mask[j + 1] > 0).float().cuda()
                # todo 获得归一化的参数,每个层级的上面的归一化参数
                norm = mask_per_level.sum() * 2
                # todo 计算得到高层级和低层级的特征图之间的loss_map
                loss_map = torch.pow(feature_expand - features[j + 1], 2)

                # todo 1)计算得到attention
                # loss_map = torch.sum(loss_map, dim=1)
                # for i in range(batch_size):
                #     sup_loss_self += (loss_map[i] * mask_per_level[i]).sum()

                # todo 2）逐通道与mask对应点进行乘, 首先将mask的维度增加一维， 并对损失进行归一化
                mask_per_level = mask_per_level.unsqueeze(1)
                loss = (loss_map * mask_per_level).sum()
                loss = loss/(norm + 1)
                # todo 3) 通路选择，只有选择的通路在添加自监督
                if j not in self.ss_path:
                    loss = 0

                # todo 4）损失计算
                sup_loss_self += loss
        return sup_loss_self

    # todo 交叉监督， 利用cls去监督fpn， 尺度对应的监督模式
    def ss_cross(self, fea_fpn, fea_cls, mask):
        sup_loss_cross = 1e-6
        norm = 0
        # todo fea_x = [5, [n, c, h, w]]  outs=[3,[5,[n,c,h,w]]]
        fpn_chan = fea_fpn[0].size(1)
        cls_chan = fea_cls[0][0].size(1)
        # todo 遍历5个尺度
        for j in range(len(fea_fpn)):
            # todo 对cls的特征进行通道转化
            feature_expand = self.adapt_conv(fea_cls[0][j], cls_chan, fpn_chan)
            # todo 获取每一层的上面的mask
            mask_per_level = (mask[j] > 0).float().cuda()
            norm = mask_per_level.sum() * 2
            # todo 计算得到cls和fpn的特征图之间的loss_map
            loss_map = torch.pow(feature_expand - fea_fpn[j], 2)

            # todo 2）逐通道与mask对应点进行乘, 首先将mask的维度增加一维， 并对损失进行归一化
            mask_per_level = mask_per_level.unsqueeze(1)
            loss = (loss_map * mask_per_level).sum()
            loss = loss / (norm + 1)
            # todo 计算损失
            sup_loss_cross += loss
        return sup_loss_cross

    # todo 高效率版本, 不在将所有的特征转换到各个层级里面去，直接对所有的尺度特征进行处理
    def ss_cross_h(self, fea_fpn, cls_feat, pos_lists, pos_inds):
        # todo fea_fpn=[5,[n,c,h,w]]  cls_feat = [n,[nums, 20]]
        # todo pos_lists = [n,[nums_pos]],  pos_inds= [n, [nums_pos]]
        # todo 将fpn的特征也全部展开, 从不同的层级转换为每一正图片上
        sup_loss_cross = 0.0
        norm = 0
        cls_chan = cls_feat[0].size(1)
        fea_chan = fea_fpn[0].size(1)
        # todo 将fea = [n,[nums,128]]
        fea_flat = levels_to_images(fea_fpn)
        fea_flat = [
            item.reshape(-1, 128)for item in fea_flat]
        # todo 每一个batch
        for i in range(len(fea_flat)):
            fea_expand = self.adapt_fc(fea_flat[i], fea_chan, cls_chan)
            # todo 获取每一组的mask
            norm = pos_lists[i].float().cuda()
            norm = norm.sum() * 2
            # todo 计算得到map
            loss_map = torch.pow(fea_expand - cls_feat[i], 2)
            loss_map = torch.sum(loss_map, dim=1)
            # todo 计算得到每一个物体的所在的权重，由于pos_lists代表是损失的，与准确度是相反的
            score = 1/pos_lists[i]
            # print(score.shape)
            # print(pos_lists[i].shape)
            loss = (loss_map[pos_inds[i]] * score).sum()
            loss = loss/(norm + 1)
            sup_loss_cross += loss
        return sup_loss_cross

    # todo 双异相互监督*******************************************
    def ss_cls(self, cls_feat, pos_labels, pos_index, pos_gt):
        # todo cls_feat = [nums, 20]]，bbox_feat=[nums, 4]]， pos_labels=[nums, 1]
        # todo 定义分类互斥损失
        sup_loss_far = 0.0
        norm_far = 0
        # todo 定义分类同相损失
        sup_loss_near = 0.0
        norm_near = 0
        # todo 根据所有的正样本的类别进行处理
        for i in range(len(cls_feat)):
            # todo 当属于不同类比时候，计算互斥损失,
            # todo 生成一个权重列表，把不同类别的样本的筛选处理, 所有的异类别的都为true
            o_weights = (pos_labels != pos_labels[i]).unsqueeze(1)
            # todo 生成同类别的权重列表
            s_weights = (pos_labels == pos_labels[i]).unsqueeze(1)
            # todo 生成一个同batch，同样本的， 同类别的
            batch = (pos_index == pos_index[i]).cuda()
            gt = (pos_gt == pos_gt[i]).cuda()
            same = (pos_labels == pos_labels[i]).cuda()
            group = (batch * gt * same).unsqueeze(1)
            # todo 将当前样本的特征展成所有样本的等同数量的样本
            o_feats = cls_feat[i].repeat(len(cls_feat), 1)
            # todo 计算异类别之间的相远距离
            loss_far = (torch.pow(o_feats - cls_feat, 2) * o_weights).sum()
            norm_far = o_weights.sum()
            # todo 计算同类别之间的相近距离
            loss_near = (torch.pow(o_feats - cls_feat, 2) * group).sum()
            norm_near = group.sum()
            sup_loss_far += loss_far / norm_far
            sup_loss_near += loss_near /norm_near
        # todo 通过两个函数的嵌套，将距离最大转化为损失小, 互斥损失
        # sup_loss = torch.log2(1+torch.sigmoid(-sup_loss_far/4000))
        # todo 同向损失
        sup_loss = sup_loss_near
        return sup_loss

    def ss_reg(self, bbox_feat, pos_index, iou_bbox, pos_gt):
        # todo bbox_feat=[nums, 4]]， pos_labels=[nums, 1], pos_index
        sup_loss_b = 0.0
        B1 = []
        B2 = []
        # todo 1，利用pos_index，把每一个batch的结果先过滤出来的
        # todo 2，利用pos_inds把每一个batch分配相同的真值的样本过滤出来
        # todo 3，利用iou_bbox把得到的过滤的求出每一组中的最大值，作为监督的样本
        # todo 4,用这个样本去监督所有的其他样本
        for i in range(len(bbox_feat)):
            # todo 1 过滤batch
            batch = (pos_index == pos_index[i]).cuda()
            # todo 2 过滤每个gt
            gt = (pos_gt == pos_gt[i]).cuda()
            # todo 得到每一个组的内容
            group = batch * gt
            iou_max, index_max = torch.max(group * iou_bbox, dim=0)
            # todo 得到每一个组的数量, 归一化
            group_nums = (group == True).sum()
            # todo 得到当前的样本以及最大值的样本获取出来，计算的互监督的损失
            b1 = bbox_feat[i].unsqueeze(0)
            b2 = bbox_feat[index_max].unsqueeze(0)
            B1.append(b1)
            B2.append(b2)
        B1 = torch.cat(B1, 0)
        B2 = torch.cat(B2, 0)
        loss_reg = self.loss_distill_reg(B1, B2, avg_factor=1.0)
        loss_reg_norm = loss_reg / len(bbox_feat)
        sup_loss_b += loss_reg_norm
        return sup_loss_b

    # todo 适应层的定义**************************************************************************
    # todo 通道转换函数,适用于有长宽的特征图进行通道转换
    def adapt_conv(self, feature, in_channel, out_channel):
        # todo 定义通道函数
        a_conv = nn.Conv2d(in_channel, out_channel, 1, padding=0).cuda()
        normal_init(a_conv, std=0.01)
        adapt_new = a_conv(feature)
        return adapt_new

    # todo 通道转换函数，单维度的特征进行通道转换
    def adapt_fc(self, feature, in_channel, out_channel):
        # todo 定义通道函数
        a_fc = nn.Linear(in_channel, out_channel, bias=True).cuda()
        adapt_new = a_fc(feature)
        return adapt_new

    # todo 绘图区*******************************************************************************
    # todo 热力图绘制
    def heatmap_level(self, img_metas, heatmaps, path, name='heat'):
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
                # todo 热力图按照通道相加，
                heat = torch.sum(heatmaps[j][i], dim=0).detach().cpu().numpy()
                heat = torch.mean(heatmaps[j][i], dim=0).cpu().clone().detach().numpy()
                plt.imshow(heat, interpolation='bilinear')
                plt.colorbar()
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/" + name + "_level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def mask_level(self, img_metas, masks, path):
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
            for j in range(len(masks)):
                mask = masks[j][i].cpu().numpy() > 0
                plt.imshow(mask, cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/mask_level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    def gt_mask_level(self, img_metas, masks, path):
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
            for j in range(len(masks)):
                mask = masks[j][i].cpu().numpy() > 0
                plt.imshow(mask, cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())
                path_level = path_pic + "/gt_mask_level" + str(j + 1) + ".png"
                plt.savefig(path_level)
                plt.close("all")

    # todo 绘制带有bboxes的图片
    def show_bboxes(self,
                    bboxes,
                    img_metas,
                    path,
                    thickness=10):
        self.mkdir(path)
        # todo 选择你喜欢的颜色
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        cyan = (255, 255, 0)
        yellow = (0, 255, 255)
        magenta = (255, 0, 255)
        white = (255, 255, 255)
        black = (0, 0, 0)
        chartreuse = (127, 255, 0)
        colors = chartreuse
        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            path_pic = path + filename.split("/")[-1].split(".")[0]
            det_path = path_pic + "/det.png"
            flip = img_meta['flip']
            img_shape = img_meta['img_shape']
            img = cv2.imread(filename)
            # todo 针对图片尺寸的被放缩后的状态进行修正
            img = cv2.resize(img, (img_shape[1], img_shape[0]))
            if flip:
                img = cv2.flip(img, 1)
            box = bboxes[i]
            for j in range(len(box)):
                left_top = (box[j, 0], box[j, 1])
                right_bottom = (box[j, 2], box[j, 3])
                cv2.rectangle(img, left_top, right_bottom, colors, thickness=thickness)
            cv2.imwrite(det_path, img)
            print("it is ok")

    # todo 生成文件夹
    def mkdir(self, path):
        isexists = os.path.exists(path)
        if not isexists:
            os.makedirs(path)
            print("file makes successfully")
        else:
            print("file is exist")

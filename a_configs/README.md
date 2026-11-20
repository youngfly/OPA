# a_configs — OPA 论文实验配置系统

本目录是 OPA 论文（第三篇，标签分配方向）及其相关实验的全部训练/测试配置，基于 **mmdetection 2.25.1 个人 fork**（自定义模块直接注册进 `mmdet/`）

---

## 目录结构

```
a_configs/
├── paper3_OPA/                  103 个 —— 论文 3 主实验 + 补充实验
│   ├── coco/
│   │   ├── GA/                    3  GA 主方法（r50/r101 + 低显存变体）
│   │   ├── label_assign/         28  标签分配基线对比（ATSS/PAA/AutoAssign/FCOS/
│   │   │                           FOVEA/FSAF/GFL/RetinaNet/DDOD/TOOD/DW 等）
│   │   ├── dw/                    7  DWHead 补充实验
│   │   ├── swin/                  5  Swin backbone 补充实验
│   │   ├── vis_assign/            7  分配过程可视化（4 实验 + 3 底座）
│   │   ├── regnet/                2  RegNet backbone
│   │   ├── detr/                  2  DETR
│   │   ├── deformable_detr/       1  Deformable DETR
│   │   ├── sparse_rcnn/           2  Sparse R-CNN
│   │   └── extra_exp/             8  Faster R-CNN ×4 / Mask R-CNN ×4（fcos/ 为空目录）
│   └── voc/
│       ├── GA/                    4  GA 主方法（r50/r101 × 普通/低显存）
│       ├── labelassign/          19  标签分配基线对比
│       ├── RegNet/               11  GA_regnet ×2 / reg_800 ×2 / regnet_3.2G ×7
│       ├── branch_test/           2  IoU/GIoU 分支消融（1 实验 + 1 底座）
│       └── transformer/           2  Transformer 检测器
├── paper_sd/                     79 个 —— 自蒸馏/师生蒸馏（作者已标废弃）
│   ├── dior/                     41  baseline_resnet ×15 / baseline_regnet ×12 /
│   │                               compose ×5 / T_S_D_Res ×4 / mini ×4 / self_statistics ×1
│   ├── nwpu/                     23  baseline_resnet ×17 / T_S_D_Res ×4 / baseline_regnet ×2
│   ├── dota/                     11  baseline_resnet ×8 / baseline_regnet ×3
│   └── voc/                       4  ss_regnet ×3 / ss_resnet ×1
└── paper_ssl/                    12 个 —— MAE 自监督预训练 backbone 实验
    ├── dior/                      8  vit_base ×2 / vit_mini ×2 / vit_tiny ×2 / swin ×2
    │                               （vit_large/ 为空目录）
    └── nwpu/                      4  vit_tiny ×2 / Resnet ×2（VIT_ATSS 套 ResNet-18 对照）
```

---

## 三条实验线

### 1. paper3_OPA —— GA/OPA 标签分配（论文 3 主线）

核心方法 **GA**：检测器 `GA`（继承 PAA，`mmdet/models/detectors/ga.py`）+ 头 `GAHead`（继承 ATSSHead，`mmdet/models/dense_heads/ga_head.py`，OPA 核心实现：GMM 分配 + paa_reassign + IoU 分支 + score voting）。

GA 配置的统一关键参数：

| 参数                       | 值                   | 说明                                                    |
| -------------------------- | -------------------- | ------------------------------------------------------- |
| `reg_decoded_bbox`       | True                 | 解码后框上算 GIoU                                       |
| `loss_bbox`              | GIoULoss, weight=1.3 |                                                         |
| `score_voting`           | True                 | 推理端分数投票（注意：2.25 下失效，见根 README §9 #4） |
| `topk`                   | 9                    | **死参数**：代码硬编码 9，config 传值被忽略       |
| `test_cfg.min_bbox_size` | 0                    | 依赖 rpn 系 `>=0` 过滤行为（官方 2.25 已含）          |

### 2. paper_sd —— 自蒸馏/师生蒸馏（存档）

- **SS**（`detectors/ss.py`）自蒸馏：延迟监督（第 3 epoch 后启动）+ 特征通路蒸馏，配套 SSHead / SS_BAYHead（贝叶斯变体）。
- **Distill_PAA**（`detectors/dis_paa.py`）师生蒸馏：config 传 `tea_model` + `tea_pretrained`，教师冻结，adapt_conv 对齐通道，dis_cate/dis_geo 蒸馏。配置名 `D_` 前缀。
- 数据集：DIOR / NWPU / DOTA / VOC，消费自定义数据集类 DiorDataset / NwpuDataset / DotaDataset。
- 蒸馏开关与损失权重**硬编码在检测器代码里**。

### 3. paper_ssl —— MAE 自监督 backbone（ViT/Swin）

- backbone：`MaskedAutoencoderViT`（MAE 预训练 base/mini，`backbones/vit.py`）与 timm 系 `VisionTransformer`（tiny）；检测器 `VIT_ATSS`（neck 可选，ViT 可直连 head）。
- 预训练权重需要重新训练得到。

---

## 命名约定与 `_base_` 体系

- 普通：`{方法}{深度}_{数据集}.py`，如 `atss101_coco.py`、`paa_r50_nwpu.py`
- RegNet：`{方法}_{算力}G_f{宽度}_{数据集}.py`，如 `paa_3.2G_f0.5_voc.py`
- 蒸馏：`D_` 前缀（师生）；`ss` 前缀（自蒸馏）；`paab` = PAA + Bayesian
- 每个实验目录自带局部底座 `1_dataset_schedule_runtime.py` 或 `2_dataset_schedule_runtime.py`（共 24 个），封装 data_root / pipeline / schedule / runtime；部分目录另有 `coco_detection.py`、`schedule_2x.py` 等官方风格底座

## 自定义代码 ↔ 配置接线表

| 自定义类（mmdet 内）                    | 驱动配置                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| GA + GAHead                             | paper3_OPA：coco/GA、voc/GA、vis_assign/ga_r101、voc/RegNet/GA_regnet（共 10） |
| DWHead                                  | coco/dw ×7 + label_assign/dw ×1                                              |
| ATSSHead_iou_giou                       | voc/branch_test/atss_r101_voc.py                                               |
| ATSSHead_ASSIGN / PAAHead_ASSIGN        | coco/vis_assign：atss50 / paa50 / paa101                                       |
| SS + SSHead / SS_BAYHead                | paper_sd：voc/ss_*、dior compose、self_statistics                              |
| Distill_PAA                             | paper_sd：dior/nwpu 的 T_S_D_Res/D_paa_* ×2                                   |
| VIT_ATSS + MAE ViT                      | paper_ssl 全部（vit_base 除外，见根 README §9 #1）                            |
| DiorDataset / NwpuDataset / DotaDataset | paper_sd + paper_ssl 的数据底座                                                |
| NIoULoss                                | 不在 config 里：ss.py / dis_paa.py 内部硬编码构建（weight=2.0）                |

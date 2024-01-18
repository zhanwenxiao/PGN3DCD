from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_geometric.data import Data
from torch_geometric.nn import knn

from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch


log = logging.getLogger(__name__)


class SiameseKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self.threshold_dis_init = 0.5
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if self._weight_classes is not None:
            if len(self._weight_classes) != self._num_classes:
                self._weight_classes = None
        try:
            self._ignore_label = dataset.ignore_label
        except:
            self._ignore_label = None
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

        # Build Difference Module
        # self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
        #                                            d_pos=self.mask_dim,
        #                                            gauss_scale=self.gauss_scale,
        #                                            normalize=self.normalize_pos_enc)
        self.batch_size = 10
        self.self_cross_att_list = []
        self.self_self_att_list = []
        self.diff_conv_list = Sequential()
        self.pos_xyz_list = Sequential()
        init_dim = 128
        self.k_n = [1, 1, 1, 1, 1]
        for i in range(len(self.down_modules)):
            cross_att = Cross_Attention(int(init_dim * pow(2, i)))
            self_att = Self_Attention(int(init_dim * pow(2, i)))
            # self_att = Attention(int(up_f_dim / pow(2, i)))
            self.self_cross_att_list.append(cross_att)
            self.self_self_att_list.append(self_att)
            self.diff_conv_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(init_dim * pow(2, i) * 2, init_dim * pow(2, i), bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i)),
                        nn.LeakyReLU(0.2),
                    ]
                ),
            )
            self.pos_xyz_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(3, init_dim * pow(2, i), bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i)),
                        nn.LeakyReLU(0.2),
                    ]
                ),
            )

        # Build final MLP
        self.last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                self.last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=self.last_mlp_opt.dropout,
                bn_momentum=self.last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
        self.loss_names = ["loss_cd"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])
        self.last_feature = None
        self.visual_names = ["data_visual"]
        print('total : ' + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print('upconv : ' + str(sum(p.numel() for p in self.up_modules.parameters() if p.requires_grad)))
        print('downconv : ' + str(sum(p.numel() for p in self.down_modules.parameters() if p.requires_grad)))

    def set_class_weight(self,dataset):
        self._weight_classes = dataset.weight_classes
        # No ponderation if weights for the corresponding number of class are available
        if len(self._weight_classes) != self._num_classes:
            print('number of weights different of the number of classes')
            self._weight_classes = None

    def set_input(self, data, device):
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
        else:
            self.pre_computed = None
            self.upsample = None
        if getattr(data, "pos_target", None) is not None:
            data.x_target = add_ones(data.pos_target, data.x_target, True)
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                self.upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                self.pre_computed_target = None
                self.upsample_target = None

            self.input0, self.input1 = data.to_data()
            self.batch_idx_target = data.batch_target
            self.labels = data.y.to(device)
        else:
            self.input = data
            self.labels = None

    def dim_correct(self, knn_idx, layer_index):
        try:
            knn_idx = knn_idx.reshape(knn_idx[0], -1, self.k_n[layer_index])
        except:
            print("wrong dim")
        return knn_idx

    # mIoU: 81.76%
    # def forward(self, compute_loss = True, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #     nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     nn_list1 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch) # 找除了本身点之外，另一个本点云中的最近点
    #     nn_list1 = nn_list1.reshape(nn_list1.shape[0], -1, 2)
    #     # nn_list = self.dim_correct(nn_list, 0)
    #     mask = self.get_mask(data0.pos, data1.pos, nn_list, nn_list1)
    #     mask_l.append(mask)
    #     data1.mask = mask
    #
    #     for i in range(len(self.down_modules) - 1):
    #         data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
    #         mask_l.append(data1.mask)
    #         nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[i], data0.batch, data1.batch)
    #         # nn_list_1of2 = self.dim_correct(nn_list_1of2, i+1)
    #         # nn_list_2of1 = knn(data1.pos, data0.pos, self.k_n, data1.batch, data0.batch)
    #         diff = data1.clone()
    #         # diff.x = self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list_2of1, nn_list_1of2, i)
    #         diff.x = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, i, data1.x.device)
    #         stack_down.append(diff)
    #         # nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         # diff = data1.clone()
    #         # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         # stack_down.append(diff)
    #     #1024 : last layer
    #     data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
    #     nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[-1], data0.batch, data1.batch)
    #     data = data1.clone()
    #     data.x = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, len(self.down_modules) - 1, data1.x.device) # self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list_1of2, len(self.down_modules) - 1)
    #     # nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     # data = data1.clone()
    #     # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append(data)
    #         data = self.inner_modules[0](data)
    #         innermost = True
    #     for i in range(len(self.up_modules)):
    #         if i == 0 and innermost:
    #             data = self.up_modules[i]((data, stack_down.pop()))
    #             # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])last llasla
    #         else:
    #             data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
    #             # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])
    #     self.last_feature = data.x
    #     self.last_feature = self.self_att11(self.last_feature, None, False, mask_l[0])
    #     if self._use_category:
    #         self.output = self.FC_layer(self.last_feature, self.category)
    #     else:
    #         self.output = self.FC_layer(self.last_feature)
    #
    #     if self.labels is not None and compute_loss:
    #         self.compute_loss()
    #
    #     self.data_visual = self.input1
    #     self.data_visual.pred = torch.max(self.output, -1)[1]
    #
    #     return self.output

    # def forward(self, compute_loss = True, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     # 0
    #     nn_list = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     nn_list1 = knn(data0.pos, data0.pos, 2, data0.batch, data0.batch) # 找除了本身点之外，另一个本点云中的最近点
    #     nn_list1 = nn_list1.reshape(nn_list1.shape[0], -1, 2)
    #     # nn_list = self.dim_correct(nn_list, 0)
    #     mask1 = self.get_mask(data1.pos, data0.pos, nn_list, nn_list1, data0.batch)
    #     data0.x = mask1
    #
    #     # 1
    #     nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     nn_list1 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch) # 找除了本身点之外，另一个本点云中的最近点
    #     nn_list1 = nn_list1.reshape(nn_list1.shape[0], -1, 2)
    #     # nn_list = self.dim_correct(nn_list, 0)
    #     mask2 = self.get_mask(data0.pos, data1.pos, nn_list, nn_list1, data1.batch)
    #     data1.x = mask2
    #
    #     # mask_l.append(mask)
    #     # data1.mask = mask
    #
    #     for i in range(len(self.down_modules) - 1):
    #         data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
    #         # mask_l.append(data1.mask)
    #         nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[i], data0.batch, data1.batch)
    #         # nn_list_1of2 = self.dim_correct(nn_list_1of2, i+1)
    #         # nn_list_2of1 = knn(data1.pos, data0.pos, self.k_n, data1.batch, data0.batch)
    #         diff = data1.clone()
    #         # diff.x = self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list_2of1, nn_list_1of2, i)
    #         diff.x = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, i, data1.x.device)
    #         stack_down.append(diff)
    #         # nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         # diff = data1.clone()
    #         # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         # stack_down.append(diff)
    #     #1024 : last layer
    #     data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
    #     nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[-1], data0.batch, data1.batch)
    #     data = data1.clone()
    #     data.x = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, len(self.down_modules) - 1, data1.x.device) # self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list_1of2, len(self.down_modules) - 1)
    #     # nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     # data = data1.clone()
    #     # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append(data)
    #         data = self.inner_modules[0](data)
    #         innermost = True
    #     for i in range(len(self.up_modules)):
    #         if i == 0 and innermost:
    #             data = self.up_modules[i]((data, stack_down.pop()))
    #             # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])
    #         else:
    #             data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
    #             # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])
    #     self.last_feature = data.x
    #     # self.last_feature = self.self_att11(self.last_feature, None, False, mask_l[0])
    #     if self._use_category:
    #         self.output = self.FC_layer(self.last_feature, self.category)
    #     else:
    #         self.output = self.FC_layer(self.last_feature)
    #
    #     if self.labels is not None and compute_loss:
    #         self.compute_loss()
    #
    #     self.data_visual = self.input1
    #     self.data_visual.pred = torch.max(self.output, -1)[1]
    #
    #     return self.output

    def forward(self, compute_loss = True, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []
        mask_l = []

        data0 = self.input0
        data1 = self.input1

        # 1
        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        nn_list1 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch) # 找除了本身点之外，另一个本点云中的最近点
        nn_list1 = nn_list1.reshape(nn_list1.shape[0], -1, 2)
        # nn_list = self.dim_correct(nn_list, 0)
        mask = self.get_mask(data0.pos, data1.pos, nn_list, nn_list1, data1.batch)
        data1.mask = mask

        for i in range(len(self.down_modules) - 1):
            data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            mask_l.append(data1.mask)
            nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[i], data0.batch, data1.batch)
            diff = data1.clone()
            diff_f = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, i, data1.x.device)
            # enc_q_pos = self.pos_enc(data0.pos)
            # enc_k_pos = self.pos_enc(data1.pos)
            # diff.x = self.self_att_list[i](f0, f1, diff_f, None, None, None)
            diff.x = self.diff_former_module(data1.x, data0.x, diff_f, i, nn_list_1of2, self.k_n[i])
            stack_down.append(diff)
        #1024 : last layer
        data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
        nn_list_1of2 = knn(data0.pos, data1.pos, self.k_n[-1], data0.batch, data1.batch)
        data = data1.clone()
        diff_f = self.nearest_feature_difference(data1.x, data0.x, nn_list_1of2, len(self.down_modules) - 1, data1.x.device)

        # enc_q_pos = self.pos_enc(data0.pos)
        # enc_k_pos = self.pos_enc(data1.pos)
        # data.x = self.self_att_list[-1](f0, f1, diff_f, None, None, None)
        data.x = self.diff_former_module(data1.x, data0.x, diff_f, -1, nn_list_1of2, self.k_n[-1])

        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
                # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
                # data.x = self.self_att_list[i](data.x, None, False, mask_l[-1 - i])
        self.last_feature = data.x
        # self.last_feature = self.self_att11(self.last_feature, None, False, mask_l[0])
        if self._use_category:
            self.output = self.FC_layer(self.last_feature, self.category)
        else:
            self.output = self.FC_layer(self.last_feature)

        if self.labels is not None and compute_loss:
            self.compute_loss()

        self.data_visual = self.input1
        self.data_visual.pred = torch.max(self.output, -1)[1]

        return self.output

    def compute_loss(self):
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)
        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            print('lambda_internal_losses')
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # Final cross entrop loss
        if self._ignore_label is not None:
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes, ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.output, self.labels, weight=self._weight_classes)

        if torch.isnan(self.loss_seg).sum() == 1:
            print(self.loss_seg)
        self.loss += self.loss_seg

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    def reset_final_layer(self, cuda = True):
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                self.last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=self.last_mlp_opt.dropout,
                bn_momentum=self.last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = self.last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(self.last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, self.last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(self.last_mlp_opt.nn[i], momentum=self.last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = self.last_mlp_opt.nn[i]

            if self.last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=self.last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
            if cuda:
                self.FC_layer.cuda()

    # def nearest_feature_difference(self, f_1, f_2, knearest_idx_1, layer_id, device):
    #     # 2、Feature Difference Module
    #     knearest_idx_1 = knearest_idx_1.reshape(knearest_idx_1.shape[0], -1, self.k_n[layer_id])
    #     # 2.0、Get Nearest Feature
    #     f_2 = f_2[knearest_idx_1[1, :, :], :]
    #     # 2.1、Sub
    #     subabs = torch.mean(f_1.unsqueeze(1) - f_2, 1)
    #     # 2.2、Angle
    #     sub = torch.mean(f_1.unsqueeze(1) - f_2, 1) # f_1 - f_2
    #     f_dis = torch.sqrt(torch.sum(torch.square(sub), dim=-1)).unsqueeze(-1)
    #     angle = torch.divide(sub, f_dis)
    #     angle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
    #     # # 2.3、CVA
    #     # sub = f_1.unsqueeze(1) - f_2  # f_1 - f_2
    #     # cva = torch.mean(torch.sqrt(torch.sum(torch.square(sub), dim=-1)), 1).unsqueeze(1)
    #     # 2.4、Feature Concate
    #     dif_f = self.diff_conv_list[layer_id].to(device)(torch.concat((subabs, angle), dim=1))
    #
    #     return dif_f

    # Single Target
    # def feature_difference_module(self, f_1, f_2, f1_xyz, f2_xyz, knearest_idx_2, layer_index):
    #     shape_0 = knearest_idx_2.shape[0]
    #     k_n_mult_ptnum = knearest_idx_2.shape[1]
    #     device = f1_xyz.device
    #     self.threshold_dis = self.threshold_dis_init * np.power(4, layer_index)
    #     try:
    #         knearest_idx_2 = knearest_idx_2.reshape(shape_0, f_2.shape[0], self.k_n[layer_index])
    #     except:
    #         print("wrong dim")
    #
    #     enc_xyz_1 = self.pos_xyz_list[layer_index].to(device)(f1_xyz)
    #     enc_xyz_2 = self.pos_xyz_list[layer_index].to(device)(f2_xyz)
    #
    #     f_1 = self.self_att_list[layer_index][0](f_1, enc_xyz_1, True, None)
    #     f_2 = self.self_att_list[layer_index][0](f_2, enc_xyz_2, True, None)
    #
    #     # 3、Feature Difference Module
    #     dif_f_2 = self.nearest_feature_difference(f_2, f_1, knearest_idx_2, layer_index, device)
    #
    #     # 3、Get Transformer Mask
    #     mask_2 = self.get_mask(f1_xyz, f2_xyz, knearest_idx_2)
    #
    #     # 4、Self Attention
    #     fout2 = self.self_att_list[layer_index][1](dif_f_2, None, False, mask_2)
    #
    #     return fout2

    def get_mask(self, f_1, f_2, knearest_idx_1, knearest_idx_2, batch_idx):

        # nearest_2of1 = f_1[knearest_idx_1[1, :], :]
        # f_2_ex = f_2#f_2.unsqueeze(1).repeat(1, nearest_2of1.shape[1], 1)
        # dis_mat_1 = torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of1), dim=-1)).view(-1, 1)
        #
        # batch_size = 10
        # for bs in range(batch_size):
        #     index_bs = torch.argwhere(batch_idx == bs)
        #     dis_bs = dis_mat_1[index_bs]
        #     dis_mat_1[index_bs] = ((dis_bs - torch.min(dis_bs, dim=0)) / (torch.max(dis_bs) - torch.min(dis_bs))) + 1
        #
        # return dis_mat_1

        # 2在1中的邻近点
        nearest_2of1 = f_1[knearest_idx_1[1, :], :]
        f_2_ex = f_2 # f_2.unsqueeze(1).repeat(1, nearest_2of1.shape[1], 1)
        dis_mat_1 = torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of1), dim=-1)).view(-1, 1)

        nearest_2of2 = f_2[knearest_idx_2[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, nearest_2of2.shape[1], 1)
        dis_mat_2 = torch.max(torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of2), dim=-1)), dim=-1, keepdim=True)[0]

        mask_2 = torch.where((dis_mat_1 > 2 * dis_mat_2) | (2 * dis_mat_1 > dis_mat_2), 1.0, 0.0) #False, True)

        return mask_2

    def nearest_feature_difference(self, f_1, f_2, knearest_idx_1, layer_id, device):
        # # 2、Feature Difference Module
        # # 2.0、Get Nearest Feature
        # f_2 = f_2[knearest_idx_1[1, :], :]
        # # 2.1、Sub
        # subabs = f_1 - f_2
        # # 2.2、Angle
        # sub = f_1 - f_2 # f_1 - f_2
        # f_dis = torch.sqrt(torch.sum(torch.square(sub), dim=-1)).unsqueeze(-1)
        # angle = torch.divide(sub, f_dis)
        # angle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
        # # 2.3、CVA
        # # sub = f_1 - f_2  # f_1 - f_2
        # # cva = torch.mean(torch.sqrt(torch.sum(torch.square(sub), dim=-1)), 1).unsqueeze(1)
        # # 2.4、Feature Concate
        # dif_f = self.diff_conv_list[layer_id].to(device)(torch.concat((subabs, angle), dim=1))
        #
        # return dif_f

        # 2、Feature Difference Module
        knearest_idx_1 = knearest_idx_1.reshape(knearest_idx_1.shape[0], -1, self.k_n[layer_id])
        # 2.0、Get Nearest Feature
        f_1 = f_1.unsqueeze(1).repeat(1, self.k_n[layer_id], 1)
        f_2 = f_2[knearest_idx_1[1, :, :], :]
        # 2.1、Sub
        subabs = f_1 - f_2
        # 2.2、Angle
        sub = f_1 - f_2 # f_1 - f_2
        f_dis = torch.sqrt(torch.sum(torch.square(sub), dim=-1)).unsqueeze(-1)
        angle = torch.divide(sub, f_dis)
        angle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
        # 2.3、CVA
        # sub = f_1 - f_2  # f_1 - f_2
        # cva = torch.mean(torch.sqrt(torch.sum(torch.square(sub), dim=-1)), 1).unsqueeze(1)
        # 2.4、Feature Concate
        dif_f = self.diff_conv_list[layer_id].to(device)(torch.concat((subabs, angle), dim=-1))

        return dif_f

    def diff_former_module(self, f0, f1, diff_f, idx, nn_list_1of2, k_num):
        output = self.self_cross_att_list[idx](f0, f1, diff_f, None, None, None, nn_list_1of2, k_num)
        output = self.self_self_att_list[idx](output, None)
        return output


    # Double Target
    # def feature_difference_module(self, f_1, f_2, f1_xyz, f2_xyz, knearest_idx_1, knearest_idx_2, layer_index):
    #     device = f1_xyz.device
    #     self.threshold_dis = self.threshold_dis_init * np.power(4, layer_index)
    #
    #     enc_xyz_1 = self.pos_xyz_list[layer_index].to(device)(f1_xyz)
    #     enc_xyz_2 = self.pos_xyz_list[layer_index].to(device)(f2_xyz)
    #
    #     f_1 = self.self_att_list[layer_index][0](f_1, enc_xyz_1, True, None)
    #     f_2 = self.self_att_list[layer_index][0](f_2, enc_xyz_2, True, None)
    #
    #     # 3、Feature Difference Module
    #     dif_f_1 = self.nearest_feature_difference(f_1, f_2, knearest_idx_1, layer_index, device)
    #     dif_f_2 = self.nearest_feature_difference(f_2, f_1, knearest_idx_2, layer_index, device)
    #
    #     # 3、Get Transformer Mask
    #     mask_1, mask_2 = self.get_mask(f1_xyz, f2_xyz, knearest_idx_1, knearest_idx_2)
    #
    #     # 4、Self Attention
    #     fout1 = self.self_att_list[layer_index][1](dif_f_1, None, False, mask_1)
    #     fout2 = self.self_att_list[layer_index][1](dif_f_2, None, False, mask_2)
    #
    #     return fout1, fout2
    #
    # def get_mask(self, f_1, f_2, knearest_idx_1, knearest_idx_2):
    #     # 1在2中的邻近点
    #     nearest_1of2 = f_2[knearest_idx_1[1, :], :]
    #     nearest_1of2 = nearest_1of2.reshape(-1, self.k_n, nearest_1of2.shape[-1])
    #     f_1_ex = f_1.unsqueeze(1).repeat(1, nearest_1of2.shape[1], 1)
    #     dis_mat_1 = torch.mean(torch.sqrt(torch.sum(torch.square(f_1_ex - nearest_1of2), dim=-1)), dim=-1).view(-1, 1)
    #     mask_1 = torch.where(dis_mat_1 > self.threshold_dis, False, True)
    #
    #     # 2在1中的邻近点
    #     nearest_2of1 = f_1[knearest_idx_2[1, :], :]
    #     nearest_2of1 = nearest_2of1.reshape(-1, self.k_n, nearest_2of1.shape[-1])
    #     f_2_ex = f_2.unsqueeze(1).repeat(1, nearest_2of1.shape[1], 1)
    #     dis_mat_2 = torch.mean(torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of1), dim=-1)), dim=-1).view(-1, 1)
    #     mask_2 = torch.where(dis_mat_2 > self.threshold_dis, False, True)
    #
    #     return mask_1, mask_2
    #
    # def nearest_feature_difference(self, f_1, f_2, knearest_idx_1, layer_id, device):
    #     # 2、Feature Difference Module
    #     # 2.0、Get Nearest Feature
    #     f_2 = f_2[knearest_idx_1[1, :], :]
    #     f_2 = f_2.reshape(-1, self.k_n, f_2.shape[-1])
    #     # 2.1、Sub
    #     subabs = torch.mean(torch.abs(f_1.unsqueeze(1) - f_2), 1)
    #     # 2.2、Angle
    #     sub = torch.mean(torch.abs(f_1.unsqueeze(1) - f_2), 1) # f_1 - f_2
    #     f_dis = torch.sqrt(torch.sum(torch.square(sub), dim=-1)).unsqueeze(-1)
    #     angle = torch.divide(sub, f_dis)
    #     angle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle)
    #     # 2.3、CVA
    #     sub = f_1.unsqueeze(1) - f_2  # f_1 - f_2
    #     cva = torch.mean(torch.sqrt(torch.sum(torch.square(sub), dim=-1)), 1).unsqueeze(1)
    #     # 2.4、Feature Concate
    #     dif_f = self.diff_conv_list[layer_id].to(device)(torch.concat((subabs, angle, cva), dim=1))
    #
    #     return dif_f

class Cross_Attention(nn.Module):
    def __init__(self, channels):
        super(Cross_Attention, self).__init__()
        self.q_conv = Linear(channels, channels)
        self.k_conv = Linear(channels, channels)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = Linear(channels, channels)

        self.fc_gamma = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        self.trans_conv = Linear(channels, channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, value, q_pos, k_pos, mask, knearest_idx, k_num):

        # b, n, c
        if q_pos is not None:
            q = q + q_pos

        if k_pos is not None:
            k = k + k_pos

        # b, c, n
        q = self.q_conv.to(q.device)(q)
        k = self.k_conv.to(k.device)(k)
        v = self.v_conv.to(value.device)(value)

        knearest_idx_1 = knearest_idx.reshape(knearest_idx.shape[0], -1, k_num)
        k = k[knearest_idx_1[1, :, :], :]
        q = q.unsqueeze(1).repeat(1, k_num, 1)

        energy = self.fc_gamma.to(q.device)(q - k)

        if mask is not None:
            energy.masked_fill_(mask, -1e12)
            # energy = torch.multiply(energy, mask)

        attention = self.softmax(energy)
        # b, c, n
        # x_r = torch.multiply(attention, v)
        # x = torch.mean(self.trans_conv.to(x_r.device)(x_r) + value, 1)

        res = torch.einsum('mnf,mnf->mf', attention, v)
        x = self.trans_conv.to(q.device)(res)
        return x


class Self_Attention(nn.Module):
    def __init__(self, channels):
        super(Self_Attention, self).__init__()
        self.energy_conv = Linear(channels, channels)
        self.v_conv = Linear(channels, channels)
        self.pos_norm = FastBatchNorm1d(channels)

        self.trans_conv = Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, value, xyz):
        # b, n, c
        if xyz is not None:
            value = self.pos_norm.to(value.device)(value + xyz)

        # b, c, n
        energy = self.energy_conv.to(value.device)(value)
        value = self.v_conv.to(value.device)(value)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.multiply(value, attention)

        x = self.trans_conv.to(value.device)(x_r) + value
        return x

def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=True,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st
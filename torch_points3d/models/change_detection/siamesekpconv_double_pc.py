from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
import copy

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

class BaseFactoryPSI:
    def __init__(self, module_name_down_1, module_name_down_2, module_name_up_1, module_name_up_2, modules_lib):
        self.module_name_down_1 = module_name_down_1
        self.module_name_down_2 = module_name_down_2
        self.module_name_up_1 = module_name_up_1
        self.module_name_up_2 = module_name_up_2
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            if "1" in flow:
                return getattr(self.modules_lib, self.module_name_up_1, None)
            else:
                return getattr(self.modules_lib, self.module_name_up_2, None)
        elif "1" in flow:
            return getattr(self.modules_lib, self.module_name_down_1, None)
        else:
            return getattr(self.modules_lib, self.module_name_down_2, None)


####################SIAMESE ENCODER FUSION KP CONV ############################
class SiameseKPConv(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self.threshold_dis_init = 0.5
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
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
        opt = copy.deepcopy(option)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}
        self._init_from_compact_format(opt, model_type, dataset, modules)

        # Unshared weight :  2 down modules
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))
        self.loss_names = ["loss_cd"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]

        # Build Difference Module
        self.batch_size = 10
        self.self_att_list = []
        # self_att = Attention(in_feat)
        self.diff_conv_list = Sequential()
        self.pos_xyz_list = Sequential()
        init_dim = 128
        up_dim_init = 2048
        self.k_n = [1, 1, 1, 1, 1]
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            self_att = Attention(int(init_dim * pow(2, i)))
            # self_att2 = Attention(init_dim * pow(2, i))
            # self.self_att_list.append([self_att1, self_att2])
            self.self_att_list.append(self_att)
            self.diff_conv_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(init_dim * pow(2, i), init_dim * pow(2, i) // 2, bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i) // 2),
                        nn.LeakyReLU(0.2),
                    ]
                ),
            )
            # self.sift_module.add_module(
            #     str(i),
            #     Sequential(
            #         *[
            #             conv_bn(3, init_dim * pow(2, i), [1, 2], [1, 2]),
            #             conv_bn(init_dim * pow(2, i), init_dim * pow(2, i), [1, 2], [1, 2]),
            #             conv_bn(init_dim * pow(2, i), init_dim * pow(2, i), [1, 2], [1, 2]),
            #         ]
            #     ),
            # )
            self.pos_xyz_list.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(8, init_dim * pow(2, i) // 2, bias=False),
                        FastBatchNorm1d(init_dim * pow(2, i)// 2),
                        nn.LeakyReLU(0.2)
                    ]
                ),
            )


    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules_1 = nn.ModuleList()
        self.down_modules_2 = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules_1 = nn.ModuleList()
        self.up_modules_2 = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv_1.get('save_sampling_id')
        self.save_sampling_id_2 = opt.down_conv_2.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name_1 = opt.down_conv_1.module_name
        down_conv_cls_name_2 = opt.down_conv_2.module_name
        up_conv_cls_name_1 = opt.up_conv_1.module_name if opt.get('up_conv_1') is not None else None
        up_conv_cls_name_2 = opt.up_conv_2.module_name if opt.get('up_conv_2') is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name_1, down_conv_cls_name_2, up_conv_cls_name_1, up_conv_cls_name_2, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_1, i, "DOWN_1")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_1.append(down_module)
        for i in range(len(opt.down_conv_2.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_2, i, "DOWN_2")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_2.append(down_module)

        # Up modules
        if up_conv_cls_name_1:
            for i in range(len(opt.up_conv_1.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_1, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_1.append(up_module)
        if up_conv_cls_name_2:
            for i in range(len(opt.up_conv_2.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv_2, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules_2.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

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
            self.labels0 = data.y.to(device)
            self.labels1 = data.y_target.to(device)
            self.labels = [self.labels0, self.labels1]
        else:
            self.input = data
            self.labels0 = None
            self.labels1 = None
            self.labels = [self.labels0, self.labels1]

    # def forward(self, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
    #     nn_list = knn(data0.pos, data1.pos, self.k_n[0], data0.batch, data1.batch)
    #     mask = self.get_mask(self, data0.x, data1.x, nn_list)
    #     mask = self.sampler_data(mask, self.down_modules_2[0].sampler)
    #     mask_l.append(mask)
    #
    #     diff = data1.clone()
    #     diff.x = self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list, 0)
    #     stack_down.append(diff)
    #     data1.x = torch.cat((data1.x, diff.x), axis=1)
    #
    #     for i in range(1, len(self.down_modules_1) - 1):
    #         data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
    #         mask = self.sampler_data(mask, self.down_modules_2[i].sampler)
    #         mask_l.append(mask)
    #         nn_list = knn(data0.pos, data1.pos, self.k_n[i], data0.batch, data1.batch)
    #         diff = data1.clone()
    #         diff.x = self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list, i)
    #         stack_down.append(diff)
    #         data1.x = torch.cat((data1.x, diff.x), axis=1)
    #     #1024
    #     data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
    #     mask = self.sampler_data(mask, self.down_modules_2[-1].sampler)
    #     mask_l.append(mask)
    #     nn_list = knn(data0.pos, data1.pos, self.k_n[-1], data0.batch, data1.batch)
    #     data = data1.clone()
    #     data.x = self.feature_difference_module(data0.x, data1.x, data0.pos, data1.pos, nn_list, len(self.down_modules_1) - 1)
    #     data.x = torch.cat((data1.x, data.x), axis=1)
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append(data1)
    #         data = self.inner_modules[0](data)
    #         innermost = True
    #     for i in range(len(self.up_modules)):
    #         if i == 0 and innermost:
    #             data = self.up_modules[i]((data, stack_down.pop()))
    #         else:
    #             data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
    #     last_feature = data.x
    #     if self._use_category:
    #         self.output = self.FC_layer(last_feature, self.category)
    #     else:
    #         self.output = self.FC_layer(last_feature)
    #
    #     if self.labels is not None:
    #         self.compute_loss()
    #
    #     self.data_visual = self.input1
    #     self.data_visual.pred = torch.max(self.output, -1)[1]
    #
    #     return self.output

    # def forward(self, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
    #     nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     diff = data1.clone()
    #     diff.x = data1.x - data0.x[nn_list[1, :], :]
    #     stack_down.append(diff)
    #     data1.x = torch.cat((data1.x, diff.x), axis=1)
    #
    #     for i in range(1, len(self.down_modules_1) - 1):
    #         data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
    #         nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         diff = data1.clone()
    #         diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         stack_down.append(diff)
    #         data1.x = torch.cat((data1.x, diff.x), axis=1)
    #     #1024
    #     data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
    #
    #     nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     data = data1.clone()
    #     data.x = data1.x - data0.x[nn_list[1,:],:]
    #     data.x = torch.cat((data1.x, data.x), axis=1)
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append(data1)
    #         data = self.inner_modules[0](data)
    #         innermost = True
    #     for i in range(len(self.up_modules)):
    #         if i == 0 and innermost:
    #             data = self.up_modules[i]((data, stack_down.pop()))
    #         else:
    #             data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
    #     last_feature = data.x
    #     if self._use_category:
    #         self.output = self.FC_layer(last_feature, self.category)
    #     else:
    #         self.output = self.FC_layer(last_feature)
    #
    #     if self.labels is not None:
    #         self.compute_loss()
    #
    #     self.data_visual = self.input1
    #     self.data_visual.pred = torch.max(self.output, -1)[1]
    #
    #     return self.output

    # def forward(self, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     # 3、Get Transformer Mask
    #     data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
    #     nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     mask0 = self.get_mask(data1.pos, data0.pos, nn_list0)
    #     nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     mask1 = self.get_mask(data0.pos, data1.pos, nn_list1)
    #     mask_l.append([mask0, mask1])
    #
    #     diff0 = data0.clone()
    #     diff0.x = diff0.x = data0.x - data1.x[nn_list0[1, :], :]
    #     diff1 = data1.clone()
    #     diff1.x = diff1.x = data1.x - data0.x[nn_list1[1, :], :]
    #     stack_down.append([diff0, diff1])
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data0.mask = mask0
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #     data1.mask = mask1
    #
    #     for i in range(1, len(self.down_modules_1) - 1):
    #         data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
    #         mask_l.append([data0.mask, data1.mask])
    #
    #         nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #         diff0 = data0.clone()
    #         diff0.x = diff0.x = data0.x - data1.x[nn_list0[1, :], :]
    #         nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         diff1 = data1.clone()
    #         diff1.x = diff1.x = data1.x - data0.x[nn_list1[1, :], :]
    #
    #         data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #         data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #         stack_down.append([diff0, diff1])
    #
    #     #1024
    #     data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
    #     mask_l.append([data0.mask, data1.mask])
    #
    #     nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     diff0 = data0.clone()
    #     diff0.x = diff0.x = data0.x - data1.x[nn_list0[1, :], :]
    #     nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     diff1 = data1.clone()
    #     diff1.x = diff1.x = data1.x - data0.x[nn_list1[1, :], :]
    #
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append([diff0, diff1])
    #         data0 = self.inner_modules[0](data0)
    #         data1 = self.inner_modules[0](data1)
    #         innermost = True
    #     for i in range(len(self.up_modules_1)):
    #         if i == 0 and innermost:
    #             diff0, diff1 = stack_down.pop()
    #             data0 = self.up_modules_1[i]((data0, diff0))
    #             data1 = self.up_modules_2[i]((data1, diff1))
    #         else:
    #             diff0, diff1 = stack_down.pop()
    #             data0 = self.up_modules_1[i]((data0, diff0), precomputed=self.upsample_target)
    #             data1 = self.up_modules_2[i]((data1, diff1), precomputed=self.upsample_target)
    #     last_feature0 = data0.x
    #     last_feature1 = data1.x
    #     if self._use_category:
    #         self.output0 = self.FC_layer(last_feature0, self.category)
    #         self.output1 = self.FC_layer(last_feature1, self.category)
    #     else:
    #         self.output0 = self.FC_layer(last_feature0)
    #         self.output1 = self.FC_layer(last_feature1)
    #
    #     if self.labels0 is not None and self.labels1 is not None:
    #         self.compute_loss()
    #
    #     self.data_visual0 = self.input0
    #     self.data_visual0.pred = torch.max(self.output0, -1)[1]
    #     self.data_visual1 = self.input1
    #     self.data_visual1.pred = torch.max(self.output1, -1)[1]
    #
    #     self.output = [self.output0, self.output1]
    #
    #     return self.output0, self.output1du

    # def forward(self, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     # 3、Get Transformer Mask
    #     data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
    #     nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     mask0 = self.get_mask(data1.pos, data0.pos, nn_list0)
    #     nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     mask1 = self.get_mask(data0.pos, data1.pos, nn_list1)
    #     mask_l.append([mask0, mask1])
    #
    #     diff0 = data0.clone()
    #     diff0.x = self.nearest_feature_difference(data0.x, data1.x, nn_list0, 0, data0.x.device) #diff.x = data1.x - data0.x[nn_list[1, :], :]
    #     diff1 = data1.clone()
    #     diff1.x = self.nearest_feature_difference(data1.x, data0.x, nn_list1, 0, data1.x.device) #diff.x = data1.x - data0.x[nn_list[1, :], :]
    #     stack_down.append([diff0, diff1])
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data0.mask = mask0
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #     data1.mask = mask1
    #
    #     for i in range(1, len(self.down_modules_1) - 1):
    #         data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
    #         mask_l.append([data0.mask, data1.mask])
    #
    #         nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #         diff0 = data0.clone()
    #         diff0.x = self.nearest_feature_difference(data0.x, data1.x, nn_list0, i, data0.x.device) # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         diff1 = data1.clone()
    #         diff1.x = self.nearest_feature_difference(data1.x, data0.x, nn_list1, i, data1.x.device) # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #
    #         data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #         data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #         stack_down.append([diff0, diff1])
    #
    #     #1024
    #     data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
    #     mask_l.append([data0.mask, data1.mask])
    #
    #     nn_list0 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     diff0 = data0.clone()
    #     diff0.x = self.nearest_feature_difference(data0.x, data1.x, nn_list0, len(self.down_modules_1) - 1, data1.x.device) # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     diff0.x = self.self_att_list[0](diff0.x, None, False, mask_l[-1][0])
    #     nn_list1 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     diff1 = data1.clone()
    #     diff1.x = self.nearest_feature_difference(data1.x, data0.x, nn_list1, len(self.down_modules_1) - 1, data1.x.device) # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     diff1.x = self.self_att_list[0](diff1.x, None, False, mask_l[-1][1])
    #
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append([diff0, diff1])
    #         data0 = self.inner_modules[0](data0)
    #         data1 = self.inner_modules[0](data1)
    #         innermost = True
    #     for i in range(len(self.up_modules_1)):
    #         if i == 0 and innermost:
    #             diff0, diff1 = stack_down.pop()
    #             diff0.x = self.self_att_list[i+1](diff0.x, None, False, mask_l[-2 - i][0])
    #             diff1.x = self.self_att_list[i+1](diff1.x, None, False, mask_l[-2 - i][1])
    #             data0 = self.up_modules_1[i]((data0, diff0))
    #             data1 = self.up_modules_2[i]((data1, diff1))
    #         else:
    #             diff0, diff1 = stack_down.pop()
    #             diff0.x = self.self_att_list[i+1](diff0.x, None, False, mask_l[-2 - i][0])
    #             diff1.x = self.self_att_list[i + 1](diff1.x, None, False, mask_l[-2 - i][1])
    #
    #             data0 = self.up_modules_1[i]((data0, diff0), precomputed=self.upsample_target)
    #             data1 = self.up_modules_2[i]((data1, diff1), precomputed=self.upsample_target)
    #             # data.x = self.self_att_list[i](data.x, None, False, mask_l[-2 - i])
    #     last_feature0 = data0.x
    #     last_feature1 = data1.x
    #     if self._use_category:
    #         self.output0 = self.FC_layer(last_feature0, self.category)
    #         self.output1 = self.FC_layer(last_feature1, self.category)
    #     else:
    #         self.output0 = self.FC_layer(last_feature0)
    #         self.output1 = self.FC_layer(last_feature1)
    #
    #     if self.labels0 is not None and self.labels1 is not None:
    #         self.compute_loss()
    #
    #     self.data_visual0 = self.input0
    #     self.data_visual0.pred = torch.max(self.output0, -1)[1]
    #     self.data_visual1 = self.input1
    #     self.data_visual1.pred = torch.max(self.output1, -1)[1]
    #
    #     self.output = [self.output0, self.output1]
    #
    #     return self.output0, self.output1

    # def forward(self, *args, **kwargs) -> Any:
    #     """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
    #     stack_down = []
    #     mask_l = []
    #
    #     data0 = self.input0
    #     data1 = self.input1
    #
    #     # 3、Get Transformer Mask
    #     data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
    #     nn_list01 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     nn_list00 = knn(data0.pos, data0.pos, 2, data0.batch, data0.batch)
    #     mask0 = self.get_mask(data1.pos, data0.pos, nn_list01, nn_list00)
    #     nn_list10 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     nn_list11 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch)
    #     mask1 = self.get_mask(data0.pos, data1.pos, nn_list10, nn_list11)
    #     mask_l.append([mask0, mask1])
    #
    #     diff0 = data0.clone()
    #     diff0.x = self.nearest_feature_difference(data1, data0, nn_list01, nn_list00, 0) #diff.x = data1.x - data0.x[nn_list[1, :], :]
    #     diff0.x = self.self_att_list[0](diff0.x, None, False, mask0)
    #     diff1 = data1.clone()
    #     diff1.x = self.nearest_feature_difference(data0, data1, nn_list10, nn_list11, 0) #diff.x = data1.x - data0.x[nn_list[1, :], :]
    #     diff1.x = self.self_att_list[0](diff1.x, None, False, mask1)
    #     stack_down.append([diff0, diff1])
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #
    #     for i in range(1, len(self.down_modules_1) - 1):
    #         data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
    #         data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
    #         nn_list01 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #         nn_list00 = knn(data0.pos, data0.pos, 2, data0.batch, data0.batch)
    #         mask0 = self.get_mask(data1.pos, data0.pos, nn_list01, nn_list00)
    #         nn_list10 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #         nn_list11 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch)
    #         mask1 = self.get_mask(data0.pos, data1.pos, nn_list10, nn_list11)
    #
    #         diff0 = data0.clone()
    #         diff0.x = self.nearest_feature_difference(data1, data0, nn_list01, nn_list00, i) # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         diff0.x = self.self_att_list[i](diff0.x, None, False, mask0)
    #         diff1 = data1.clone()
    #         diff1.x = self.nearest_feature_difference(data0, data1, nn_list10, nn_list11, i) # diff.x = data1.x - data0.x[nn_list[1,:],:]
    #         diff1.x = self.self_att_list[i](diff1.x, None, False, mask1)
    #
    #         data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #         data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #         stack_down.append([diff0, diff1])
    #
    #     #1024
    #     data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
    #     data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
    #     nn_list01 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
    #     nn_list00 = knn(data0.pos, data0.pos, 2, data0.batch, data0.batch)
    #     mask0 = self.get_mask(data1.pos, data0.pos, nn_list01, nn_list00)
    #     nn_list10 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
    #     nn_list11 = knn(data1.pos, data1.pos, 2, data1.batch, data1.batch)
    #     mask1 = self.get_mask(data0.pos, data1.pos, nn_list10, nn_list11)
    #
    #     diff0 = data0.clone()
    #     diff0.x = self.nearest_feature_difference(data1, data0, nn_list01, nn_list00, -1) # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     diff0.x = self.self_att_list[-1](diff0.x, None, False, mask0)
    #     diff1 = data1.clone()
    #     diff1.x = self.nearest_feature_difference(data0, data1, nn_list10, nn_list11, -1) # data.x = data1.x - data0.x[nn_list[1,:],:]
    #     diff1.x = self.self_att_list[-1](diff1.x, None, False, mask1)
    #
    #     data0.x = torch.cat((data0.x, diff0.x), axis=1)
    #     data1.x = torch.cat((data1.x, diff1.x), axis=1)
    #     innermost = False
    #     if not isinstance(self.inner_modules[0], Identity):
    #         stack_down.append([diff0, diff1])
    #         data0 = self.inner_modules[0](data0)
    #         data1 = self.inner_modules[0](data1)
    #         innermost = True
    #     for i in range(len(self.up_modules_1)):
    #         if i == 0 and innermost:
    #             diff0, diff1 = stack_down.pop()
    #             data0 = self.up_modules_1[i]((data0, diff0))
    #             data1 = self.up_modules_2[i]((data1, diff1))
    #         else:
    #             diff0, diff1 = stack_down.pop()
    #             data0 = self.up_modules_1[i]((data0, diff0), precomputed=self.upsample_target)
    #             data1 = self.up_modules_2[i]((data1, diff1), precomputed=self.upsample_target)
    #     last_feature0 = data0.x
    #     last_feature1 = data1.x
    #     if self._use_category:
    #         self.output0 = self.FC_layer(last_feature0, self.category)
    #         self.output1 = self.FC_layer(last_feature1, self.category)
    #     else:
    #         self.output0 = self.FC_layer(last_feature0)
    #         self.output1 = self.FC_layer(last_feature1)
    #
    #     if self.labels0 is not None and self.labels1 is not None:
    #         self.compute_loss()
    #
    #     self.data_visual0 = self.input0
    #     self.data_visual0.pred = torch.max(self.output0, -1)[1]
    #     self.data_visual1 = self.input1
    #     self.data_visual1.pred1 = torch.max(self.output1, -1)[1]
    #
    #     self.output = [self.output0, self.output1]
    #
    #     return self.output0, self.output1

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []
        mask_l = []

        data0 = self.input0
        data1 = self.input1

        # 3、Get Transformer Mask
        data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
        nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

        # dis0 = torch.sqrt(torch.sum(torch.square(data0.pos - data1.pos[nn_list1[1, :], :]), dim=-1))
        # dis1 = torch.sqrt(torch.sum(torch.square(data1.pos - data0.pos[nn_list0[1, :], :]), dim=-1))
        # tt1 = dis0.cpu().numpy()
        # tt2 = dis1.cpu().numpy()
        # dist_med_1 = torch.median(dis0)
        # dist_med_2 = torch.median(dis1)

        diff0 = data0.clone()
        diff0.x = data0.x - data1.x[nn_list1[1, :], :]
        diff1 = data1.clone()
        diff1.x = data1.x - data0.x[nn_list0[1, :], :]
        stack_down.append([diff0, diff1])

        for i in range(1, len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)

            nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
            nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff0 = data0.clone()
            diff0.x = data0.x - data1.x[nn_list1[1, :], :]
            diff1 = data1.clone()
            diff1.x = data1.x - data0.x[nn_list0[1, :], :]

            stack_down.append([diff0, diff1])

        # 1024
        data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
        nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data0 = data0.clone()
        data1 = data1.clone()
        data0.x = data0.x - data1.x[nn_list1[1, :], :]
        data1.x = data1.x - data0.x[nn_list0[1, :], :]

        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append([diff0, diff1])
            data0 = self.inner_modules[0](data0)
            data1 = self.inner_modules[0](data1)
            innermost = True
        for i in range(len(self.up_modules_1)):
            if i == 0 and innermost:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i]((data0, diff0))
                data1 = self.up_modules_2[i]((data1, diff1))
            else:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i]((data0, diff0), precomputed=self.upsample_target)
                data1 = self.up_modules_2[i]((data1, diff1), precomputed=self.upsample_target)
        last_feature0 = data0.x
        last_feature1 = data1.x
        if self._use_category:
            self.output0 = self.FC_layer(last_feature0, self.category)
            self.output1 = self.FC_layer(last_feature1, self.category)
        else:
            self.output0 = self.FC_layer(last_feature0)
            self.output1 = self.FC_layer(last_feature1)

        if self.labels0 is not None and self.labels1 is not None:
            self.compute_loss()

        self.data_visual0 = self.input0
        self.data_visual0.pred = torch.max(self.output0, -1)[1]
        self.data_visual1 = self.input1
        self.data_visual1.pred = torch.max(self.output1, -1)[1]

        self.output = [self.output0, self.output1]

        return self.output0, self.output1

    def sampler_data(self, data, samplers):
        for sampler in samplers:
            if sampler is not None:
                data = sampler(data.clone())
        return data

    def compute_loss(self):
        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output0.device)
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
            self.loss_seg = F.nll_loss(self.output0, self.labels0, weight=self._weight_classes, ignore_index=self._ignore_label) + F.nll_loss(self.output1, self.labels1, weight=self._weight_classes, ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.output0, self.labels0, weight=self._weight_classes) + F.nll_loss(self.output1, self.labels1, weight=self._weight_classes)

        if torch.isnan(self.loss_seg).sum() == 1:
            print(self.loss_seg)
        self.loss += self.loss_seg

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    # Single Target
    def feature_difference_module(self, f_1, f_2, f1_xyz, f2_xyz, knearest_idx_2, layer_index):
        shape_0 = knearest_idx_2.shape[0]
        k_n_mult_ptnum = knearest_idx_2.shape[1]
        device = f1_xyz.device
        self.threshold_dis = self.threshold_dis_init * np.power(4, layer_index)
        try:
            knearest_idx_2 = knearest_idx_2.reshape(shape_0, f_2.shape[0], self.k_n[layer_index])
        except:
            print("wrong dim")

        enc_xyz_1 = self.pos_xyz_list[layer_index].to(device)(f1_xyz)
        enc_xyz_2 = self.pos_xyz_list[layer_index].to(device)(f2_xyz)

        f_1 = self.self_att_list[layer_index][0](f_1, enc_xyz_1, True, None)
        f_2 = self.self_att_list[layer_index][0](f_2, enc_xyz_2, True, None)

        # 3、Feature Difference Module
        dif_f_2 = self.nearest_feature_difference(f_2, f_1, knearest_idx_2, layer_index, device)

        if layer_index == 0:

            # 3、Get Transformer Mask
            mask_2 = self.get_mask(f1_xyz, f2_xyz, knearest_idx_2)

            # 4、Self Attention
            fout2 = self.self_att_list[layer_index][1](dif_f_2, None, False, mask_2)
        else:
            # 4、Self Attention
            fout2 = self.self_att_list[layer_index][1](dif_f_2, None, False, None)

        return fout2

    def get_mask(self, f_1, f_2, knearest_idx_1, knearest_idx_2):
        knearest_idx_1 = knearest_idx_1.reshape(knearest_idx_1.shape[0], -1, 1)
        knearest_idx_2 = knearest_idx_2.reshape(knearest_idx_2.shape[0], -1, 2)
        # 2在1中的邻近点
        nearest_2of1 = f_1[knearest_idx_1[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, 1, 1)
        dis_mat_1 = torch.mean(torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of1), dim=-1)), 1).view(-1, 1)

        nearest_2of2 = f_2[knearest_idx_2[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, 2, 1)
        dis_mat_2 = torch.max(torch.sqrt(torch.sum(torch.square(f_2_ex - nearest_2of2), dim=-1)), dim=-1, keepdim=True)[0]

        mask_2 = torch.where((dis_mat_1 > 2 * dis_mat_2) | (2 * dis_mat_1 > dis_mat_2), 2.0, 1.0)

        return mask_2

    def nearest_feature_difference(self, data_1, data_2, k_idx1, k_idx2, layer_id):
        knearest_idx_1 = k_idx1.reshape(k_idx1.shape[0], -1, 1)
        knearest_idx_2 = k_idx2.reshape(k_idx2.shape[0], -1, 2)
        # Position Difference
        xyz_tile = data_2.pos.unsqueeze(1).repeat(1, self.k_n[layer_id], 1)
        # PC2-PC11
        pos_21 = data_1.pos[knearest_idx_1[1, :, :], :]
        relative_xyz_211 = xyz_tile - pos_21
        relative_dis_211 = torch.sqrt(torch.sum(torch.square(relative_xyz_211), dim=-1)).unsqueeze(-1)
        # PC2-PC12
        pos_12 = data_2.pos[knearest_idx_2[1, :, :], :][:, 1, :].unsqueeze(1)
        relative_xyz_2112 = pos_12 - pos_21
        relative_dis_2112 = torch.sqrt(torch.sum(torch.square(relative_xyz_2112), dim=-1)).unsqueeze(-1)

        relative_pos_feature = torch.cat([relative_dis_211, relative_xyz_211, relative_xyz_2112, relative_dis_2112], axis=-1)
        diff_f_xyz = self.pos_xyz_list[layer_id](relative_pos_feature)

        # Feature Difference
        feat_f_2 = data_1.x[knearest_idx_1[1, :, :], :]
        diff_f_feat = data_2.x.unsqueeze(1).repeat(1, self.k_n[layer_id], 1) - feat_f_2
        diff_f_feat = self.diff_conv_list[layer_id](diff_f_feat)

        # Feature Concat
        dif_f = torch.mean(torch.cat([diff_f_feat, diff_f_xyz], axis=-1), 1)
        return dif_f


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.energy_conv = Linear(channels, channels)
        self.v_conv = Linear(channels, channels)
        self.pos_norm = FastBatchNorm1d(channels)

        self.trans_conv = Linear(channels, channels)
        self.after_norm = FastBatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v, xyz, add_pos, mask):
        # b, n, c
        if add_pos:
            value = self.pos_norm.to(v.device)(v + xyz)
        else:
            value = v

        # b, c, n
        energy = self.energy_conv.to(value.device)(value)
        value = self.v_conv.to(value.device)(value)

        if mask is not None:
            # energy.masked_fill_(mask, -1e12)
            energy = torch.multiply(energy, mask)

        attention = self.softmax(energy)
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.multiply(value, attention)

        x = self.act.to(value.device)(self.after_norm.to(value.device)(self.trans_conv.to(value.device)(x_r))) + v
        return x
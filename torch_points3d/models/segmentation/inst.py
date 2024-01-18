from typing import Any
import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn, Tensor
import copy

import numpy as np
import torch
from .base import Segmentation_MP
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform.grid_transform import group_data

log = logging.getLogger(__name__)

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def get_instance_masks(sem_labels, inst_labels, input_batch, ignore_class_threshold=100, filter_out_classes=[], label_offset=0):
    target = []
    sem_labels = sem_labels.squeeze(-1)
    inst_labels = inst_labels.squeeze(-1)

    batch_list = input_batch.unique()

    for _batch in batch_list:
        label_ids = []
        inst_ids = []
        masks = []
        batch_idx = torch.argwhere(input_batch==_batch).squeeze(-1)
        instance_ids = inst_labels[batch_idx].unique()

        for instance_id in instance_ids:
            if instance_id == -100:
                continue

            ttata = sem_labels[batch_idx].cpu().numpy().reshape(-1, 1)
            adac = inst_labels[batch_idx].cpu().numpy().reshape(-1, 1)
            ada = np.concatenate((ttata, adac), axis=-1)

            tmp = sem_labels[batch_idx][
                inst_labels[batch_idx] == instance_id
            ]
            label_id = tmp[0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            label_ids.append(label_id)
            inst_ids.append(instance_id)
            masks.append(torch.where(inst_labels[batch_idx] == instance_id, 1.0, 0.0))

        if len(label_ids) == 0:
            return list()

        # l = torch.clamp(label_ids - label_offset, min=0)
        target.append({"labels": label_ids, "masks": masks, "insts":inst_ids})
    return target

class KPConvPaper(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
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
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

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
        self.loss_names = ["loss_seg"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]

        # Build Difference Module
        self.sample_feat = []
        self.sample_y = []
        self.AptSampling = []
        init_dim = 64
        self.k_n = [16, 16, 8, 8, 4]
        self.ratio = [0.8, 0.8, 0.6, 0.6, 0.5]
        self.sem_ratio = [0.1, 0.3, 0.5, 0.5]
        self.mask_dim = 128
        num_classes = 15
        for i in range(len(opt.down_conv.down_conv_nn)):
            adpsampling = AdaptiveSampling3D(self.ratio[i], init_dim * pow(2, i), self.k_n[i], num_classes)
            self.AptSampling.append(adpsampling)

        self.decoder_query_norm_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()

        self.transformer_feat_cross_attention_layers = nn.ModuleList()
        self.transformer_feat_ffn_layers = nn.ModuleList()
        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.Sample_FC_layer = Sequential()

        self.decoder_layers = 3
        query_dim, feat_dim, mask_dim = 2048, 64, 128
        nheads, dim_feedforward, pre_norm = 8, 128, False
        ini_sample_dim = 128
        for i in range(self.decoder_layers):
            self.Sample_FC_layer.add_module(
                str(i),
                Sequential(
                    *[
                        Lin(ini_sample_dim, self._num_classes, bias=False),
                        nn.LogSoftmax(-1),
                    ]
                ),
            )
            ini_sample_dim *= 2

            self.decoder_query_norm_layers.append(nn.LayerNorm(feat_dim))
            self.mask_embed_layers.append(nn.Linear(query_dim, mask_dim))
            self.mask_features_layers.append(nn.Linear(feat_dim, mask_dim))

            self.transformer_feat_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=mask_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_feat_ffn_layers.append(
                FFNLayer(
                    d_model=mask_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=mask_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=mask_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=mask_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

            query_dim = mask_dim
            feat_dim = mask_dim

        self.Sample_FC_layer.add_module(
            str(i+1),
            Sequential(
                *[
                    Lin(ini_sample_dim, self._num_classes, bias=False),
                    nn.LogSoftmax(-1),
                ]
            ),
        )
        self.mask_embed_layers.append(nn.Linear(query_dim, last_mlp_opt.nn[0] + self._num_categories))
        self.mask_features_layers.append(nn.Linear(feat_dim, last_mlp_opt.nn[0] + self._num_categories))

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        data.x = add_ones(data.pos, data.f, True)
        del data.f

        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        target = get_instance_masks(data.y, data.inst_y, data.batch)

        self.input = data
        self.labels = data.y
        self.inst_labels = data.inst_y
        self.batch_idx = data.batch
        #self.target = target

        if self._use_category:
            self.category = data.category

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data = self.input
        sample_idx = 0
        for i in range(len(self.down_modules) - 1):
            if self.down_modules[i].sampler[0] != None:
                data, _, s_y = self.AptSampling[i](data.clone(), self.sem_ratio[sample_idx])
                s_feat = self.Sample_FC_layer[sample_idx](data.x)
                self.sample_feat.append(s_feat)
                self.sample_y.append(s_y)
                sample_idx += 1
            data = self.down_modules[i](data)
            stack_down.append(data)

        if self.down_modules[-1].sampler[0] != None:
            data, self.inst_y, self.labels = self.AptSampling[-1](data.clone(), self.sem_ratio[-1])
            s_feat = self.Sample_FC_layer[sample_idx](data.x)
            self.sample_feat.append(s_feat)
            self.sample_y.append(self.labels)
            self.batch_inst_y = data.batch
        data = self.down_modules[-1](data)
        innermost = False

        queries = data.x

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample)

        pc_features = data.x
        last_feature, self.output_mask = self.MaskDecoder(pc_features, queries)

        if self._use_category:
            self.output = self.FC_layer(last_feature, self.category)
        else:
            self.output = self.FC_layer(last_feature)

        if self.labels is not None:
            self.compute_loss()

        self.data_visual = self.input
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
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # Final cross entrop loss
        # # 1、Class Loss
        # self.loss_class = F.nll_loss(self.output, self.labels, weight=self._weight_classes, ignore_index=IGNORE_LABEL)
        # self.loss += self.loss_class
        # 1、Mask Loss
        batch_inst_idx_list = self._get_idx_from_batch(self.batch_inst_y)
        batch_idx_list = self._get_idx_from_batch(self.batch_idx)
        i = 0
        for batch_inst_idx, batch_idx in zip(batch_inst_idx_list, batch_idx_list):
            b_inst_y = self.inst_y[batch_inst_idx].clone().detach().cpu().numpy()
            uni_inst_ys = list(set(b_inst_y))
            for uni_inst_y in uni_inst_ys:
                if uni_inst_y == -100:
                    continue

                target_idx = torch.argwhere(self.inst_labels[batch_idx]==uni_inst_y)
                # 1.1、Mask-> mask Loss
                target_mask = torch.where(self.inst_labels[batch_idx]==uni_inst_y, 1.0, 0.0)

                q_idx = torch.argwhere(self.inst_y[batch_inst_idx]==uni_inst_y).squeeze(-1)
                out_mask = self.output_mask[batch_inst_idx].permute(1, 0)[batch_idx].permute(1, 0)
                out_mask = out_mask[q_idx]

                inst_num = out_mask.shape[0]
                target_mask_padding = target_mask.unsqueeze(0).repeat(inst_num, 1)

                min_loss_mask, min_loss_dice, min_idx = 10000000, 10000000, -1
                for q in range(inst_num):
                    l_out_mask, l_target_mask_padding = out_mask[q].unsqueeze(0), target_mask_padding[q].unsqueeze(0)
                    loss_mask = sigmoid_ce_loss_jit(l_out_mask, l_target_mask_padding, 1)
                    loss_dice = dice_loss_jit(l_out_mask, l_target_mask_padding, 1)
                    if loss_mask < min_loss_mask:
                        min_loss_mask = loss_mask
                        min_idx = q
                    if loss_dice < min_loss_dice:
                        min_loss_dice = loss_dice
                        min_idx = q

                self.loss += min_loss_mask
                self.loss += min_loss_dice

            # 1.2、Mask-> cls Loss
            target_cls = self.labels[batch_inst_idx]
            l_out_cls = self.output[batch_inst_idx]
            loss_cls = F.cross_entropy(
                l_out_cls,
                target_cls,
                ignore_index = 0
            )
            self.loss += loss_cls

        # 2、Sample Loss
        for s_f, s_y in zip(self.sample_feat, self.sample_y):
            self.loss_sample = F.cross_entropy(s_f, s_y, weight=self._weight_classes, ignore_index=0)
            self.loss += self.loss_sample

        print("aaa")


    def _get_idx_from_batch(self, batch):
        batch_idx = []
        np_b = batch.clone().detach().cpu().numpy()
        uni_np_bs = list(set(np_b))
        for uni_np_b in uni_np_bs:
            uni_np_b_idx = np.argwhere(np_b == uni_np_b).squeeze(-1)
            batch_idx.append(uni_np_b_idx)
        return batch_idx

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    def MaskDecoder(self, feature, query):
        for layer_idx in range(self.decoder_layers):
            norm_features = self.decoder_query_norm_layers[layer_idx](feature)
            outputs_mask_embed = self.mask_embed_layers[layer_idx](query)
            outputs_mask_features = self.mask_features_layers[layer_idx](norm_features)
            outputs_mask = torch.einsum("qc,lc->ql", outputs_mask_embed, outputs_mask_features)

            # Masked Attention mask
            attn_mask = (outputs_mask < 0.)

            # Refine feature
            feature = self.transformer_feat_cross_attention_layers[layer_idx](outputs_mask_features, outputs_mask_embed)
            feature = self.transformer_feat_ffn_layers[layer_idx](feature)

            # Refine Query
            query = self.transformer_query_cross_attention_layers[layer_idx](outputs_mask_embed, feature, attn_mask)
            query = self.transformer_query_self_attention_layers[layer_idx](query)
            query = self.transformer_query_ffn_layers[layer_idx](query)

        outputs_mask_embed = self.mask_embed_layers[-1](query)
        outputs_mask_features = self.mask_features_layers[-1](feature)
        outputs_mask = torch.einsum("qc,lc->ql", outputs_mask_embed, outputs_mask_features)

        return outputs_mask_embed , outputs_mask


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

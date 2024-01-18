import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn

log = logging.getLogger(__name__)

class AdaptiveSampling3D:
    def __init__(self, ratio, d_model, knn_num, num_classes):
        self._ratio = ratio
        self.num_classes = num_classes
        self.knn_num = knn_num
        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.conv_proposal_cls_logits = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes + 1),
        )

    def _get_idx_from_batch(self, batch):
        batch_idx = []
        np_b = batch.clone().detach().cpu().numpy()
        uni_np_bs = list(set(np_b))
        for uni_np_b in uni_np_bs:
            uni_np_b_idx = np.argwhere(np_b == uni_np_b).squeeze(-1)
            batch_idx.append(uni_np_b_idx)
        return batch_idx

    def __call__(self, data, sem_ratio):
        self.Q_linear.to(data.pos.device)
        self.K_linear.to(data.pos.device)
        self.conv_proposal_cls_logits.to(data.pos.device)

        nn_list = knn(data.pos, data.pos, self.knn_num, data.batch, data.batch)
        knn_f = data.x[nn_list[1, :], :].reshape(-1, self.knn_num, data.x.shape[-1])
        cen_f = data.x.unsqueeze(1)
        q, k = cen_f, knn_f
        batch_idx_list = self._get_idx_from_batch(data.batch)
        Topk_list = []
        batch_list, idx_neigh_list, pos_list, x_list, inst_y, y_l = [], [], [], [], [], []
        for batch_idx in batch_idx_list:
            q_b, k_b, x_b = q[batch_idx], k[batch_idx], data.x[batch_idx]

            # Key point score
            Q, K = self.Q_linear(q_b), self.K_linear(k_b)
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            R = torch.std(attn_output_weights, dim=-1, unbiased=False).squeeze(-1)
            key_socre = torch.arange(R.shape[0])
            key_topk_idx = torch.argsort(R, -1)
            key_score_tmp = key_socre.clone()
            key_socre[key_topk_idx] = key_score_tmp

            # Semantic point score
            proposal_cls_logits = self.conv_proposal_cls_logits(x_b)  # b, num, c
            proposal_cls_probs = proposal_cls_logits.softmax(dim=-1)  # b, num, c
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs.max(-1)[1], num_classes=self.num_classes + 1)  # b, c, num
            proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
            sem_socre = torch.arange(proposal_cls_probs.shape[0])
            sem_topk_idx = torch.argsort(proposal_cls_probs.flatten(-1).max(-1)[0], -1)
            sem_score_tmp = sem_socre.clone()
            sem_socre[sem_topk_idx] = sem_score_tmp

            # top-k indices
            sum_score = (1 - sem_ratio) * key_socre + sem_ratio * sem_socre
            topK_num = int(sum_score.shape[0] * self._ratio)
            Topk = torch.topk(sum_score, topK_num, dim=0)[1]  # b, q

            batch_list.append(data.batch[batch_idx][Topk])
            idx_neigh_list.append(data.idx_neighboors[batch_idx][Topk])
            pos_list.append(data.pos[batch_idx][Topk])
            x_list.append(data.x[batch_idx][Topk])
            inst_y.append(data.inst_y[batch_idx][Topk])
            y_l.append(data.y[batch_idx][Topk])

        data.x = torch.concat(x_list)
        data.pos = torch.concat(pos_list)
        data.idx_neighboors = torch.concat(idx_neigh_list)
        data.batch = torch.concat(batch_list)
        inst_y = torch.concat(inst_y)
        y_l = torch.concat(y_l)

        return data, inst_y, y_l
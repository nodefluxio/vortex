import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

from typing import Optional, List
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment

from ...modules.backbones import get_backbone

supported_models = [
    'DETR'
]

_activation_fn = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu
}

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        self.tensors = self.tensors.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        else:
            self.mask = None
        return self

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @classmethod
    def from_batch_tensor(cls, tensor_list: List[torch.Tensor]):
        if any(img.ndim == 4 and img.size(0) == 1 for img in tensor_list):
            tensor_list = list(tensor_list)
            for i in range(len(tensor_list)):
                if tensor_list[i].ndim == 4 and tensor_list[i].size(0) == 1:
                    tensor_list[i] = tensor_list[i][0]
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            ## axis-wise max
            max_size, _ = torch.stack([torch.tensor(x.shape) for x in tensor_list]).max(0)
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size.tolist()
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError("NestedTensor conversion only support 3")
        return cls(tensor, mask)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward,
                                          dropout, activation, normalize_before)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward,
                                          dropout, activation, normalize_before, 
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, 
                                                activation, normalize_before)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if normalize_before else None

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, normalize_before)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class DETR(nn.Module):
    def __init__(self, backbone, n_classes, num_queries=100, train_backbone=True, dilation=False, 
                 aux_loss=False, hidden_dim=256, position_embedding='sine', nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", normalize_before=False, return_intermediate_dec=True,
                 lr_backbone=1e-5, pretrained_backbone=False, **kwargs):

        if 'freeze_backbone' in kwargs:
            warnings.warn("unused 'freeze_backbone' argument.")
            kwargs.pop('freeze_backbone')
        if 'resnet' in backbone or 'resnext' in backbone:
            kwargs['norm_layer'] = FrozenBatchNorm2d
            kwargs['replace_stride_with_dilation'] = [False, False, dilation]
        elif dilation:
            warnings.warn("'dilation' argument is not yet used for backbone other than resnet.")

        super().__init__()
        backbone = get_backbone(backbone, pretrained=pretrained_backbone, **kwargs)
        ## always freeze stage1 and stage2 or freeze all when not train_backbone
        if lr_backbone <= 0:
            train_backbone = False
        for name, p in backbone.named_parameters():
            if not train_backbone or 'stage3' not in name and 'stage4' not in name and 'stage5' not in name:
                p.requires_grad_(False)
        backbone_num_channels = backbone.out_channels[-1]
        self.backbone = nn.Sequential(
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.stage5
        )

        if position_embedding in ('v2', 'sine'):
            self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        elif position_embedding in ('v3', 'learned'):
            self.position_embedding = PositionEmbeddingLearned(hidden_dim//2)
        else:
            raise ValueError("not supported position embedding of '{}'".format(position_embedding))

        self.transformer = Transformer(
            d_model=hidden_dim, dropout=dropout, nhead=nhead,
            dim_feedforward=dim_feedforward, activation=activation,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=return_intermediate_dec
        )

        self.class_embed = nn.Linear(hidden_dim, n_classes+1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        self.lr_backbone = lr_backbone

        self.output_format = {
            "bounding_box": {"indices": [0,1,2,3], "axis": 1},
            "class_confidence": {"indices": [4], "axis": 1},
            "class_label": {"indices": [5], "axis": 1},
        }
        self.task = 'detection'

    def forward(self, samples):
        if isinstance(samples, (list, torch.torch.Tensor)):
            samples = NestedTensor.from_batch_tensor(samples)
        x = self.backbone(samples.tensors)
        mask = F.interpolate(samples.mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        features = NestedTensor(x, mask)
        pos = self.position_embedding(features).to(x.dtype)

        assert mask is not None
        proj = self.input_proj(x)
        hs = self.transformer(proj, mask, self.query_embed.weight, pos)[0]

        output_logits = self.class_embed(hs)
        output_bbox = self.bbox_embed(hs).sigmoid()
        if self.training:
            out = {'logits': output_logits[-1], 'bbox': output_bbox[-1]}
            if self.aux_loss:
                out['aux'] = self._set_aux_loss(output_logits, output_bbox)
            return out
        else:
            return torch.cat((output_bbox[-1], output_logits[-1]), -1)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_logits, outputs_bbox):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'bbox': b}
                for a, b in zip(outputs_logits[:-1], outputs_bbox[:-1])]


class DETRPostProcess(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.additional_inputs = (
            ('score_threshold', (1,)),
        )

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, score_threshold: torch.Tensor, **kwargs) -> torch.Tensor:
        if inputs.ndim != 3:
            raise RuntimeError("DETR postprocess input must have dimension of 3 "
                "(num_batch, num_queries, (num_classes + 1) + 4)")
        num_batch = inputs.size(0)
        results = []
        for i in range(num_batch):
            results.append(self._process_single_batch(inputs[i], score_threshold))
        return tuple(results)

    def _process_single_batch(self, inputs, score_threshold):
        assert inputs.ndim == 2, "single batch post process must have dimension of 2 (num_queries, num_pred)"
        bbox, logits = cxcywh_to_xyxy(inputs[:, :4]), inputs[:, 4:]

        probas = logits.softmax(-1)[:, :-1]
        conf, labels = probas.max(-1, keepdim=True)
        keep = (conf > score_threshold).flatten()

        detection = torch.cat((bbox, conf, labels.float()), -1)
        return detection[keep]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best 
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of shape [batch_size, num_queries, n_classes] with the classification logits
                 "bbox": Tensor of shape [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of shape [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "bbox": Tensor of shape [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, n_classes]
        out_bbox = outputs["bbox"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["bbox"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(cxcywh_to_xyxy(out_bbox), cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["bbox"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DETRLoss(nn.Module):
    """ Computes the loss for DETR
    The process happens in two steps:
        1) compute hungarian assignment between ground truth boxes and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, n_classes, mask=False, aux_loss=False, matcher='hungarian', matcher_args=None, eos_coef=0.1,
                 weight_ce=1., weight_bbox=5., weight_giou=2., weight_dice=1., weight_mask=1, num_decoder_layers=None):
        """ Create the criterion.
        Parameters:
            n_classes: number of object categories, omitting the special no-object category
            mask: whether to include mask loss, i.e. for instance segmentation
            aux_loss: whether to also compute for auxilary loss
            matcher: matcher name to compute a matching between targets and proposals, available: 'hungairan'
            eos_coef: relative classification weight applied to the no-object category
            weight_ce: cross entropy loss weight
            weight_bbox: bounding box loss weight
            weight_giou: generalized iou loss weight
            weight_dice: dice loss weight, only applies if mask is True
            weight_mask: mask loss weight, only applies if mask is True
            num_decoder_layers: number of decoder layer on the model, used if aux_loss is True
        """
        super().__init__()
        self.n_classes = n_classes
        self.eos_coef = eos_coef

        if matcher_args is None:
            matcher_args = {}
        if matcher == 'hungarian':
            self.matcher = HungarianMatcher(**matcher_args)
        else:
            raise RuntimeError("Unknown matcher '{}', available ['hungarian']")

        self.weight_dict = {
            'loss_ce': weight_ce,
            'loss_bbox': weight_bbox,
            'loss_giou': weight_giou
        }
        self.losses = ['labels', 'bbox', 'cardinality']
        if mask:
            self.weight_dict.update({
                'loss_mask': weight_mask,
                'loss_dice': weight_dice
            })
            self.losses.append('masks')

        self.aux_loss = aux_loss
        if self.aux_loss:
            if num_decoder_layers is None:
                num_decoder_layers = 6
                warnings.warn("'num_decoder_layers' is not specified. Using default value of 6")
            aux_weight = {k+'_'+str(i): v for k,v in self.weight_dict.items() for i in range(num_decoder_layers-1)}
            self.weight_dict.update(aux_weight)

        empty_weight = torch.ones(self.n_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self._losses_value = None

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits' in outputs
        src_logits = outputs['logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.n_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        logits = outputs['logits']
        device = logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "bbox" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'bbox' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox'][idx]
        target_boxes = torch.cat([t['bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(src_boxes),
            cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_batch_tensor([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_fn_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'bbox': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_fn_map, "unknown loss '{}', available [{}]".format(loss, ', '.join(loss_fn_map.keys()))
        return loss_fn_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, input, target):
        """ This performs the loss computation.
        Parameters:
             input: dict of tensors, see the output specification of the model for the format
             target: list of dicts, such that len(target) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs = input     # to comply with vortex format
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, target)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in target)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, target, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.aux_loss:
            assert 'aux' in outputs, "'aux_loss' is True, but 'aux' member is not found in output, "\
                "available [{}]".format(','.join(outputs.keys()))
            for i, aux_outputs in enumerate(outputs['aux']):
                indices = self.matcher(aux_outputs, target)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, target, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self._losses_value = losses
        return sum(v * self.weight_dict[k] for k,v in losses.items() if k in self.weight_dict)

    def get_losses(self):
        return self._losses_value


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (torch.Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> torch.Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def dice_loss(inputs, targets, num_boxes):
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
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def cxcywh_to_xyxy(bbox):
    x_c, y_c, w, h = bbox.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def xyxy_to_cxcywh(bbox):
    x0, y0, x1, y1 = bbox.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = torchvision.ops.boxes.box_area(boxes1)
    area2 = torchvision.ops.boxes.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class DETRCollate:
    def __init__(self, dataformat: dict):
        dataformat = {k: {n: torch.tensor(v, dtype=torch.long) if n == 'indices' else v for n,v in val.items()}
            if val else None for k,val in dataformat.items()}
        self.dataformat = EasyDict(dataformat)
        self.disable_image_auto_pad = True

    def __call__(self, batch):
        images, targets = list(zip(*batch))
        images = NestedTensor.from_batch_tensor(images)

        collated_targets = []
        for target in targets:
            out_target = {}
            if target.ndim != 2:
                raise RuntimeError("expects dimensionality of target is 2 got %s" % target.ndim)
            if not "class_label" in self.dataformat or self.dataformat.class_label is None:
                out_target['labels'] = torch.zeros(target.shape[0], dtype=torch.int64)
            else:
                out_target['labels'] = torch.index_select(
                    input=target,
                    dim=self.dataformat.class_label.axis,
                    index=self.dataformat.class_label.indices
                ).flatten().type(torch.int64)
            out_target['bbox'] = torch.index_select(
                input=target,
                dim=self.dataformat.bounding_box.axis,
                index=self.dataformat.bounding_box.indices
            ).float()
            out_target['bbox'][:, :2] += out_target['bbox'][:, 2:] / 2  ## x,y,w,h -> cx,cy,w,h
            collated_targets.append(out_target)
        return images, collated_targets


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be one of [relu, gelu, glu], not {activation}.")


def create_model_components(preprocess_args: EasyDict, network_args: EasyDict, loss_args: EasyDict, postprocess_args: EasyDict) -> EasyDict:
    network_args['aux_loss'] = False if not 'aux_loss' in network_args else network_args['aux_loss']
    loss_args['n_classes'] = network_args['n_classes']
    loss_args['aux_loss'] = network_args['aux_loss']
    if 'num_decoder_layers' in network_args:
        loss_args['num_decoder_layers'] = network_args['num_decoder_layers']

    model = DETR(**network_args)
    optim_params = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": model.lr_backbone,
        },
    ]
    model_components = {
        'network': model,
        'loss': DETRLoss(**loss_args),
        'collate_fn': 'DETRCollate',
        'postprocess': DETRPostProcess(**postprocess_args),
        'param_groups': optim_params
    }
    return EasyDict(model_components)

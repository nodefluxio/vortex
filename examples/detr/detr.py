"""
Training Object Detection Example
=================================
This example will shows you how to train DETR model using pytorch lightning
We will use DETR model from https://github.com/facebookresearch/detr.
"""

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vision

import pytorch_lightning as pl

from collections import OrderedDict

from vortex.development.utils.registry import Registry
from vortex.development.networks.models import ModelBase
from vortex.development.utils.profiler.lightning import Profiler
from abc import abstractmethod

from easydict import EasyDict
from copy import copy
from pathlib import Path

from module.model import build_model
from module.dataset import build_dataset, BetterCOCO, COCODetection, VOCDetection, CocoEvaluator
from module.utils import collate_fn


# %%
# 1. Preparing Dataset
# --------------------
# We'll use pytorch lightning's DataModule interface to define our data module.
# Also we will use yaml to store hyperparameter and training config.

class DataLoader(pl.LightningDataModule):
    def __init__(self, args: dict):
        super().__init__()
        # assume args has the following structure
        # module: '...', args: {...}
        self.args = args['dataset']
        self.loader_args = args['dataloader']
        self.prepare_data()
    
    def _init_train_set(self):
        # support VOC and COCO dataset
        args = self.args
        args['args'].update({'image_set': 'train'})
        self.train_dataset = build_dataset(**args)
        # VOC has 20 class and COCO actually has 80 class,
        # but from DETR model, it is defined as 91 class with some class defined as N/A.
        if isinstance(self.train_dataset,VOCDetection):
            self.num_classes = 20
        elif isinstance(self.train_dataset,COCODetection):
            self.num_classes = 91
    
    def _init_val_set(self):
        args = self.args
        # validate on train for now
        args['args'].update({'image_set': 'train'})
        self.val_dataset = build_dataset(**args)
        if isinstance(self.val_dataset, VOCDetection):
            # we customize VOC dataset to be able to use COCO validator
            # here we convert our dataset to COCO format for validation and
            # load as coco dataset. Note that BetterCOCO dataset is derived from COCO
            # with additional method to easily visualize image and label using mapping style
            filename = Path('tmp/coco_fmt.json')
            filename.parent.mkdir(exist_ok=True,parents=True)
            self.val_dataset.to_coco(filename=filename)
            self.val_coco_fmt = BetterCOCO(filename)
        else: # assume COCO
            filename = args['args']['ann_file']
            root = args['args']['img_folder']
            self.val_coco_fmt = BetterCOCO(filename,root=root)

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self._init_train_set()
        if not hasattr(self, 'val_dataset'):
            self._init_val_set()
    
    def train_dataloader(self):
        kwargs = self.loader_args
        kwargs.update({'collate_fn': collate_fn})
        return torch.utils.data.DataLoader(self.train_dataset, **kwargs)
    
    def val_dataloader(self):
        kwargs = copy(self.loader_args)
        kwargs.update(dict(
            collate_fn=collate_fn,
            shuffle=False
        ))
        return torch.utils.data.DataLoader(self.val_dataset, **kwargs)

import os, sys

# Helper contex manager to suppress python print to std out.
# This will be used to disable COCO evaluator print.
class SuppressedPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# %%
# 2. Model Definition
# -------------------
# Here we define our model, training and validation step, as well as what value to be logged.
# We use vortex ``ModelBase`` which is derived from pytorch lightning module, with additional
# functionality such as available_metrics, output_format, predict, those extensions
# are mostly used for exporting, but we'll skip for now.

class DETR(ModelBase):
    def __init__(self, args: dict, gt=None):
        super().__init__()
        args.setdefault('device', self.device)
        args.setdefault('num_classes', 20)
        args = EasyDict(args)
        self.args = args
        model, criterion, postprocessors = build_model(args)
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.init_evaluator(gt)
    
    def init_evaluator(self, gt=None):
        # gt can be COCO object from pycocotools or None (can be specified later)
        # here we use CocoEvaluator from pycocotools to validate the model.
        # This method may need to be recalled again after validation to reset the state
        # of the validator
        self.gt = gt
        self.coco_evaluator = CocoEvaluator(self.gt, ['bbox']) \
            if self.gt else None

    def available_metrics(self):
        # skip for now
        return None

    def output_format(self):
        # skip for now
        return {}

    def predict(self, *args, **kwargs):
        # simply do forward for now
        return self.model(*args,**kwargs)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_dict = self.criterion(y_hat, y)
        # weight for each loss
        weight_dict = self.criterion.weight_dict
        # total loss
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # log totall loss
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, targets = batch
        outputs = self(x)
        # output is dictionary with 'pred_logits', 'pred_boxes', 'aux_outputs' fields
        # with:
        # - pred_logits' shape: NxDx(C+1) where C is number of classes
        # - pred_boxes' shape: NxDx4
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log('val_loss', loss.detach(), logger=True)
        if self.coco_evaluator is None:
            return
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        # can't be empty for now, skip if empty
        if len(results) == 0:
            return
        res = {int(target['image_id']): output for target, output in zip(targets, results)}
        self.coco_evaluator.update(res)
    
    def validation_epoch_end(self, validation_step_outputs):
        coco_evaluator = self.coco_evaluator
        if coco_evaluator is not None:
            # cant compute on empty results
            if len(coco_evaluator.eval_imgs['bbox']):
                with SuppressedPrints():
                    coco_evaluator.synchronize_between_processes()
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()
                res = coco_evaluator.coco_eval['bbox'].stats.tolist()
                params = coco_evaluator.coco_eval['bbox'].params
                iou_low, iou_high = params.iouThrs[0], params.iouThrs[-1]
                max_det_l, max_det_m, max_det_h = params.maxDets
                legends = [
                    f'(AP) @[ IoU={iou_low}:{iou_high} | area=   all | maxDets={max_det_h} ]',
                    f'(AP) @[ IoU=0.50      | area=   all | maxDets={max_det_h} ]',
                    f'(AP) @[ IoU=0.75      | area=   all | maxDets={max_det_h} ]',
                    f'(AP) @[ IoU={iou_low}:{iou_high} | area= small | maxDets={max_det_h} ]',
                    f'(AP) @[ IoU={iou_low}:{iou_high} | area=medium | maxDets={max_det_h} ]',
                    f'(AP) @[ IoU={iou_low}:{iou_high} | area= large | maxDets={max_det_h} ]',
                    f'(AR) @[ IoU={iou_low}:{iou_high} | area=   all | maxDets={max_det_l} ]',
                    f'(AR) @[ IoU={iou_low}:{iou_high} | area=   all | maxDets={max_det_m} ]',
                    f'(AR) @[ IoU={iou_low}:{iou_high} | area= small | maxDets={max_det_h} ]',
                    f'(AR) @[ IoU={iou_low}:{iou_high} | area=medium | maxDets={max_det_h} ]',
                    f'(AR) @[ IoU={iou_low}:{iou_high} | area= large | maxDets={max_det_h} ]',
                ]
                self.log_dict(dict(zip(legends,res)), on_epoch=True, prog_bar=False, logger=True)
                mAP, mAP50, mAP75 = res[:3]
            # for empty results, just report zero for now
            else:
                mAP, mAP50, mAP75 = 0, 0, 0
            self.log('mean AP', mAP, on_epoch=True, logger=True, prog_bar=True)
            self.log('mean AP (50)', mAP50, on_epoch=True, logger=True, prog_bar=True)
            self.log('mean AP (75)', mAP75, on_epoch=True, logger=True, prog_bar=True)
        self.init_evaluator(self.gt) # restart evaluator for next epoch
    
    def configure_optimizers(self):
        args = self.args
        model_without_ddp = self.model
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

# Helper function to create pytorch lighnting's trainer.
# Accepts dictionary containing trainer configuration and return Trainer instance.
def build_trainer(args: dict):
    args = EasyDict(args['trainer'])
    loggers = [
        pl.loggers.TensorBoardLogger('logs/'),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, gpus=args.gpus, logger=loggers,
        check_val_every_n_epoch=args.check_val_every_n_epoch
    )
    return trainer

def load_state_dict(backbone):
    if backbone=='resnet50':
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu', check_hash=True)
        return state_dict

def main(args: dict):
    dataset = DataLoader(args)
    args.update({'num_classes': dataset.num_classes})
    model      = DETR(args)
    # let's start from pretrained to save time
    # here we will start from fully pretrained model if we are training on COCO,
    # for VOC, we'll take the pretrained coco but class embedding layers.
    state_dict = load_state_dict(args['backbone'])
    if state_dict is not None:
        if dataset.num_classes==91:
            # we have coco here, can safely load all from state dict
            model.model.load_state_dict(state_dict['model'],strict=True)
        else:
            # reject trained class embedding, return to random
            filtered = list(filter(lambda x: 'class_embed' not in x, state_dict['model'].keys()))
            state_dict = {key: state_dict['model'][key] for key in filtered}
            model.model.load_state_dict(state_dict,strict=False)
    trainer = build_trainer(args)
    # pass gt from dataset to model's evaluator
    val_gt = dataset.val_coco_fmt
    model.init_evaluator(val_gt)
    # do some training
    trainer.fit(model, dataset)

if __name__=="__main__":
    import argparse
    import yaml
    if hasattr(__builtins__,'__IPYTHON__'):
        # running on notebook, for demonstration purpose
        config_file = 'config.yml'
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',default='config.yml')
        args = parser.parse_args()
        config_file = args.config
    with open(config_file) as f:
        args = yaml.load(f)
    main(args)
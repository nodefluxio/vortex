import torch
import warnings
import pytorch_lightning as pl

from .registry import register_model
from .model import ModelBase
from .base_connector import BackbonePoolConnector
from ..modules.losses.classification import ClassificationLoss
from ..modules.postprocess.base_postprocess import SoftmaxPostProcess
from ..modules.preprocess import get_preprocess
from ..modules.backbones import get_backbone

from easydict import EasyDict

all_models = ['softmax']
supported_models = ['softmax']

class Classification(BackbonePoolConnector):
    def __init__(self, model_name: str, backbone: dict, n_classes: int, *args, **kwargs):
        super(Classification, self).__init__(backbone, feature_type="classifier", 
            n_classes=n_classes, *args, **kwargs)

        # self.head = get_head(model_name, self.backbone.out_channels, n_classes)
        self.task = "classification"
        self.output_format = {
            "class_label": {"indices": [0], "axis": 0},
            "class_confidence": {"indices": [1], "axis": 0}
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        # x = self.head(x)
        return x


def create_model_components(model_name: str, preprocess_args: EasyDict, network_args: EasyDict, 
        loss_args: EasyDict, postprocess_args: EasyDict, stage: str):
    """ build model components for classification task

    This module will return model components for different stages.

    loss function will use `nn.NLLLoss`.
    
    Args:
        config (EasyDict): model config from config file
    
    Raises:
        KeyError: head model is not available
    
    Returns:
        EasyDict: model components
    """
    if not isinstance(model_name, str):
        raise TypeError("expects string got %s" % type(model_name))
    if not isinstance(preprocess_args, dict):
        raise TypeError("expects `preprocess_args` to have type dict, got %s" % type(preprocess_args))
    if not isinstance(network_args, dict):
        raise TypeError("expects `network_args` to have type dict, got %s" % type(network_args))
    if not isinstance(loss_args, dict):
        raise TypeError("expects `loss_args` to have type dict, got %s" % type(loss_args))
    if not isinstance(postprocess_args, dict):
        raise TypeError("expects `postprocess_args` to have type dict, got %s" % type(postprocess_args))
    if not model_name.lower() in all_models:
        raise KeyError("model %s not supported, available : %s" % (model_name, all_models))

    if "input_normalization" in preprocess_args:
        preprocess_args = preprocess_args.input_normalization
        
    components = {
        'preprocess': get_preprocess('normalizer', **preprocess_args),
        'network': Classification(model_name.lower(), **network_args),
        'postprocess': SoftmaxPostProcess(dim=1)
    }
    if stage == 'train':
        if 'additional_args' in loss_args:
            loss_args.update(loss_args.additional_args)
            loss_args.pop('additional_args')
        components['loss'] = ClassificationLoss(**loss_args)
        components['collate_fn'] = None
    elif stage == 'validate':
        pass
    else:
        raise NotImplementedError("stage other than 'train' and 'export' is not yet "\
            "implemented in classification, got stage: %s" % stage)
    
    return EasyDict(components)


class ClassificationModel(ModelBase):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        warnings.filterwarnings("ignore")
        self.metrics = {
            "accuracy": pl.metrics.Accuracy(),
            "f1_weighted": pl.metrics.F1(num_classes=num_classes, average="weighted"),
            "pr_curve": pl.metrics.PrecisionRecallCurve(num_classes=num_classes)
        }
        warnings.resetwarnings()
        for avg in ("micro", "macro"):
            self.metrics.update({
                "f1_"+avg: pl.metrics.F1(num_classes=num_classes, average=avg),
                "precision_"+avg: pl.metrics.Precision(num_classes=num_classes, average=avg),
                "recall_"+avg: pl.metrics.Recall(num_classes=num_classes, average=avg)
            })

        self._avail_metrics = {m: "max" for m in self.metrics if m != "pr_curve"}
        self._avail_metrics["train_loss"] = "min"
        self._avail_metrics["val_loss"] = "min"


    @property
    def output_format(self):
        return {
            "class_label": {"indices": [0], "axis": 0},
            "class_confidence": {"indices": [1], "axis": 0}
        }

    @property
    def available_metrics(self):
        return self._avail_metrics

    def training_step_end(self, outputs):
        self.log('train_loss', outputs['loss'], on_epoch=True, on_step=False, logger=True)
        return outputs

    def validation_step_end(self, outputs, test=False):
        assert all(x in outputs for x in ("pred", "target")), "validation_step is expected to return " \
            "'pred' and 'target' key to calculate metrics, got {}".format(list(outputs.keys()))

        if not test and 'loss' in outputs:
            self.log('val_loss', outputs['loss'], on_epoch=True, on_step=False, logger=True, prog_bar=True)
        for name, op in self.metrics.items():
            # prog_bar = True if name == "accuracy" or "micro" in name else False
            self.log(
                name, 
                op(outputs["pred"].detach().cpu(), outputs["target"].detach().cpu()), 
                on_epoch=(True if name != "pr_curve" else False),
                prog_bar=(True if name == "accuracy" else False),
                logger=(False if name == "pr_curve" else True)
            )

    def validation_epoch_end(self, outputs):
        self.log("pr_curve", self.metrics["pr_curve"].compute(), logger=False)

    def test_step_end(self, outputs):
        if 'loss' in outputs:
            self.log('test_loss', outputs['loss'], on_epoch=True, prog_bar=True)
        self.validation_step_end(outputs, test=True)


@register_model()
class Softmax(ClassificationModel):
    def __init__(self, network_args: dict, preprocess_args: dict, postprocess_args: dict, loss_args: dict):
        num_classes = network_args['n_classes']
        super().__init__(num_classes)

        self.backbone = get_backbone(
            network_args['backbone'],
            n_classes=num_classes,
            pretrained=network_args.get('pretrained_backbone', True),
            feature_type="classifier"
        )

        self.preprocess = get_preprocess('normalizer', **preprocess_args)
        self.postproces = SoftmaxPostProcess(**postprocess_args)
        self.criterion = ClassificationLoss(**loss_args)

    def forward(self, x):
        x = self.backbone(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.postproces(x)
        return x

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = self.criterion(pred, target)
        return {
            'loss': loss,
            'pred': pred,
            'target': target
        }

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = self.criterion(pred, target)
        return {
            'loss': loss,
            'pred': pred,
            'target': target
        }

    def test_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = self.criterion(pred, target)
        return {
            'loss': loss,
            'pred': pred,
            'target': target
        }


## backward compatibility for name 'softmax'
register_model('softmax', Softmax)

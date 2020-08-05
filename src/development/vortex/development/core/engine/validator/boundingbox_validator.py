import torch
import logging
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from easydict import EasyDict
from collections import OrderedDict
from collections.abc import Sequence
from functools import singledispatch
from typing import Union, List, Dict, Type, Any

from vortex.development.predictor.base_module import BasePredictor, create_predictor
from vortex.development.utils.metrics.evaluator import DetectionEvaluator as Evaluator
from vortex.development.utils.prediction import BoundingBox

from vortex.development.utils.profiler.speed import TimeData
from vortex.development.utils.profiler.resource import CPUMonitor, GPUMonitor
from vortex.development.core.factory import create_runtime_model
from vortex.development.predictor.base_module import BasePredictor, create_predictor
from vortex.development.predictor.utils import get_prediction_results

from .base_validator import BaseValidator
from vortex.runtime.basic_runtime import BaseRuntime

class BoundingBoxValidator(BaseValidator):
    __output_format__ = ['bounding_box', 'class_label', 'class_confidence']
    ## model output format requirements
    def __init__(self, 
                 predictor: Union[BasePredictor,BaseRuntime], 
                 dataset, 
                 metric_type: str='voc', 
                 *args, **kwargs):
        
        super(BoundingBoxValidator, self).__init__(predictor, dataset, *args, **kwargs)

        img, lbl = dataset[0]
        if not (isinstance(img, torch.Tensor) or isinstance(img, np.ndarray)):
            raise RuntimeError("expects dataset to return `image` of type np.ndarray or " \
                "torch.Tensor, got %s" % type(img))
        if not (isinstance(lbl, torch.Tensor) or isinstance(lbl, np.ndarray)):
            raise RuntimeError("expects dataset to return `label` of type np.ndarray or " \
                "torch.Tensor, got %s" % type(lbl))

        # self.score_threshold = np.array([score_threshold], dtype=np.float32)
        # self.iou_threshold = np.array([iou_threshold], dtype=np.float32)
        for key in self.prediction_args:
            if isinstance(self.prediction_args[key],float) or isinstance(self.prediction_args[key],int):
                self.prediction_args[key] = np.array([self.prediction_args[key]], dtype=np.float32)
        self.metric_type = metric_type
        assert metric_type in ['voc'], "unsupported metric type : {}, available 'voc'".format(metric_type)

        self.evaluator = None
    
    def validation_args(self) -> Dict[str,Any] :
        """
        reports validation args used for this run, 
        override if additional validation args exists
        e.g. score_threshold, iou_threshold etc.
        """
        args = super(type(self),self).validation_args()
        args.update(dict(
            metric_type=self.metric_type
        ))
        for key in self.prediction_args:
            args.update(dict(
                key=self.prediction_args[key].item()
            ))
        return args
    
    def eval_init(self, *args, **kwargs):
        self.evaluator = Evaluator()

    def predict(self, image, *args, **kwargs):
        results = super(type(self),self).predict(
            image=image, **self.prediction_args
        )
        return results
    
    def update_results(self, index, results, targets, last_index):
        ## TODO : unify shape for single batch & multiple batch loader
        if self.batch_size == 1 :
            targets = [targets]

        ## Using the following assertion raise error on the end of the data loader when
        ## len(dataloader[-1]) % self.batch_size != 0
        # assert self.batch_size == len(results) == len(targets)

        # Fix
        if not last_index:
            assert self.batch_size == len(results) == len(targets)

        for i, (result, target) in enumerate(zip(results,targets)) :
            i = index * self.batch_size + i
            self._update_results(i, [result], target)

    def _update_results(self, index, results, targets):
        results = results[0] # single batch

        if results['class_label'].max() >= len(self.class_names):
            self.class_names.append('N/A') ## background

        self.logger('labels :')
        label_bboxes = []
        bboxes = np.take(targets, self.labels_fmt.bounding_box.indices,
                            axis=self.labels_fmt.bounding_box.axis)
        class_labels = np.array([[0]]*bboxes.shape[1], dtype=np.int32)
        assert hasattr(self.labels_fmt, 'class_label')
        if not self.labels_fmt.class_label is None :
            class_labels = np.take(targets, self.labels_fmt.class_label.indices, 
                axis=self.labels_fmt.class_label.axis)

        class_labels_to_str = lambda class_label : self.class_names[class_label] \
            if self.class_names is not None else 'class_{}'.format(class_label)

        create_bbox = lambda gt : dict(x=gt[0],y=gt[1],w=gt[2],h=gt[3])
        to_xywh = lambda bbox : (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

        label_bboxes = list(map(lambda gt : BoundingBox(
            **create_bbox(gt[0]), img_name='{}'.format(index),
            class_label=class_labels_to_str(int(gt[1])), 
        ), zip(bboxes, class_labels)))

        for bbox in label_bboxes :
            self.logger(bbox)

        self.logger('detections :')
        result_bboxes = []

        to_bbox = lambda bounding_box, class_label, class_confidence, idx : BoundingBox(
            **create_bbox(to_xywh(bounding_box)),
            class_label=class_labels_to_str(int(class_label)), 
            img_name='{}'.format(index),
            confidence=float(class_confidence),
        )

        result_bboxes = []
        ## onlty perform mapping if has detections
        if results['bounding_box'] is not None :
            result_bboxes = map(lambda result : to_bbox(*result, index), 
                zip(results['bounding_box'], results['class_label'], results['class_confidence'])
            )
            result_bboxes = list(result_bboxes)

        for result in result_bboxes :
            self.logger(result)

        self.evaluator.update(result_bboxes, label_bboxes)
    
    def compute_metrics(self) :
        eval_results = self.evaluator.evaluate(iou_threshold=0.5)
        self.logger(eval_results)
        self.pr_curves = eval_results[0]
        return {
            # 'pr_curves': eval_results[0],
            'mean_ap': eval_results[1],
        }
    
    def save_metrics(self, output_directory) :
        if output_directory is not None:
            self.output_directory = output_directory
        filename = self.output_directory / '{}_{}.png'.format(self.experiment_name, self.predictor_name)
        plt.clf()
        plt.cla()
        ax = plt.gca()
        lines = []
        labels = []
        for i, result in enumerate(self.pr_curves):
            precision = result['precision']
            recall = result['recall']
            l, = ax.plot(recall, precision)
            label = '{} (ap :{:.3f})'.format(
                list(self.evaluator.bboxes.class_map.values())[i], result['ap'])
            lines.append(l)
            labels.append(label)
        ax.legend()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title("Precision Recall Curve")
        plt.grid()
        plt.legend(lines, labels, loc='center left',
                    prop=dict(size=8), bbox_to_anchor=(1., 0.5))
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(filename, aspect='auto')
        return {
            'Precision-Recall Curve' : filename
        }
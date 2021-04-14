from .metric_base import MetricBase
from .registry import METRICS
from pathlib import Path
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from itertools import cycle
from typing import Union, List, Dict, Type, Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

@METRICS.register()
class ClassificationMetrics(MetricBase):
    def __init__(self, class_names=None, *args, **kwargs):
        super().__init__()
        self.eval_init()
        self.class_names = class_names

    def eval_init(self, *args, **kwargs) :
        self.correct = 0
        self.results = {
            'truths' : {}, 
            'predictions' : {}, 
            'scores' : {},
        } ## {str : {int : []}}
        self.n_data = 0

    def update(self, inputs, targets):
        if isinstance(inputs, torch.Tensor):
            # assume has [batch_index, 2] shape
            result_class_label = inputs[:,0].flatten().numpy()
            result_class_confidence = inputs[:,1].flatten().numpy()
        else:
            result_class_label = list(map(lambda x: int(x['class_label']), inputs))
            result_class_confidence = list(map(lambda x: float(x['class_confidence']), inputs))
        label = targets
        if isinstance(label, np.ndarray):
            label = label.flatten()
        elif isinstance(label, torch.Tensor):
            label = label.flatten().cpu().numpy()
        correct = np.sum(np.equal(result_class_label, label))
        self.correct += correct
        self.n_data  += len(targets)
        for batch_idx, (gt, result_class, result_confidence) in enumerate(zip(label, result_class_label, result_class_confidence)) :
            gt = int(gt)
            if gt in self.results['truths']:
                self.results['truths'][gt].append(gt)
                self.results['predictions'][gt].append(result_class)
                self.results['scores'][gt].append(result_confidence)
            else:
                self.results['truths'][gt] = [gt]
                self.results['predictions'][gt] = [result_class]
                self.results['scores'][gt] = [result_confidence]
    
    def compute(self):
        metrics = {'accuracy': self.correct / self.n_data}

        num_classes = len(self.results['truths'])
        y_true, y_pred, score = [], [], []
        for i in range(num_classes):
            y_true.extend(self.results['truths'][i])
            y_pred.extend(self.results['predictions'][i])
            score.extend(self.results['scores'][i])

        for avg in ('micro', 'macro', 'weighted'):
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
            metrics.update({
                "precision ({})".format(avg): p,
                "recall ({})".format(avg): r,
                "f1_score ({})".format(avg): f1
            })
        return metrics

    def save_metrics(self, output_directory, experiment_name='classification_results') :
        results = pd.DataFrame(self.results)
        n_classes = len(results['truths'])

        ## compute confusion matrix
        y_true_flat, y_pred_flat = [], []
        for i in range(n_classes) :
            y_true_flat.extend(results['truths'][i])
            y_pred_flat.extend(results['predictions'][i])
        cm = confusion_matrix(
            y_true=y_true_flat,
            y_pred=y_pred_flat,
        )
        cm = cm / cm.sum(axis=1, keepdims=True)
        df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))

        to_list = lambda mapping : list(mapping[i] for i in range(len(mapping)))
        truths = np.array(to_list(results['truths']))
        predictions = np.array(to_list(results['predictions']))
        scores = np.array(to_list(results['scores']))

        ## truths, predictions, scores are mappings 
        ## truths : class_label -> class_label, 
        ## predictions : class_label -> prediction, 
        ## scores : class_label -> score,
        ## each size is (n_classes, n_samples)
        average_precisions, precisions, recalls = [], [], []
        roc_aucs, fprs, tprs = [], [], []
        for class_truths, class_predictions, class_scores in zip(truths, predictions, scores):
            n_samples = len(class_predictions)
            scores_mat = np.zeros((n_samples, n_classes))
            truths_mat = np.zeros_like(scores_mat)
            ## one hot encoding, fill with scores and label
            scores_mat[np.arange(n_samples), class_predictions] = class_scores
            truths_mat[np.arange(n_samples), class_truths] = 1

            average_precisions.append(
                average_precision_score(truths_mat.flatten(), scores_mat.flatten())
            )
            precision, recall, _ = precision_recall_curve(
                truths_mat.flatten(),
                scores_mat.flatten(),
            )
            precisions.append(precision)
            recalls.append(recall)

            roc_aucs.append(roc_auc_score(truths_mat.flatten(), scores_mat.flatten()))
            fpr, tpr, _ = roc_curve(truths_mat.flatten(), scores_mat.flatten())
            fprs.append(fpr)
            tprs.append(tpr)

        assets = {}

        ## plot confusion matrix
        plt.clf()
        plt.cla()
        plt.gcf().set_size_inches((6.4,4.8))
        ax = plt.gca()
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(
            df_cm, annot=True, ax=ax,
            annot_kws={"size": 10} # font size
        )
        plt.autoscale()
        plt.tight_layout()
        filename = output_directory / '{}.png'.format(experiment_name)
        plt.savefig(filename)
        sn.reset_defaults()
        assets.update({
            'Confusion Matrix' : filename,
        })

        ## plot pr curve
        plt.clf()
        plt.cla()
        ax = plt.gca()
        plt.gcf().set_size_inches((6.4,4.8))
        lines, labels = [], []
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        for i, (precision, recall, ap, color) in enumerate(zip(precisions, recalls, average_precisions, colors)) :
            l, = ax.plot(recall, precision, color=color)
            class_name = 'class_{}'.format(i) if self.class_names is None else self.class_names[i]
            label = '{} (ap :{:.2f}'.format(class_name, ap)
            lines.append(l)
            labels.append(label)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title("Precision Recall Curve")
        plt.legend(lines, labels, loc='center left',
                    prop=dict(size=8), bbox_to_anchor=(1., 0.5))
        plt.autoscale()
        plt.tight_layout()
        filename = output_directory / '{}_pr_curve.png'.format(experiment_name)
        plt.savefig(filename)
        assets.update({
            'Precision Recall' : filename,
        })

        ## plot roc auc curve
        plt.clf()
        plt.cla()
        ax = plt.gca()
        plt.gcf().set_size_inches((6.4,4.8))
        lines, labels = [], []
        for i, (fpr, tpr, auc, color) in enumerate(zip(fprs, tprs, roc_aucs, colors)) :
            l, = ax.plot(fpr, tpr, color=color)
            class_name = 'class_{}'.format(i) if self.class_names is None else self.class_names[i]
            label = '{} (auc :{:.2f})'.format(class_name, auc)
            lines.append(l)
            labels.append(label)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(lines, labels, loc='center left',
                    prop=dict(size=8), bbox_to_anchor=(1., 0.5))
        plt.autoscale()
        plt.tight_layout()
        filename = output_directory / '{}_roc_curve.png'.format(experiment_name)
        plt.savefig(filename)
        assets.update({
            'ROC Curve' : filename,
        })

        return assets
    
    def report(self, output_directory='.', experiment_name='classification_results'):
        return self.save_metrics(output_directory,experiment_name)
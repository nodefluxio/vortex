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

from vortex.predictor.base_module import BasePredictor

from .base_validator import BaseValidator

class ClassificationValidator(BaseValidator):
    ## TODO : read from core task definition
    __output_format__ = ['class_label', 'class_confidence']
    def __init__(self, predictor, dataset, device = torch.device('cpu'), *args, **kwargs):
        super(ClassificationValidator, self).__init__(
            predictor=predictor, dataset=dataset, *args, **kwargs
        )
        self.results = None
        self.confusion_matrix = None
    
    def eval_init(self, *args, **kwargs) :
        self.correct = 0
        self.results = {
            'truths' : {}, 
            'predictions' : {}, 
            'scores' : {},
        } ## {str : {int : []}}
    
    def update_results(self, index : int, results : List[Dict[str,np.ndarray]], targets : Union[np.ndarray,torch.Tensor], last_index : bool) :
        label = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        if self.labels_fmt.class_label is not None:
            idx = self.labels_fmt.class_label.indices
            axs = self.labels_fmt.class_label.axis
            label = np.take(label, idx, axis=axs)
        result_class_label = list(map(lambda x: int(x['class_label']), results))
        result_class_confidence = list(map(lambda x: float(x['class_confidence']), results))
        if isinstance(label, np.ndarray) :
            label = label.flatten().tolist()
        self.logger('results : {}'.format(result_class_label))
        self.logger('targets : {}'.format(label))
        self.logger('scores  : {}'.format(result_class_confidence))
        correct = np.sum(np.equal(result_class_label, label))
        self.correct += correct
        for batch_idx, (gt, result_class, result_confidence) in enumerate(zip(label, result_class_label, result_class_confidence)) :
            gt = int(gt)
            if gt in self.results['truths'] :
                self.results['truths'][gt].append(gt)
                self.results['predictions'][gt].append(result_class)
                self.results['scores'][gt].append(result_confidence)
            else :
                self.results['truths'][gt] = [gt]
                self.results['predictions'][gt] = [result_class]
                self.results['scores'][gt] = [result_confidence]

    def compute_metrics(self):
        n_data = len(self.dataset.dataset) if isinstance(self.dataset, torch.utils.data.DataLoader) \
            else len(self.dataset)
        metrics = {'accuracy': self.correct / n_data}

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

    def save_metrics(self, output_directory) :
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
        filename = self.output_directory / '{}_{}.png'.format(self.experiment_name, self.predictor_name)
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
        filename = self.output_directory / '{}_{}_pr_curve.png'.format(self.experiment_name, self.predictor_name)
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
        filename = self.output_directory / '{}_{}_roc_curve.png'.format(self.experiment_name, self.predictor_name)
        plt.savefig(filename)
        assets.update({
            'ROC Curve' : filename,
        })

        return assets
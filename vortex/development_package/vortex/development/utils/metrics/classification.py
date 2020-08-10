import warnings
from itertools import chain, combinations

import numpy as np


def unique_labels(*labels, sort=False):
    lbls = set(chain.from_iterable(
        np.unique(np.asarray(y)) if hasattr(y, '__array__') else set(y) 
        for y in labels
    ))
    try:
        lbls = np.array(sorted(lbls), dtype=np.int)
    except:
        raise RuntimeError("Unknown data type caught in labels, "\
            "make sure all of your label data is integer value")
    if sort:
        lbls = np.sort(lbls)
    return lbls


def multilabel_confusion_matrix(label, pred, num_classes=None):
    """Compute a multilabel confusion matrix for each class

    In multilabel confusion matrix (MCM), each class is represented as 
    2x2 matrix with: `MCM(0,0)` is true negative (tn); `MCM(0, 1)` is 
    false positive (fp); `MCM(1, 0)` is false negative (fn); and 
    `MCM(1, 1)` is true positive (tp).

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        num_classes (int, optional): number of class for the entire 
            labels. Expect int or None (default), if None classes 
            will be determined from the unique value of labels and 
            prediction value. 
    Returns:
        multi_confusion (numpy.array): multilabel confusion matrix, 
            represented in 2x2 matrix corresponding to each classes. 
            If `num_classes` is None, `n_outputs` is the number of 
            unique value in `label` and `pred`, else `n_outputs` 
            equals to `num_classes`. The result is returned as an 
            ordered values of classes.
            Output shape (`n_outputs`, 2, 2)
    """
    label = np.array(label, dtype=np.int)
    pred = np.array(pred, dtype=np.int)
    if label.ndim != 1 or pred.ndim != 1:
        raise RuntimeError("dimension of 'label' ({})  and 'pred' ({}) "\
            "is expected to be 1".format(label.ndim, pred.ndim))
    uniq_lbl = unique_labels(label, pred)
    if num_classes is not None:
        num_uniq = len(uniq_lbl)
        if num_classes < num_uniq:
            raise RuntimeError("number of classes ({}) must be greater "\
                "than number of  unique classes in 'label' and 'pred' "\
                "({})".format(num_classes, num_uniq))
        assert isinstance(num_classes, int), "'num_classes' argument expect 'int'"
        uniq_lbl = np.arange(num_classes)

    tp = label == pred
    tp_bins = label[tp]
    if len(tp_bins):
        tp_sum = np.bincount(tp_bins, minlength=len(uniq_lbl))
    else: # Pathological case
        true_sum = pred_sum = tp_sum = np.zeros(len(uniq_lbl))
    if len(label):
        true_sum = np.bincount(label, minlength=len(uniq_lbl))
    if len(pred):
        pred_sum = np.bincount(pred, minlength=len(uniq_lbl))

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = label.shape[0] - tp - fp - fn
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def _prf_divide(numerator, denominator, metric, zero_division="warn"):
    """Performs division and handles divide-by-zero.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result

    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0
    if zero_division == 'warn':
        warnings.warn("Got zero division when calculating {}, set the "\
            "corresponding result to 0.0".format(metric), UserWarning)
    return result


def precision_recall_fscore(label, pred, beta=1.0, num_classes=None, 
                            average="micro", zero_division="warn"):
    """ Compute precision, recall, and F-score for each class.

    If `average` is None metrics is returned for each class as a numpy.array, 
    with shape of `n_outputs`; when `num_classes` is None `n_outputs` is
    determined from the number of unique classes in both `labels` and `pred`,
    else `n_outputs` is equals to `num_classes`.

    Read more in `sklearn.metrics.precision_recall_fscore_support 
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html>`

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        beta (float, optional): beta value for F-score, represents 
            the weight (strength) of recall vs precision. 
            Defaults to 1.0 (F1).
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels and prediction value. 
            Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'micro'`.
            available flag:
            `'micro'` : 
                Calculate metrics by averaging globally across classes.
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
            `None` :
                Return metrics for each class (not averaged).

        zero_division : value to set when divided by zero, one of "warn" (default), 
            0, or 1. If set to 'warn', this act like 0 but also throws warning.

    Returns:
        precision : float if average is not None, else numpy.array with shape (`n_outputs`)
        recall : float if average is not None, else numpy.array with shape (`n_outputs`)
        f_score : float if average is not None, else numpy.array with shape (`n_outputs`)
    """
    _avail_ave = (None, "micro", "macro", "weighted")
    _avail_zero = ("warn", 0, 1)
    if average not in _avail_ave:
        raise RuntimeError("Unknown 'average' value of ({}), available "\
            "[{}]".format(average, ', '.join(str(a) for a in _avail_ave)))
    if beta < 0:
        raise RuntimeError("'beta' should be >=0 in the F-beta score")
    if zero_division not in _avail_zero:
        raise RuntimeError("'zero_division' argument got {}, expect one "\
            "of [{}]".format(zero_division, ', '.join(_avail_zero)))

    mcm = multilabel_confusion_matrix(label, pred, num_classes=num_classes)
    tp_sum = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum = tp_sum + mcm[:, 1, 0]
    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _prf_divide(tp_sum, pred_sum, 'precision', zero_division)
    recall = _prf_divide(tp_sum, true_sum, 'recall', zero_division)

    beta2 = beta*beta
    if zero_division == "warn" and (pred_sum[true_sum == 0] == 0).any():
        warnings.warn("Got zero division when calculating F-score, set the "\
            "corresponding result to 0.0", UserWarning)
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = (beta2*precision) + recall
        denom[denom == 0.0] = 1
        f_score = (1 + beta2) * precision * recall / denom

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0)
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
    return precision, recall, f_score


def precision(label, pred, num_classes=None, average="micro"):
    """ Compute precision metric for each class.

    If `average` is None metrics is returned for each class as a numpy.array, 
    with shape of `n_outputs`; when `num_classes` is None `n_outputs` is
    determined from the number of unique classes in both `labels` and `pred`,
    else `n_outputs` is equals to `num_classes`.

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels and prediction value. 
            Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'micro'`.
            available flag:
            `'micro'` : 
                Calculate metrics by averaging globally across classes.
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
            `None` :
                Return metrics for each class (not averaged).

    Returns:
        precision : float if average is not None, else numpy.array with shape (`n_outputs`)
    """
    p, _, _ = precision_recall_fscore(label, pred, num_classes=num_classes,
                                      average=average)
    return p

def recall(label, pred, num_classes=None, average="micro"):
    """ Compute recall metric for each class.

    If `average` is None metrics is returned for each class as a numpy.array, 
    with shape of `n_outputs`; when `num_classes` is None `n_outputs` is
    determined from the number of unique classes in both `labels` and `pred`,
    else `n_outputs` is equals to `num_classes`.

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels and prediction value. 
            Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'micro'`.
            available flag:
            `'micro'` : 
                Calculate metrics by averaging globally across classes.
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
            `None` :
                Return metrics for each class (not averaged).

    Returns:
        recall : float if average is not None, else numpy.array with shape (`n_outputs`)
    """
    _, r, _ = precision_recall_fscore(label, pred, num_classes=num_classes,
                                      average=average)
    return r

def fbeta_score(label, pred, beta=1.0, num_classes=None, average="micro",
                zero_division="warn"):
    """ Compute F-score metric for each class.

    If `average` is None metrics is returned for each class as a numpy.array, 
    with shape of `n_outputs`; when `num_classes` is None `n_outputs` is
    determined from the number of unique classes in both `labels` and `pred`,
    else `n_outputs` is equals to `num_classes`.

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        beta (float, optional): beta value for F-score, represents 
            the weight (strength) of recall vs precision. 
            Defaults to 1.0 (F1).
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels and prediction value. 
            Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'micro'`.
            available flag:
            `'micro'` : 
                Calculate metrics by averaging globally across classes.
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
            `None` :
                Return metrics for each class (not averaged).

        zero_division : value to set when divided by zero, one of "warn" (default), 
            0, or 1. If set to 'warn', this act like 0 but also throws warning.

    Returns:
        f_score : float if average is not None, else numpy.array with shape (`n_outputs`)
    """
    _, _, f = precision_recall_fscore(label, pred, num_classes=num_classes,
                                      beta=beta, average=average)
    return f 

def f1_score(label, pred, num_classes=None, average="micro", zero_division="warn"):
    """ Compute F1-score metric for each class.

    If `average` is None metrics is returned for each class as a numpy.array, 
    with shape of `n_outputs`; when `num_classes` is None `n_outputs` is
    determined from the number of unique classes in both `labels` and `pred`,
    else `n_outputs` is equals to `num_classes`.

    Args:
        label : 1d array-like, ground-truth (target) value.
        pred : 1d array-like, predicted (estimated) value.
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels and prediction value. 
            Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'micro'`.
            available flag:
            `'micro'` : 
                Calculate metrics by averaging globally across classes.
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
            `None` :
                Return metrics for each class (not averaged).

        zero_division : value to set when divided by zero, one of "warn" (default), 
            0, or 1. If set to 'warn', this act like 0 but also throws warning.

    Returns:
        f_score : float if average is not None, else numpy.array with shape (`n_outputs`)
    """
    _, _, f = precision_recall_fscore(label, pred, num_classes=num_classes,
                                      beta=1.0, average=average)
    return f


def _check_uniq_num_cls(label, fn_name, num_classes=None):
    uniq_lbl = unique_labels(label)
    if num_classes is not None:
        num_uniq = len(uniq_lbl)
        if num_classes < num_uniq:
            raise RuntimeError("number of classes ({}) must be greater "
                "than number of unique classes found in 'label' ({})".format(
                num_classes, num_uniq))
        assert isinstance(num_classes, int), "'num_classes' argument expect 'int'"
        uniq_lbl = np.arange(num_classes)
    else:
        num_classes = len(uniq_lbl)
    if num_classes < 2:
        raise RuntimeError("'{}' does not support single class data.".format(fn_name))

    return uniq_lbl, num_classes


def label_binarize(y, num_classes=None, unique=None):
    """ Binarize label (from multiclass to binary) like one hot encoding.

    Return an array with shape of (n_data, n_class); where `n_data` is 
    total number of data in `y` (flattened). If `num_classes` is None,
    `n_class` is determined from labels in `y`, else `n_class` is equals
    to `num_classes` argument.

    Args:
        y (numpy.array): sequence of integer labels to encode
        num_classes (int, optional): [description]. Defaults to None.

    Returns:
        result (numpy.array): encoded label, shape (n_data, n_class)
    """
    y = np.asarray(y)
    uniq_lbl, lbl_num_cls = _check_uniq_num_cls(y, "label_binarize")
    num_data = lbl_num_cls
    ordered = True

    if lbl_num_cls == 2: ## binary
        if unique is not None or (num_classes is not None and num_classes != 2):
            raise RuntimeError("binary data does not support 'unique' or "
                "'num_classes' argument")
        return y.reshape(-1, 1)

    if unique is not None and num_classes is None:
        if len(np.unique(unique)) != len(unique):
            raise ValueError("Unique label {} is not unique".format(unique))
        uniq_lbl = np.asarray(unique, dtype=int)
        num_classes = len(uniq_lbl)
        if uniq_lbl.max()+1 != num_classes:
            ordered = False
            num_data = uniq_lbl.max() + 1
    elif num_classes is not None and unique is None:
        uniq_lbl = np.arange(num_classes)
    elif num_classes is None and unique is None:
        num_classes = lbl_num_cls
    else:
        raise RuntimeError("Choose either one of 'num_classes' or 'unique'")

    if num_classes < 2:
        raise RuntimeError("label_binarize does not support single class data.")

    y = y.ravel()
    if num_classes > lbl_num_cls and ordered:
        num_data = num_classes

    result = np.zeros((y.size, num_data), dtype=np.int)
    result[np.arange(y.size), y] = 1
    if not ordered:
        mask = np.searchsorted(np.arange(num_data), uniq_lbl)
        result = result[:, mask]
    elif num_classes < lbl_num_cls:
        result = result[:, :num_classes]
    return result


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def _binary_curve(label, score):
    """ Calculate true and false positive for binary classification 
    in each threshold.

    Args:
        label : 1-d array-like, ground truth label.
        score : 1-d array-like, prediction score (probability).

    Returns:
        fps (numpy.array) : false positive count
        tps (numpy.array) : true positive count
        threshold (numpy.array) : threshold value that achieve 'fps' and 'tps'
    """
    uniq_lbl = unique_labels(label)
    if len(uniq_lbl) != 2:
        raise ValueError("data is not binary, got {} unique labels".format(
                         len(uniq_lbl)))
    if label.shape != score.shape:
        raise ValueError("'label' and 'score' is expected to have the same shape")
    label = label.ravel()
    score = score.ravel()
    label = (label == 1)

    # sort score
    desc_score_idxs = np.argsort(score, kind="mergesort")[::-1]
    score = score[desc_score_idxs]
    label = label[desc_score_idxs]

    uniq_val_idx = np.where(np.diff(score))[0]
    threshold_idxs = np.r_[uniq_val_idx, label.size - 1]

    tps = stable_cumsum(label)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, score[threshold_idxs]


def pr_curve(label, score, classes=None):
    """ Compute Precision-Recall (PR) curve.

    For binary cases, return shape (1, N); and for multiclass cases,
    return shape (C, N) with N is the number of threshold value in the 
    curve, and C is number of classes. Number of classes is determined 
    from unique values found in labels, or the same as dim 2 of 'score'.

    Args:
        label : array-like, shape (n_sample,)
            Ground-truth (target) value of binary or multiclass cases.
        score : array-like
            Predicted (estimated) probability value. For binary case, 
            expected shape of (n_sample,); for multiclass case expected 
            shape of (n_sample, n_classes).
        classes (int, list, optional): 
            In multilabel cases, determine which classes to for the ROC
            Curve to be computed. If 'None', compute all available classes.

    Returns:
        precision (numpy.array) : precision score list, shape (1, N) for 
            binary, and (C, N) for multiclass.
        recall (numpy.array) : recall score list, shape (1, N) for binary, 
            and (C, N) for multiclass.
        threshold (numpy.array) : shape (1, N) for binary, and (C, N) for multiclass.
            Decreasing thresholds value used to compute fpr and tpr. The first value 
            on each class of 'thresholds' represents no instances being predicted and 
            is arbitrarily set to `max(score) + 1`.
    """
    label = np.array(label, dtype=np.int)
    score = np.array(score, dtype=np.float)

    _, num_classes = _check_uniq_num_cls(label, "roc_curve")
    is_binary = (score.ndim == 1 or (score.ndim == 2 and score.shape[1] == 1)
                 or num_classes == 2)

    if isinstance(classes, list):
        avail_cls = list(range(num_classes))
        if not all(c in avail_cls for c in classes):
            raise RuntimeError("Not all classes specified [{}] could "
                "be determined with num_class of {}".format(classes, num_classes))
    elif isinstance(classes, int):
        classes = [classes]
    elif classes is not None:
        raise RuntimeError("Unknown value of 'classes' {}, please provide "
            "an int or list".format(classes))

    if not is_binary:
        label = label_binarize(label, num_classes=num_classes)
    if classes is not None:
        if is_binary:
            raise RuntimeError("Cannot specify 'classes' for binary case.")
        label = label[:, classes]
        score = score[:, classes]
    if score.ndim == 1:
        score = score.reshape(-1, 1)
    if label.ndim == 1:
        label = label.reshape(-1, 1)

    precisions, recalls = [], []
    thresholds = []
    for i in range(score.shape[1]):
        fp, tp, th = _binary_curve(label[:, i], score[:, i])
        p = tp / (tp + fp)
        p[np.isnan(p)] = 0
        r = tp / tp[-1]

        last_idx = tp.searchsorted(tp[-1])
        sl = slice(last_idx, None, -1)
        precisions.append(np.r_[p[sl], 1])
        recalls.append(np.r_[r[sl], 0])
        thresholds.append(th[sl])

    if score.shape[1] == 1:
        precisions = precisions[0]
        recalls = recalls[0]
        thresholds = thresholds[0]
    return precisions, recalls, thresholds


def roc_curve(label, score, classes=None):
    """ Compute Receiver Operating Characteristic (ROC) curve.

    For binary cases, return shape (1, N); and for multiclass cases,
    return shape (C, N) with N is the number of threshold value in the 
    curve, and C is number of classes. Number of classes is determined 
    from unique values found in labels, or the same as dim 2 of 'score'.

    Args:
        label : array-like, shape (n_sample,)
            Ground-truth (target) value of binary or multiclass cases.
        score : array-like
            Predicted (estimated) probability value. For binary case, 
            expected shape of (n_sample,); for multiclass case expected 
            shape of (n_sample, n_classes).
        classes (int, list, optional): 
            In multilabel cases, determine which classes to for the ROC
            Curve to be computed. If 'None', compute all available classes.

    Returns:
        fpr (numpy.array) : false positive rate values, shape (1, N) for 
            binary, and (C, N) for multiclass.
        tpr (numpy.array) : true positive rate values, shape (1, N) for 
            binary, and (C, N) for multiclass.
        threshold (numpy.array) : shape (1, N) for binary, and (C, N) for multiclass.
            Decreasing thresholds value used to compute fpr and tpr. The first value 
            on each class of 'thresholds' represents no instances being predicted and 
            is arbitrarily set to `max(score) + 1`.
    """
    label = np.array(label, dtype=np.int)
    score = np.array(score, dtype=np.float)

    _, num_classes = _check_uniq_num_cls(label, "roc_curve")
    is_binary = (score.ndim == 1 or (score.ndim == 2 and score.shape[1] == 1)
                 or num_classes == 2)

    if isinstance(classes, list):
        avail_cls = list(range(num_classes))
        if not all(c in avail_cls for c in classes):
            raise RuntimeError("Not all classes specified [{}] could "
                "be determined with num_class of {}".format(classes, num_classes))
    elif isinstance(classes, int):
        classes = [classes]
    elif classes is not None:
        raise RuntimeError("Unknown value of 'classes' {}, please provide "
            "an int or list".format(classes))

    if not is_binary:
        label = label_binarize(label, num_classes=num_classes)
    if classes is not None:
        if is_binary:
            raise RuntimeError("Cannot specify 'classes' for binary case.")
        label = label[:, classes]
        score = score[:, classes]
    if score.ndim == 1:
        score = score.reshape(-1, 1)
    if label.ndim == 1:
        label = label.reshape(-1, 1)

    fpr, tpr = [], []
    thresholds = []
    for i in range(score.shape[1]):
        fp, tp, th = _binary_curve(label[:, i], score[:, i])
        
        if len(fp) > 2:
            optimal_idxs = np.where(np.r_[True,
                                        np.logical_or(np.diff(fp, 2), 
                                                        np.diff(tp, 2)),
                                        True])[0]
            fp = fp[optimal_idxs]
            tp = tp[optimal_idxs]
            th = th[optimal_idxs]

        ## add extra threshold to make sure starts at (0, 0)
        fp = np.r_[0, fp]
        tp = np.r_[0, tp]
        th = np.r_[th[0]+1, th]
        if fp[-1] <= 0:
            warnings.warn("No negative value in label, false positive "
                          "rate is meaningless")
            fp = np.repeat(np.nan, fp.shape)
        else:
            fp = fp / fp[-1]
        if tp[-1] <= 0:
            warnings.warn("No negative value in label, true positive "
                          "rate is meaningless")
            tp = np.repeat(np.nan, tp.shape)
        else:
            tp = tp / tp[-1]
        fpr.append(fp)
        tpr.append(tp)
        thresholds.append(th)

    if score.shape[1] == 1:
        fpr, tpr = fpr[0], tpr[0]
        thresholds = thresholds[0]
    return fpr, tpr, thresholds


def roc_auc_score(label, score, num_classes=None, average="macro", method="ovr", 
                  max_fpr=None):
    """ Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Args:
        label : array-like of shape (n_sample,) or (n_sample, n_classes)
            Ground-truth (target) value of binary or multiclass cases.
        score : 1d array-like, predicted (estimated) probability value.
        num_classes (int, optional): number of class for the entire 
            labels. If None classes will be determined from the unique 
            value of labels. Expect int or None (default).
        average (str, optional): flag for averaging behavior. Defaults to `'macro'`.
            Available flag:
            `'macro'` :
                Calculate metrics for each classes, and find their unweighted
                mean. This does not take label imbalance into account.
            `'weighted'` :
                Calculate metrics for each classes, and find their average weighted
                by support (the number of true instances for each label). This 
                alters 'macro' to account for label imbalance, which can result in
                an F-score that is not between precision and recall.
        method (str, optional): Determine algorithm to use on multiclass
            classification case. Defaults to `'ovr'`.
            Available:
            `'ovr'` :
                Computes the AUC of each class against the rest using one-vs-rest
                algorithm. Sensitive to class imbalance even when `average == 'macro'`,
                because class imbalance affects the composition of each of the
                'rest' groupings.
            `'ovo'` :
                Computes the AUC of each class against each one of other classes then find
                the average. Insensitive to class imbalance when `average == 'macro'`.
    Returns:
        auc (float) : the auc score.
    """
    label = np.array(label, dtype=np.int)
    score = np.array(score, dtype=np.float)

    classes, num_classes = _check_uniq_num_cls(label, "roc_auc_score",
        num_classes=num_classes)
    if num_classes > 2: ## multiclass
        if max_fpr is not None and max_fpr != 1.:
            raise RuntimeError("Partial AUC computation not available in "
                "multiclass setting, 'max_fpr' must be set to `None`, "
                "received `max_fpr={0}` instead".format(max_fpr))
        if method not in ("ovo", "ovr"):
            raise RuntimeError("Unknown 'multiclass' argument ({}), "
                "must be one of ('ovo', 'ovr')".format(method))
        if not np.allclose(1, score.sum(axis=1)):
            raise ValueError("Prediction score need to be probabilities "
                "for multiclass, each prediction sample should sum to 1 "
                "(result of Softmax function).")

        _avail_ave = ("macro", "weighted")
        if average not in _avail_ave:
            raise RuntimeError("Unknown 'average' value of ({}), available "\
                "[{}]".format(average, ', '.join(_avail_ave)))
        _avail_method = ("ovr", "ovo")
        if method not in _avail_method:
            raise RuntimeError("Unknown 'method' value of ({}), available "\
                "[{}]".format(method, ', '.join(_avail_method)))

        if num_classes != score.shape[1]:
            raise ValueError("Number of classes found in 'labels' or provided "
                "in arguments is not equal to number of columns in 'score'.")
        if label.shape[0] != score.shape[0]:
            raise ValueError("Number of samples (dim 0) in 'labels' ({}) and "
                "in 'score' ({}) is not equal.".format(label.shape[0], score.shape[0]))

        if method == "ovo":
            label_encoded = np.searchsorted(classes, label)
            num_pairs = num_classes * (num_classes-1) // 2
            pair_scores = np.empty(num_pairs)

            is_weighted = average == "weighted"
            prevelance = np.empty(num_pairs) if is_weighted else None
            for idx, (a,b) in enumerate(combinations(classes, 2)):
                a_mask = (label_encoded == a)
                b_mask = (label_encoded == b)
                ab_mask = np.logical_or(a_mask, b_mask)
                if is_weighted:
                    prevelance[idx] = np.average(ab_mask)

                a_true = a_mask[ab_mask]
                b_true = b_mask[ab_mask]
                a_true_score = _binary_roc_auc_score(a_true, score[ab_mask, a])
                b_true_score = _binary_roc_auc_score(b_true, score[ab_mask, b])
                pair_scores[idx] = (a_true_score + b_true_score) / 2
            return np.average(pair_scores, weights=prevelance)
        else:
            true_binary = label_binarize(label, unique=classes)

            average_weight = None
            if average == "weighted":
                average_weight = np.sum(true_binary, axis=0)
                if np.isclose(average_weight.sum(), 0.0):
                    return 0
            if true_binary.ndim == 1:
                true_binary = true_binary.reshape((-1, 1))
            if score.ndim == 1:
                score = score.reshape((-1, 1))

            values = np.zeros(num_classes)
            for c in range(num_classes):
                label_c = true_binary.take([c], axis=1).ravel()
                score_c = score.take([c], axis=1).ravel()
                values[c] = _binary_roc_auc_score(label_c, score_c)
            return np.average(values, weights=average_weight)
    else: ## binary
        label = label_binarize(label, num_classes=num_classes)[:, 0]
        return _binary_roc_auc_score(label, score, max_fpr=max_fpr)


def _binary_roc_auc_score(label, score, max_fpr=None):
    if len(np.unique(label)) != 2:
        raise ValueError("'label' is expected to have 2 unique value.")

    fpr, tpr, _ = roc_curve(label, score)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected 'max_fpr' in range (0, 1], got: {}".format(max_fpr))

    stop = np.searchsorted(fpr, max_fpr, 'right')
    x_interp = [fpr[stop-1], fpr[stop]]
    y_interp = [tpr[stop-1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    min_area = 0.5 * (max_fpr*max_fpr)
    max_area = max_fpr
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

def auc(x, y):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    if x.shape != y.shape:
        raise RuntimeError("x ({}) and y ({}) must have the same shape".format(
            x.shape, y.shape))
    if x.shape[0] < 2:
        raise ValueError("At least 2 points are required to compute AUC, got"
            "shape of {}".format(x.shape))

    direction = 1
    dx = np.diff(x)
    if np.all(dx <= 0):
        direction = -1
    elif np.any(dx < 0):
        raise ValueError("x is neither increasing nor decreasing {}".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

import numpy as np
import pytest
import numpy.testing as npt

from vortex.development.utils.metrics.classification import (
    multilabel_confusion_matrix,
    precision_recall_fscore,
    precision, recall,
    fbeta_score, f1_score,
    label_binarize,
    roc_curve, pr_curve,
    roc_auc_score, auc
)


def get_data(binary=False, score=False):
    if binary:
        x = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 
                      0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 
                      1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    else:
        x = np.array([2, 2, 0, 2, 1, 1, 0, 1, 2, 1, 2, 1, 1, 1, 1, 0, 2, 
                      2, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 0, 1, 0, 1, 1, 
                      0, 0, 0, 0, 2, 0, 1, 2, 0, 1, 0, 1, 1, 0, 0, 1])
    if score and binary:
        y = np.array([0.45694917, 0.29426178, 0.62909544, 0.52564127, 0.43930741,
                      0.40326766, 0.63564666, 0.7078242 , 0.43521499, 0.2973276 ,
                      0.73049925, 0.51426788, 0.5       , 0.58127285, 0.2910559 ,
                      0.40226652, 0.59710459, 0.42453628, 0.60622856, 0.30087059,
                      0.23674613, 0.70308893, 0.38839061, 0.41488322, 0.57563921,
                      0.29777361, 0.7138464 , 0.58414426, 0.36815957, 0.34806711,
                      0.39806773, 0.24045098, 0.31232754, 0.47886189, 0.55994448,
                      0.1957087 , 0.16537287, 0.5       , 0.59267271, 0.50743622,
                      0.45198026, 0.58069845, 0.48409389, 0.64544662, 0.32097684,
                      0.24951254, 0.54268176, 0.66017933, 0.49305559, 0.40135854])
    elif score and not binary:
        y = np.array([[0.08686481, 0.36784587, 0.54528932], [0.00608075, 0.37930655, 0.6146127 ],
                      [0.31197867, 0.47184302, 0.21617831], [0.07962032, 0.22574944, 0.69463024],
                      [0.08847057, 0.28220589, 0.62932354], [0.17820301, 0.22524605, 0.59655094],
                      [0.65799435, 0.27391896, 0.06808669], [0.07210765, 0.40975541, 0.51813694],
                      [0.10365176, 0.2883055 , 0.60804274], [0.07069342, 0.14640041, 0.78290617],
                      [0.04225733, 0.30661723, 0.65112544], [0.05212753, 0.25282374, 0.69504873],
                      [0.58021862, 0.15833403, 0.26144736], [0.48940257, 0.09298059, 0.41761684],
                      [0.03621747, 0.40861597, 0.55516656], [0.94222913, 0.04535753, 0.01241334],
                      [0.33026802, 0.11015509, 0.55957689], [0.13032154, 0.34524835, 0.52443011],
                      [0.32793716, 0.33667772, 0.33538513], [0.32915882, 0.13302978, 0.53781141],
                      [0.00667654, 0.46778536, 0.5255381 ], [0.06442941, 0.30416763, 0.63140296],
                      [0.13383071, 0.29925398, 0.56691531], [0.26156937, 0.34107188, 0.39735875],
                      [0.8654542 , 0.10720283, 0.02734297], [0.05772939, 0.37082729, 0.57144331],
                      [0.89594676, 0.0642072 , 0.03984605], [0.0247334 , 0.30785412, 0.66741248],
                      [0.05776183, 0.42717628, 0.51506189], [0.24544375, 0.51622814, 0.23832812],
                      [0.05212852, 0.38165493, 0.56621655], [0.96973297, 0.02236934, 0.0078977 ],
                      [0.346345  , 0.28896113, 0.36469387], [0.02013178, 0.46290312, 0.5169651 ],
                      [0.63709666, 0.17272588, 0.19017746], [0.66655212, 0.17685063, 0.15659725],
                      [0.79909719, 0.1447711 , 0.05613171], [0.40268017, 0.3716318 , 0.22568803],
                      [0.00983311, 0.39530294, 0.59486395], [0.904875  , 0.07419048, 0.02093451],
                      [0.00706293, 0.39157418, 0.60136288], [0.34186865, 0.46423605, 0.19389531],
                      [0.84673901, 0.10053164, 0.05272935], [0.08682893, 0.14184301, 0.77132806],
                      [0.92279554, 0.05398948, 0.02321499], [0.43196982, 0.30920049, 0.25882968],
                      [0.62644015, 0.11273521, 0.26082464], [0.95313682, 0.03622314, 0.01064004],
                      [0.29715717, 0.53117223, 0.1716706 ], [0.3163357 , 0.26320343, 0.42046086]])
    elif not score and binary:
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 
                      0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                      1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0])
    else:
        y = np.array([2, 2, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 
                      2, 2, 2, 1, 2, 2, 2, 0, 2, 0, 2, 1, 1, 2, 0, 2, 1, 
                      0, 0, 0, 0, 2, 0, 2, 1, 0, 2, 0, 0, 0, 0, 1, 2])
    return x, y


def test_multilabel_confusion_matrix():
    y_true, y_pred = get_data()

    def test(t, p):
        res = [[[30,  3], [ 4, 13]],
               [[25,  5], [17,  3]],
               [[22, 15], [ 2, 11]]]
        mcm = multilabel_confusion_matrix(t, p, num_classes=3)
        npt.assert_array_equal(mcm, res)

        mcm = multilabel_confusion_matrix(t, p, num_classes=None)
        npt.assert_array_equal(mcm, res)

        mcm = multilabel_confusion_matrix(t, p, num_classes=4)
        res.append([[len(t), 0], [0, 0]])
        npt.assert_array_equal(mcm, res)

        ## should raise error when num_classes < num_unique
        with pytest.raises(RuntimeError):
            mcm = multilabel_confusion_matrix(t, p, num_classes=2)

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_binary():
    y_true, y_pred = get_data(binary=True)

    def test(t, p):
        res = [[[17,  8], [ 3, 22]],
               [[22,  3], [ 8, 17]]]
        mcm = multilabel_confusion_matrix(t, p, num_classes=2)
        npt.assert_array_equal(mcm, res)

        mcm = multilabel_confusion_matrix(t, p, num_classes=None)
        npt.assert_array_equal(mcm, res)

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_precision_recall_f1_score():
    y_true, y_pred = get_data()

    for nc in (3, None):
        p, r, f = precision_recall_fscore(y_true, y_pred, num_classes=nc, average=None)
        npt.assert_array_almost_equal(p, [0.812, 0.375, 0.423], 2)
        npt.assert_array_almost_equal(r, [0.765, 0.15, 0.846], 2)
        npt.assert_array_almost_equal(f, [0.788, 0.214, 0.564], 2)

        pf = precision(y_true, y_pred, num_classes=nc, average=None)
        rf = recall(y_true, y_pred, num_classes=nc, average=None)
        npt.assert_array_almost_equal(p, pf, 3)
        npt.assert_array_almost_equal(r, rf, 3)

        f1f = f1_score(y_true, y_pred, num_classes=nc, average=None)
        ff = fbeta_score(y_true, y_pred, beta=1.0, num_classes=nc, average=None)
        npt.assert_array_almost_equal(f, f1f, 3)
        npt.assert_array_almost_equal(f, ff, 3)
    
    with pytest.warns(UserWarning):
        p, r, f = precision_recall_fscore(y_true, y_pred, num_classes=4, average=None)
        npt.assert_array_almost_equal(p, [0.812, 0.375, 0.423, 0], 2)
        npt.assert_array_almost_equal(r, [0.765, 0.15, 0.846, 0], 2)
        npt.assert_array_almost_equal(f, [0.788, 0.214, 0.564, 0], 2)

    all_ans = ((0.54, 0.54, 0.54), (0.537, 0.587, 0.522), (0.536, 0.54, 0.50))
    for ave, ans in zip(('micro', 'macro', 'weighted'), all_ans):
        ps = precision(y_true, y_pred, average=ave)
        npt.assert_array_almost_equal(ps, ans[0], 2)

        rs = recall(y_true, y_pred, average=ave)
        npt.assert_array_almost_equal(rs, ans[1], 2)

        f1s = f1_score(y_true, y_pred, average=ave)
        npt.assert_array_almost_equal(f1s, ans[2], 2)
    
    with pytest.raises(RuntimeError):
        precision_recall_fscore(y_true, y_pred, num_classes=2)


def test_precision_recall_f1_score_binary():
    y_true, y_pred = get_data(binary=True)

    for nc in (2, None):
        p, r, f = precision_recall_fscore(y_true, y_pred, num_classes=nc, average=None)
        npt.assert_array_almost_equal(p, [0.73, 0.85], 2)
        npt.assert_array_almost_equal(r, [0.88, 0.68], 2)
        npt.assert_array_almost_equal(f, [0.8, 0.756], 2)

    with pytest.warns(UserWarning):
        p, r, f = precision_recall_fscore(y_true, y_pred, num_classes=3, average=None)
        npt.assert_array_almost_equal(p, [0.73, 0.85, 0], 2)
        npt.assert_array_almost_equal(r, [0.88, 0.68, 0], 2)
        npt.assert_array_almost_equal(f, [0.8, 0.756, 0], 2)


def test_label_binarize():
    label, _ = get_data()
    label = label[:10]

    ans = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0],
                    [0, 0, 1], [0, 1, 0], [0, 1, 0],
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [0, 1, 0]])

    result = label_binarize(label)
    npt.assert_array_equal(result, ans)

    result = label_binarize(label, num_classes=3)
    npt.assert_array_equal(result, ans)

    result = label_binarize(label, unique=[0, 1, 2])
    npt.assert_array_equal(result, ans)

    result = label_binarize(label, num_classes=2)
    npt.assert_array_equal(result, ans[:, :2])

    result = label_binarize(label, unique=[0, 1])
    npt.assert_array_equal(result, ans[:, :2])

    result = label_binarize(label, num_classes=4)
    npt.assert_array_equal(result, np.c_[ans, np.zeros(ans.shape[0])])

    result = label_binarize(label, unique=[0, 1, 2, 3])
    npt.assert_array_equal(result, np.c_[ans, np.zeros(ans.shape[0])])

    result = label_binarize(label, unique=[0, 2])
    npt.assert_array_equal(result, ans[:, [0, 2]])

    result = label_binarize(label, unique=[1, 2, 3])
    npt.assert_array_equal(result, np.c_[ans[:, [1, 2]], np.zeros(ans.shape[0])])

    with pytest.raises(RuntimeError):
        result = label_binarize(label, num_classes=3, unique=[0, 1, 2])

    label, _ = get_data(binary=True)
    label = label[:10]
    result = label_binarize(label)
    npt.assert_array_almost_equal(result, label.reshape(-1, 1))

    result = label_binarize(label, num_classes=2)
    npt.assert_array_almost_equal(result, label.reshape(-1, 1))
    with pytest.raises(RuntimeError):
        result = label_binarize(label, unique=[0, 1])
    with pytest.raises(RuntimeError):
        result = label_binarize(label, num_classes=1)


def test_roc_curve():
    true, score = get_data(score=True)
    ans_fpr = [
        np.array([0.   , 0.   , 0.   , 0.121, 0.121, 0.212, 0.212, 0.273, 0.273,
                  0.303, 0.303, 1.   ]), 
        np.array([0.   , 0.033, 0.167, 0.167, 0.2  , 0.2  , 0.267, 0.267, 0.367,
                  0.367, 0.433, 0.433, 0.467, 0.467, 0.5  , 0.5  , 0.533, 0.533,
                  0.567, 0.567, 0.633, 0.633, 0.667, 0.667, 0.7  , 0.7  , 0.8  ,
                  0.8  , 1.   ]), 
        np.array([0.   , 0.027, 0.081, 0.081, 0.135, 0.135, 0.189, 0.189, 0.216,
                  0.216, 0.243, 0.243, 0.27 , 0.27 , 0.297, 0.297, 0.432, 0.432,
                  0.649, 0.649, 1.   ])
    ]
    ans_tpr = [
        np.array([0.   , 0.059, 0.706, 0.706, 0.765, 0.765, 0.824, 0.824, 0.941,
                  0.941, 1.   , 1.   ]), 
        np.array([0.  , 0.  , 0.  , 0.2 , 0.2 , 0.3 , 0.3 , 0.35, 0.35, 0.45, 0.45,
                  0.5 , 0.5 , 0.55, 0.55, 0.6 , 0.6 , 0.7 , 0.7 , 0.75, 0.75, 0.85,
                  0.85, 0.9 , 0.9 , 0.95, 0.95, 1.  , 1.  ]), 
        np.array([0.   , 0.   , 0.   , 0.231, 0.231, 0.385, 0.385, 0.462, 0.462,
                  0.538, 0.538, 0.615, 0.615, 0.692, 0.692, 0.846, 0.846, 0.923,
                  0.923, 1.   , 1.   ])
    ]
    ans_th = [
        np.array([1.97 , 0.97 , 0.637, 0.432, 0.403, 0.33 , 0.329, 0.316, 0.297,
                  0.262, 0.245, 0.006]), 
        np.array([1.531, 0.531, 0.464, 0.409, 0.395, 0.382, 0.372, 0.371, 0.341,
                  0.309, 0.307, 0.304, 0.299, 0.289, 0.288, 0.282, 0.274, 0.253,
                  0.226, 0.225, 0.173, 0.146, 0.145, 0.142, 0.133, 0.113, 0.101,
                  0.093, 0.022]), 
        np.array([1.783, 0.783, 0.695, 0.651, 0.629, 0.608, 0.597, 0.595, 0.571,
                  0.567, 0.566, 0.56 , 0.555, 0.545, 0.538, 0.524, 0.418, 0.397,
                  0.216, 0.194, 0.008])
    ]
    ans_auc = [0.93048, 0.57333, 0.76091]

    def test(f,t,th, cl=None):
        aucs = [auc(x, y) for x,y in zip(f, t)]
        if cl is None:
            cl = list(range(len(f)))
        elif isinstance(cl, int):
            cl = [cl]
        for i in range(len(f)):
            assert f[i].shape == t[i].shape and f[i].shape == th[i].shape
            npt.assert_array_almost_equal(f[i], ans_fpr[cl[i]], decimal=2)
            npt.assert_array_almost_equal(t[i], ans_tpr[cl[i]], decimal=2)
            npt.assert_array_almost_equal(th[i], ans_th[cl[i]], decimal=2)
            npt.assert_almost_equal(aucs[i], ans_auc[cl[i]], decimal=4)

    fpr, tpr, th = roc_curve(true, score)
    test(fpr, tpr, th)

    fpr, tpr, th = roc_curve(true, score, classes=2)
    test([fpr], [tpr], [th], cl=2)

    fpr, tpr, th = roc_curve(true, score, classes=[1, 2])
    test(fpr, tpr, th, cl=[1, 2])

    with pytest.raises(RuntimeError):
        fpr, tpr, th = roc_curve(true, score, classes=[1, 2, 3])

    fpr, tpr, th = roc_curve(true, np.ones(score.shape))
    aucs = [auc(x, y) for x,y in zip(fpr, tpr)]
    npt.assert_almost_equal(aucs, 0.50, decimal=2)

    fpr, tpr, th = roc_curve(true, np.zeros(score.shape))
    aucs = [auc(x, y) for x,y in zip(fpr, tpr)]
    npt.assert_almost_equal(aucs, 0.50, decimal=2)

    with pytest.raises(RuntimeError): ## incompatible shape
        auc([0, 0, 1], [0.1, 0.2])

    with pytest.raises(ValueError): ## single value data
        auc([0], [0.1])
    
    with pytest.raises(ValueError):
        auc([2, 1, 3, 4], [5, 6, 7, 8])


def test_roc_curve_binary():
    true, score = get_data(binary=True, score=True)
    ans_fpr = np.array([0.  , 0.  , 0.  , 0.04, 0.04, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12,
                        0.16, 0.16, 0.2 , 0.2 , 0.44, 0.44, 1.  ])
    ans_tpr = np.array([0.  , 0.04, 0.08, 0.08, 0.4 , 0.4 , 0.64, 0.64, 0.68, 0.76, 0.8 ,
                        0.8 , 0.84, 0.84, 0.96, 0.96, 1.  , 1.  ])
    ans_th = np.array([1.73 , 0.73 , 0.714, 0.708, 0.593, 0.584, 0.526, 0.514, 0.507,
                       0.5  , 0.493, 0.484, 0.479, 0.457, 0.435, 0.398, 0.388, 0.165])

    fpr, tpr, th = roc_curve(true, score)
    assert fpr.shape == tpr.shape and fpr.shape == th.shape
    npt.assert_array_almost_equal(fpr, ans_fpr, decimal=2)
    npt.assert_array_almost_equal(tpr, ans_tpr, decimal=2)
    npt.assert_array_almost_equal(th, ans_th, decimal=2)

    aucs = auc(fpr, tpr)
    npt.assert_almost_equal(aucs, 0.9008, decimal=3)

    with pytest.raises(RuntimeError):
        fpr, tpr, th = roc_curve(true, score, classes=1)


def test_roc_curve_toydata():
    y_true = [0, 1]
    y_score = [0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    npt.assert_array_almost_equal(fpr, [0, 0, 1])
    npt.assert_array_almost_equal(tpr, [0, 1, 1])
    npt.assert_almost_equal(roc_auc, 1.)
    npt.assert_array_almost_equal(auc(y_true, y_score), 0.5)

    y_true = [0, 1]
    y_score = [1, 0]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    npt.assert_array_almost_equal(fpr, [0, 1, 1])
    npt.assert_array_almost_equal(tpr, [0, 0, 1])
    npt.assert_almost_equal(roc_auc, 0.)
    npt.assert_array_almost_equal(auc(y_score, y_true), 0.5)

    y_true = [1, 0]
    y_score = [1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    npt.assert_array_almost_equal(fpr, [0, 1])
    npt.assert_array_almost_equal(tpr, [0, 1])
    npt.assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    npt.assert_array_almost_equal(fpr, [0, 0, 1])
    npt.assert_array_almost_equal(tpr, [0, 1, 1])
    npt.assert_almost_equal(roc_auc, 1.)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    npt.assert_array_almost_equal(fpr, [0, 1])
    npt.assert_array_almost_equal(tpr, [0, 1])
    npt.assert_almost_equal(roc_auc, .5)


def test_roc_auc_score():
    true, score = get_data(score=True)

    npt.assert_almost_equal(
        roc_auc_score(true, score, average="macro", method="ovr"), 
        0.75490979, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="macro", method="ovr", num_classes=3), 
        0.75490979, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="macro", method="ovo"), 
        0.75482655, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="macro", method="ovo", num_classes=3), 
        0.75482655, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="weighted", method="ovr"), 
        0.74353481, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="weighted", method="ovr", num_classes=3), 
        0.74353481, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="weighted", method="ovo"), 
        0.75077602, decimal=6
    )
    npt.assert_almost_equal(
        roc_auc_score(true, score, average="weighted", method="ovo", num_classes=3), 
        0.75077602, decimal=6
    )

    ## TODO: extra classes from num_classes


    with pytest.raises(RuntimeError): 
        roc_auc_score(true, score, num_classes=2)

    ## partial roc not supported in multiclass
    with pytest.raises(RuntimeError): 
        roc_auc_score(true, score, max_fpr=0.9)

    ## sum of class-wise score not 1
    with pytest.raises(ValueError): 
        sc = score
        sc[:, 2] = 0.5
        roc_auc_score(true, sc)
    
    ## dim 2 in score not equal to num classes found
    with pytest.raises(ValueError):
        roc_auc_score(true, np.c_[score, np.zeros(score.shape[0])])

    ## toy data ovo
    sc = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], 
                   [0, 0.2, 0.8]])
    tr = np.array([0, 1, 0, 2])

    auc_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    auc_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    ave_auc_01 = (auc_01 + auc_10) / 2
    auc_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    auc_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    ave_auc_02 = (auc_02 + auc_20) / 2
    auc_12 = roc_auc_score([1, 0], [0.4, 0.2])
    auc_21 = roc_auc_score([0, 1], [0.3, 0.8])
    ave_auc_12 = (auc_12 + auc_21) / 2
    
    ovo_unweighted = (ave_auc_01 + ave_auc_02 + ave_auc_12) / 3
    npt.assert_almost_equal(
        roc_auc_score(tr, sc, average="macro", method="ovo"),
        ovo_unweighted
    )
    ovo_weighted = np.average([ave_auc_01, ave_auc_02, ave_auc_12], 
        weights=[0.75, 0.75, 0.50])
    npt.assert_almost_equal(
        roc_auc_score(tr, sc, average="weighted", method="ovo"),
        ovo_weighted
    )

    ## toy data ovr
    sc = np.array([[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], 
                   [0.3, 0.3, 0.4]])
    tr = np.array([0, 1, 2, 2])
    out_0 = roc_auc_score([1, 0, 0, 0], sc[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], sc[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], sc[:, 2])

    ovr_unweighted = (out_0 + out_1 + out_2) / 3.
    npt.assert_almost_equal(
        roc_auc_score(tr, sc, average="macro", method="ovr"),
        ovr_unweighted
    )
    ovr_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    npt.assert_almost_equal(
        roc_auc_score(tr, sc, average="weighted", method="ovr"),
        ovr_weighted
    )

    true, score = get_data(binary=True, score=True)
    npt.assert_almost_equal(roc_auc_score(true, score), 0.9008)


def test_pr_curve():
    true, score = get_data(score=True)
    ans_prec = [
        np.array([0.63 , 0.615, 0.64 , 0.625, 0.609, 0.636, 0.667, 0.65 , 0.684,
                  0.722, 0.765, 0.75 , 0.8  , 0.857, 0.923, 1.   , 1.   , 1.   ,
                  1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
                  1.   ]), 
        np.array([0.455, 0.442, 0.452, 0.463, 0.475, 0.462, 0.474, 0.459, 0.472,
                  0.457, 0.441, 0.455, 0.469, 0.452, 0.467, 0.448, 0.429, 0.444,
                  0.423, 0.44 , 0.417, 0.435, 0.409, 0.429, 0.45 , 0.421, 0.389,
                  0.412, 0.438, 0.467, 0.429, 0.462, 0.5  , 0.455, 0.4  , 0.444,
                  0.375, 0.286, 0.167, 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]), 
        np.array([0.351, 0.333, 0.343, 0.353, 0.364, 0.375, 0.387, 0.4  , 0.414,
                  0.429, 0.407, 0.423, 0.44 , 0.458, 0.478, 0.5  , 0.476, 0.45 ,
                  0.474, 0.444, 0.471, 0.438, 0.467, 0.429, 0.462, 0.417, 0.455,
                  0.5  , 0.444, 0.375, 0.429, 0.5  , 0.4  , 0.25 , 0.   , 0.   ,
                  0.   , 1.   ])
    ]
    ans_rec = [
        np.array([1.   , 0.941, 0.941, 0.882, 0.824, 0.824, 0.824, 0.765, 0.765,
                  0.765, 0.765, 0.706, 0.706, 0.706, 0.706, 0.706, 0.647, 0.588,
                  0.529, 0.471, 0.412, 0.353, 0.294, 0.235, 0.176, 0.118, 0.059,
                  0.   ]), 
        np.array([1.  , 0.95, 0.95, 0.95, 0.95, 0.9 , 0.9 , 0.85, 0.85, 0.8 , 0.75,
                  0.75, 0.75, 0.7 , 0.7 , 0.65, 0.6 , 0.6 , 0.55, 0.55, 0.5 , 0.5 ,
                  0.45, 0.45, 0.45, 0.4 , 0.35, 0.35, 0.35, 0.35, 0.3 , 0.3 , 0.3 ,
                  0.25, 0.2 , 0.2 , 0.15, 0.1 , 0.05, 0.  , 0.  , 0.  , 0.  , 0.  ,
                  0.  ]), 
        np.array([1.   , 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923,
                  0.923, 0.846, 0.846, 0.846, 0.846, 0.846, 0.846, 0.769, 0.692,
                  0.692, 0.615, 0.615, 0.538, 0.538, 0.462, 0.462, 0.385, 0.385,
                  0.385, 0.308, 0.231, 0.231, 0.231, 0.154, 0.077, 0.   , 0.   ,
                  0.   , 0.   ])
    ]
    ans_thr = [
        np.array([0.245, 0.262, 0.297, 0.312, 0.316, 0.328, 0.329, 0.33 , 0.342,
                  0.346, 0.403, 0.432, 0.489, 0.58 , 0.626, 0.637, 0.658, 0.667,
                  0.799, 0.847, 0.865, 0.896, 0.905, 0.923, 0.942, 0.953, 0.97 ]), 
        np.array([0.093, 0.101, 0.107, 0.11 , 0.113, 0.133, 0.142, 0.145, 0.146,
                  0.158, 0.173, 0.177, 0.225, 0.226, 0.253, 0.263, 0.274, 0.282,
                  0.288, 0.289, 0.299, 0.304, 0.307, 0.308, 0.309, 0.337, 0.341,
                  0.345, 0.368, 0.371, 0.372, 0.379, 0.382, 0.392, 0.395, 0.409,
                  0.41 , 0.427, 0.463, 0.464, 0.468, 0.472, 0.516, 0.531]), 
        np.array([0.194, 0.216, 0.226, 0.238, 0.259, 0.261, 0.261, 0.335, 0.365,
                  0.397, 0.418, 0.42 , 0.515, 0.517, 0.518, 0.524, 0.526, 0.538,
                  0.545, 0.555, 0.56 , 0.566, 0.567, 0.571, 0.595, 0.597, 0.601,
                  0.608, 0.615, 0.629, 0.631, 0.651, 0.667, 0.695, 0.695, 0.771,
                  0.783])
    ]

    def test(p,r,th, cl=None):
        if cl is None:
            cl = list(range(len(p)))
        elif isinstance(cl, int):
            cl = [cl]
        for i in range(len(p)):
            assert p[i].shape == r[i].shape and p[i].shape[0] == th[i].shape[0] + 1
            npt.assert_array_almost_equal(p[i], ans_prec[cl[i]], decimal=2)
            npt.assert_array_almost_equal(r[i], ans_rec[cl[i]], decimal=2)
            npt.assert_array_almost_equal(th[i], ans_thr[cl[i]], decimal=2)
            assert p[i][-1] == 1

    prc, rec, thr = pr_curve(true, score)
    test(prc, rec, thr)

    prc, rec, thr = pr_curve(true, score, classes=2)
    test([prc], [rec], [thr], cl=2)

    prc, rec, thr = pr_curve(true, score, classes=[1, 2])
    test(prc, rec, thr, cl=[1, 2])

    with pytest.raises(RuntimeError):
        prc, rec, thr = pr_curve(true, score, classes=[1, 2, 3])

    prc1, rec1, thr1 = pr_curve(true, np.ones(score.shape))
    for i in range(3):
        assert prc1[i][-1] == 1
        npt.assert_equal(rec1[i], [1., 0.])
        npt.assert_equal(thr1[i], [1.])

    prc0, rec0, thr0 = pr_curve(true, np.zeros(score.shape))
    for i in range(3):
        assert prc1[i][-1] == 1
        npt.assert_equal(prc0[i], prc1[i])
        npt.assert_equal(rec0[i], rec1[i])
        npt.assert_equal(thr0[i], [0.])


def test_pr_curve_binary():
    true, score = get_data(binary=True, score=True)
    ans_prc = np.array([0.694, 0.686, 0.706, 0.727, 0.75 , 0.774, 0.8  , 0.828, 0.821,
       0.815, 0.808, 0.84 , 0.833, 0.87 , 0.864, 0.85 , 0.842, 0.889,
       0.882, 0.875, 0.867, 0.857, 0.846, 0.833, 0.909, 0.9  , 0.889,
       0.875, 0.857, 0.833, 0.8  , 0.75 , 0.667, 1.   , 1.   , 1.   ])
    ans_rec = np.array([1.  , 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.92, 0.88, 0.84,
       0.84, 0.8 , 0.8 , 0.76, 0.68, 0.64, 0.64, 0.6 , 0.56, 0.52, 0.48,
       0.44, 0.4 , 0.4 , 0.36, 0.32, 0.28, 0.24, 0.2 , 0.16, 0.12, 0.08,
       0.08, 0.04, 0.  ])
    ans_thr = np.array([0.388, 0.398, 0.401, 0.402, 0.403, 0.415, 0.425, 0.435, 0.439,
       0.452, 0.457, 0.479, 0.484, 0.493, 0.5  , 0.507, 0.514, 0.526,
       0.543, 0.56 , 0.576, 0.581, 0.581, 0.584, 0.593, 0.597, 0.606,
       0.629, 0.636, 0.645, 0.66 , 0.703, 0.708, 0.714, 0.73 ])

    prc,rec,thr = pr_curve(true, score)
    npt.assert_array_almost_equal(prc, ans_prc, decimal=2)
    npt.assert_array_almost_equal(rec, ans_rec, decimal=2)
    npt.assert_array_almost_equal(thr, ans_thr, decimal=2)
    assert prc.size == rec.size
    assert prc.size == thr.size + 1

    prc0,rec0,thr0 = pr_curve(true, np.zeros_like(score))
    prc1,rec1,thr1 = pr_curve(true, np.ones_like(score))
    npt.assert_array_almost_equal(prc0, prc1)
    npt.assert_array_almost_equal(rec0, rec1)
    assert thr0[0] == 0.
    assert thr1[0] == 1.

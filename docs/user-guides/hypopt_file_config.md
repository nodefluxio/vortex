# Hyperparameter Optimization File Configuration

An additional YAML configuration file is needed for hyperparameter optimization configuration. This hypopt configuration file will describe the parameters and objective to be optimized. Several examples of hypopt config file can be inspected on [this link](https://github.com/nodefluxio/vortex/tree/master/experiments/hypopt) In this guide, we will describe the configurations needed for Optuna-based hyperparameter optimization

---

All available configurations is listed below :

## Study Name

Flagged with `study_name` key (str) in the hypopt config file. This field is used to identify the name of the hypopt attempt. Will be combined with the [`experiment_name`](experiment_file_config.md#experiment-name) field in the experiment file to identify an [Optuna study](https://optuna.readthedocs.io/en/latest/reference/study.html). E.g. :

```yaml
study_name: detection_param_search
```

---

## Parameters

Flagged with `parameters` key (list[dict]) in the hypopt config file. This field will contain a list of hyperparameters that will be searched. However, each parameter mentioned in here must be declared first in the experiment file using any initial value. To declare a parameter we use a flattened XML structure with dot (`.`) reducer.

For example we want to search the parameter of `score_threshold` and `iou_threshold` in which their structure in the experiment file is shown below :

```yaml
## optional field for validation step
validator: {
    ## passed to validator class
    args: {
        score_threshold: 0.9,
        iou_threshold: 0.2,
    },
    val_epoch: 5,
},
```

Thus, we declare these parameters in the hypopt config as shown below:

```yaml
parameters: [
    validator.args.score_threshold: {
        suggestion: suggest_discrete_uniform,
        args: {
            low: 0.05,
            high: 0.95,
            q: 0.025,
        }
    },
    validator.args.iou_threshold: {
        suggestion: suggest_discrete_uniform,
        args: {
            low: 0.05,
            high: 0.5,
            q: 0.05,
        }
    },
]
```

Each parameter presented as (dict) type and have the following key arguments:

- `suggestion` (str) : define the Optuna trial object’s suggestion method which will be used, any `suggest_*` method should be supported. For full reference see [this link](https://optuna.readthedocs.io/en/latest/reference/trial.html)
- `args` (dict) : the corresponding arguments to the respective `suggestion` function

---

## Objective

Flagged with `objective` key (dict) in the hypopt config file. This field denotes the function which we want to optimize, e.g. minimizing training loss or maximizing validation metric. In Vortex, we provide two general objective settings related to development pipelines. E.g. :

```yaml
objective: {
    module: TrainObjective,
    args: {
        metric_type: val, ## [val, loss]
        metric_name: mean_ap, ## if metric_type==val
        ## final objective value is the reduced validation metrics
        reduction: average, # reduction is based on numpy function (e.g. np.average, np.max, etc.)
        reduction_args: {
            weights: [ 1, 2, 3, 4, 5 ]
        }
    }
}
```

Arguments :

- `module` (str) : denotes a specific objective function module. Supported objective function is available in the next sub-section
- `args` (dict) : the corresponding arguments for selected `module`

---

### Train Objective Optimization

This objective aim to optimize hyperparameters on [train pipeline](pipelines.md#training-pipeline)
. E.g. :

```yaml
objective: {
    module: TrainObjective,
    args: {
        metric_type: val, ## [val, loss]
        metric_name: mean_ap, ## if metric_type==val
        ## final objective value is the reduced validation metrics
        reduction: average, # reduction is based on numpy function (e.g. np.average, np.max, etc.)
        reduction_args: {
            weights: [ 1, 2, 3, 4, 5 ]
        }
    }
}
```

Arguments : 

- `metric_type` (str) : type of metric to be optimized : `val` or `loss`. 

    - `val` metric is extracted from in-training-loop validation process. So the provided **experiment file** must be valid for validation.
    - `loss` metric is extracted from training process

- `metric_name` (str) : only used if `metric_type` is set to `val`. This argument denotes the name of the metric which want to be optimized. The available settings for this argument are also related to the model's task

    - Detection task (see [this link](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52) for further reading) :

        - `mean_ap` : using the mean-average precision metrics

    - Classification task (see [this link](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) for further reading):

        - `accuracy` : using the accuracy metrics
        - `precision (micro)` : using the micro-average precision metrics
        - `precision (macro)` : using the macro-average precision metrics
        - `precision (weighted)` : using the weighted-average precision metrics
        - `recall (micro)` : using the micro-average recall metrics
        - `recall (macro)` : using the macro-average recall metrics
        - `recall (weighted)` : using the weighted-average recall metrics
        - `f1_score (micro)` : using the micro-average f1_score metrics
        - `f1_score (macro)` : using the macro-average f1_score metrics
        - `f1_score (weighted)` : using the weighted-average f1_score metrics

- `reduction` (str) : the reduction function used to averaged the returned value from training pipeline. Supported reduction :

    - `latest` : select the last value ( index [-1] ), 
    - `mean` : see [numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
    - `sum` : see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
    - `min` : see [numpy.min](https://numpy.org/doc/stable/reference/generated/numpy.amin.html)
    - `max` : see [numpy.max](https://numpy.org/doc/stable/reference/generated/numpy.amax.html)
    - `median` : see [numpy.median](https://numpy.org/doc/stable/reference/generated/numpy.median.html)
    - `average` : see [numpy.average](https://numpy.org/doc/stable/reference/generated/numpy.average.html)
    - `percentile` : see [numpy.percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
    - `quantile` : see [numpy.quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)

- `reduction_args` (dict) :  the corresponding arguments for selected `reduction`.

**Additional Explanation** : This objective utilize training pipelines which will return a list of values ( not singular value ), either a sequence of recorded loss value on each epoch, (`100` epoch means list of `100` loss values) or a sequence of validation metrics ( `val_epoch` set to `5` and `100` epoch means list of `100/5 = 20` metric values ).

However, Optuna expect a singular value to be optimized. Thus, we apply a reduction method using numpy functionality to pick the most representative value for that trial.

**Important Notes** : Be careful when using weighted average, because the `weights` args expect the user to input a weight array with the same length compared to the input. So you must calculate the length yourself.

E.g. :

- You want to use weighted `average` reduction on `metric_type` = loss. `epoch` is set to 50, so the input length to reduction function is 50 ( each epoch will dump 1 loss value ). Hence you need to provide `weights` in the `reduction_args` with the same length
- You want to use `average` reduction on `metric_type` = val. `epoch` is set to `50`, `val_epoch` is set to `5`, so the input length to reduction function is `50/5 = 10` ( `epoch`/`val_epoch` , each val_epoch will dump 1 validation metrics value). Hence you need to provide `weights` in the `reduction_args` with the same length

---

### Validation Objective Optimization

This objective aim to optimize hyperparameters post-training pipeline (validate, predict) and utilize [validation pipeline](pipelines.md#validation-pipeline)

E.g. :

```yaml
objective: {
    module: ValidationObjective,
    args: {
        metric_name: mean_ap
    }
}
```

Arguments : 

- `metric_name` (str) : denotes the name of the metric which want to be optimized. The available settings for this argument are also related to the model's task

    - Detection task (see [this link](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52) for further reading) :

        - `mean_ap` : using the mean-average precision metrics

    - Classification task (see [this link](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics) for further reading):

        - `accuracy` : using the accuracy metrics
        - `precision (micro)` : using the micro-average precision metrics
        - `precision (macro)` : using the macro-average precision metrics
        - `precision (weighted)` : using the weighted-average precision metrics
        - `recall (micro)` : using the micro-average recall metrics
        - `recall (macro)` : using the macro-average recall metrics
        - `recall (weighted)` : using the weighted-average recall metrics
        - `f1_score (micro)` : using the micro-average f1_score metrics
        - `f1_score (macro)` : using the macro-average f1_score metrics
        - `f1_score (weighted)` : using the weighted-average f1_score metrics

---

## Study

Flagged with `objective` key (dict) in the hypopt config file. This field set the Optuna study initialization. As stated in [this link](https://optuna.readthedocs.io/en/latest/reference/study.html#study), a study corresponds to an optimization task, i.e. a set of trials.

E.g. :

```yaml
study: {
    n_trials: 5,
    direction: maximize,
    pruner: {
        method: MedianPruner,
        args: {},
    },
    sampler: {
        method: TPESampler,    
        args: {},
    },
    args: {
        storage: sqlite://anysqldb_url.db,
        load_if_exists: True,
    }
}
```

Arguments :

- `n_trials` (int) : number of trials attempted in a study
- `direction` (str) : either `maximize` or `minimize` the objective value
- `pruner` (dict) (Optional): enable Optuna pruner configuration, to judge whether the trial should be pruned based on the reported values, full reference in [this link](https://optuna.readthedocs.io/en/latest/reference/pruners.html). Sub-arguments :

    - `method` (str) : specify the Pruner’s method which is going to be used
    - `args` (dict) : the corresponding arguments to the respective pruner `method`

- `sampler` (dict) (Optional) : enable Optuna different sampler to sample the combination of hyperparameters value in a search space, full reference in [this link](https://optuna.readthedocs.io/en/latest/reference/samplers.html). Sub-arguments :

    - `method` (str) : specify the Sampler’s method which is going to be used
    - `args` (dict) : the corresponding arguments to the respective sampler `method`

- `args` (dict) (Optional) : Additional Optuna Study’s arguments to use database to save studies. Sub-arguments :

    - `storage` (str) : database URL. If this argument is set to None, in-memory storage is used, and the Study will not be persistent, full reference in [this link](https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study)
    - `load_if_exists` (str) : flag to control the behavior to handle a conflict of study names, full reference in [this link](https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study)

---

## Additional Parameters' Override

This config is used to override any experiment file configuration besides the search parameters mentioned in [parameters section](#parameters). 

For example, using `epoch = 200` in initial experiment file config is too long if we want to set the `n_trials = 20` ( means `20 * 200` epoch without any sampler ), so for hypopt we may only use `10` epochs. In order to do that we assign the overridden parameter in this field with flattened XML structure similar to [parameters section](#parameters)

E.g.:

```yaml
override: {
    trainer.epoch: 10,
    validator.val_epoch: 2,
}
```

---

## Additional Experiment Configuration

Additional experiment configuration which wants to be added to the original experiment file while optimized. Mandatory but can be empty

E.g. :

```yaml
additional_config: {}
```
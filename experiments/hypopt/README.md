## Pre-Training Hyperparameter Search

### Experiments

- [learning_rate_search.yml](learning_rate_search.yml) : find `lr` for `SGD` and `warmup_t` for `CosineLRWithWarmUp` should work with config that use `SGD` and `CosineLRWithWarmUp`
- [optimizer_search.yml](optimizer_search.yml): find best optimizer (`SGD` or `Adam`) and scheduler (`CosineLRWithWarmUp` or None)

### Example Results

- learning rate search: `efficientnet_b0` with `cifar10` dataset
  ![efficientnet_b0_softmax_cifar10_learning_rate_search_hypopt_contour.png](../outputs/hypopt/efficientnet_b0_softmax_cifar10_learning_rate_search_hypopt_contour.png)

## Post-Training Hyperparameter Search

### Experiments

- [detection_param_search.yml](detection_param_search.yml) : find `score_threshold` and `iou_threshold` should work with any detection model
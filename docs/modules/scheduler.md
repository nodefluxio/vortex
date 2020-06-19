# Learning Rates Scheduler

This section listed all available `scheduler` configuration. Part of [`trainer` configurations](../user-guides/experiment_file_config.md#trainer) in experiment file.

---

## Pytorch Scheduler

The following example configuration uses the Pytorch `StepLR` scheduler. You can use the other scheduler by following similar fashion. List of Pytorch supported scheduler can be found in [this link](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

```yanml
scheduler: {
    method: StepLR,
    args: {
        step_size: 50,
        gamma: 0.1
    }
},
```

---

## CosineLR

Implement Cosine decay scheduler with warm restarts

Reference : 

- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [allennlp/cosine.py](https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py)

```yaml
scheduler: {
    method: CosineLRScheduler,
    args: {
        t_initial: 200,
        t_mul: 1.0,
        lr_min: 0.00001,
        warmup_lr_init: 0.00001,
        warmup_t: 5,
        cycle_limit: 1,
        t_in_epochs: True,
        decay_rate: 0.1,
    }
},
```

Arguments : 

- `t_initial` (int) : the number of iterations (epochs) within the first cycle
- `t_mul` (float) : determines the number of iterations (epochs) in the i-th decay cycle, which is the length of the last cycle multiplied by `t_mul`. default : 1
- `lr_min` (float) : minimum learning rate after decay. default : 0.
- `warmup_lr_init` (float) : starting learning rate on warmup stage. default : 0
- `warmup_t` (int) : number of epoch of warmup stage. default : 0
- `cycle_limit` (int) : number of cosine cycle. default : 0
- `t_in_epochs` (bool) : if True, update learning rate per epoch, if not, update per step. default : True
- `decay_rate` (float) : learning rate decay rate. default : 1

---

## TanhLR

Implement Hyperbolic-Tangent decay with warm restarts

Reference :

- [Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification](https://arxiv.org/abs/1806.01593)

```yaml
 scheduler: {
    method: TanhLRScheduler,
    args: {
        t_initial: 200,
        t_mul: 1.0,
        lb: -6.,
        ub: 4.,
        lr_min: 0.00001,
        warmup_lr_init: 0.00001,
        warmup_t: 5,
        cycle_limit: 1,
        t_in_epochs: True,
        decay_rate: 0.1,
    }
}
```

Arguments : 

- `t_initial` (int) : the number of iterations (epochs) within the first cycle
- `t_mul` (float) : determines the number of iterations (epochs) in the i-th decay cycle, which is the length of the last cycle multiplied by `t_mul`. default : 1
- `lb` (float) : tanh function lower bound value
- `ub` (float) : tanh function upper bound value
- `lr_min` (float) : Minimum learning rate after decay. default : 0.
- `warmup_lr_init` (float) : starting learning rate on warmup stage. default : 0
- `warmup_t` (int) : number of epoch of warmup stage. default : 0
- `cycle_limit` (int) : number of cosine cycle. default : 0
- `t_in_epochs` (bool) : if True, update learning rate per epoch, if not, update per step. default : True
- `decay_rate` (float) : learning rate decay rate. default : 1

---

## StepLRWithBurnIn

Implement StepLR scheduler with burn in (warm start), adapted from YOLOv3 training method

Reference : 

- [DeNA/PyTorch_YOLOv3: Implementation of YOLOv3 in PyTorch](https://github.com/DeNA/PyTorch_YOLOv3)

```yaml
scheduler: {
    method: StepLRWithBurnIn,
    args: {
        burn_in: 5,
        steps: [180,190],
        scales: [.1,.1],
        last_epoch: -1
    }
}
```

Arguments :

- `burn_in` (int) : number of epochs for warm up
- `steps` (list) : list of epoch when the learning rate will be reduced, e.g. [180,190] --> learning rate will be reduced on epoch 180 and epoch 190
- `scales` (list) : scale of the reduced learning rate, e.g. [0.1,0.1] --> e.g. initial lr == 0.01 , on epoch 180 will be reduced to 0.1 * 0.01 = 0.001 and on epoch 190 will be reduced to 0.1 * 0.001 = 0.0001
- `last_epoch` (int) : last epoch number. default : -1



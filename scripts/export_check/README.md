## COMPATIBILITY CHECK   
Test and report model for onnx exporting and runtime execution.
```
usage: compatibility_check.py [-h]
                              [--opset-version [OPSET_VERSION [OPSET_VERSION ...]]]
                              [--example-image EXAMPLE_IMAGE]
                              [--backbones {darknet53,efficientnet_b0,...,vgg19} ...]]
                              [--models {FPNSSD,RetinaFace,softmax} [{FPNSSD,RetinaFace,softmax} ...]]
                              [--exclude-backbones {darknet53,efficientnet_b0,...,vgg19} ...]]
                              [--exclude-models {FPNSSD,RetinaFace,softmax} [{FPNSSD,RetinaFace,softmax} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --opset-version [OPSET_VERSION [OPSET_VERSION ...]]
  --example-image EXAMPLE_IMAGE
                        optional example image for tracing
  --backbones {darknet53,efficientnet_b0,...,vgg19} ...]
                        backbone(s) to test
  --models {FPNSSD,RetinaFace,softmax} [{FPNSSD,RetinaFace,softmax} ...]
                        model(s) to test
  --exclude-backbones {darknet53,efficientnet_b0,...,vgg19_bn,vgg19} ...]
                        exclude this backbone(s) when testing
  --exclude-models {FPNSSD,RetinaFace,softmax} [{FPNSSD,RetinaFace,softmax} ...]
                        model(s) to exclude
```
- `--models` are useful if you want to test only one or two model
- `--backbones` are useful if you want to test only one or two backbone
- `--exclude-models` are useful if you want to **SKIP** test for one or two model
- `--exclude-backbones` are useful if you want to **SKIP** test for one or two backbone
- combine `--models` and `--backbones` to test several model and backbone combinations
- combine `--models`, `--backbones`, `--exclude-models`, `--exclude-backbones` as you like
   
example 
```
python3.6 exporter/scripts/compatibility_check.py --opset-version 11 --models RetinaFace --backbones efficientnet_b0
```
```
warning : this check might be strorage and memory intensive
[export opset 11] trying to export efficientnet_b0-RetinaFace:  [0/1]
./networks/models/detection/retinaface.py:156: UserWarning: inferring feature maps size from image size and stride
  warnings.warn('inferring feature maps size from image size and stride')
./networks/models/detection/retinaface.py:249: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
  nn.init.xavier_uniform(m.weight.data)
./networks/modules/preprocess/normalizer.py:18: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert img.size(3) == 3 or img.size(3) == 1, "to_tensor only support NHWC input layout, "\
./networks/modules/backbones/efficientnet.py:89: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
./networks/modules/backbones/efficientnet.py:89: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
./networks/modules/backbones/efficientnet.py:95: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if pad_h > 0 or pad_w > 0:
./networks/modules/heads/ssh.py:182: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  lmk_tensor.permute(0,2,3,1).contiguous().view(n,-1,self.n_landmarks.item()*2)
./networks/modules/postprocess/retinaface.py:20: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert input.size(0) == 1
./networks/modules/postprocess/retinaface.py:21: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert input.size(2) == (6 + self.n_landmarks*2)
./networks/modules/postprocess/retinaface.py:23: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert score_threshold.size() == Size([1])
./networks/modules/losses/utils/ssd.py:313: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert loc.size(1) == 4
./networks/modules/losses/utils/ssd.py:315: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert priors.size(1) == 4
./networks/modules/postprocess/utils/nms.py:113: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if detections.size()[0] > 1:
ONNX export failed: Couldn't export Python operator SwishJitAutoFn

Defined at:
./networks/modules/utils/activations.py(48): swish
./networks/modules/utils/activations.py(96): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(516): _slow_forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(530): __call__
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/container.py(100): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(516): _slow_forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(530): __call__
./networks/modules/backbones/base_backbone.py(35): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(516): _slow_forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(530): __call__
./networks/modules/heads/fpn.py(106): forward
./networks/models/detection/retinaface.py(235): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(516): _slow_forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(530): __call__
./predictor/base_module.py(102): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(516): _slow_forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(530): __call__
/home/nodeflux/.local/lib/python3.6/site-packages/torch/jit/__init__.py(347): wrapper
/home/nodeflux/.local/lib/python3.6/site-packages/torch/jit/__init__.py(360): forward
/home/nodeflux/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(532): __call__
/home/nodeflux/.local/lib/python3.6/site-packages/torch/jit/__init__.py(277): _get_trace_graph
/home/nodeflux/.local/lib/python3.6/site-packages/torch/onnx/utils.py(236): _trace_and_get_graph_from_model
/home/nodeflux/.local/lib/python3.6/site-packages/torch/onnx/utils.py(279): _model_to_graph
/home/nodeflux/.local/lib/python3.6/site-packages/torch/onnx/utils.py(416): _export
/home/nodeflux/.local/lib/python3.6/site-packages/torch/onnx/utils.py(66): export
/home/nodeflux/.local/lib/python3.6/site-packages/torch/onnx/__init__.py(148): export
./exporter/onnx.py(21): export
/home/nodeflux/.local/lib/python3.6/site-packages/enforce/decorators.py(112): universal
./exporter/onnx.py(135): __call__
exporter/scripts/compatibility_check.py(241): export_check
exporter/scripts/compatibility_check.py(370): main
exporter/scripts/compatibility_check.py(465): <module>


Graph we tried to export:
graph(%input : Byte(1, 640, 640, 3),
      %score_threshold : Float(1),
      %iou_threshold : Float(1),
      %model.default_boxes : Float(50400, 4),
  ## multiple lines omitted
  %1507 : Tensor = onnx::NonMaxSuppression(%1502, %1504, %1505, %1506)
  %1508 : Tensor = onnx::Constant[value={2}]()
  %1509 : Tensor = onnx::Gather[axis=1](%1507, %1508)
  %1510 : Long(13) = onnx::Squeeze[axes=[1]](%1509) # /home/nodeflux/.local/lib/python3.6/site-packages/torchvision/ops/boxes.py:36:0
  %output : Float(1, 13, 16) = onnx::Gather[axis=1](%1496, %1510) # ./networks/modules/postprocess/utils/nms.py:117:0
  return (%output)

{'RetinaFace': [('efficientnet_b0', False)]}
[eval opset 11] trying to eval efficientnet_b0-RetinaFace with cpu:  [0/2]
[eval opset 11] trying to eval efficientnet_b0-RetinaFace with cuda:  [2/2]
{'cpu': [{'RetinaFace': [('efficientnet_b0', True)]}], 'cuda': [{'RetinaFace': [('efficientnet_b0', True)]}]}
```
## Command-Line Interface Tests
### [Installing Bash Automated Testing System](https://github.com/bats-core/bats-core/#npm)
```
npm install -g bats
```
### Running Command-Line Interface Tests using [BATS](https://github.com/bats-core/bats-core/)
#### Run all CLI tests
```
bash tests/cli/test_all.sh
```
example output
```
nodeflux-G7-7588 :: ~/nodeflux/vortex ‹development*› » bash tests/cli/test_all.sh
 ✓ test short train shufflenetv2_x0.5 on mnist
 ✓ validate shufflenetv2x050 on mnist
 ✓ validate shufflenetv2x050 on mnist with batch size 4
 ✓ export mnist
 ✓ export mnist with example input
 ✓ export mnist with example input and custom weights
 ✓ predict mnist
 ✓ predict mnist batch
 ✓ predict mnist custom weight
 ✓ predict mnist onnx
 ✓ predict mnist torchscript
 ✓ predict mnist onnx on cuda
 ✓ predict mnist torchscript on cuda
 ✓ predict mnist onnx batch size 4
 ✓ predict mnist torchscript batch size 4
 ✓ predict mnist onnx on cuda batch size 4
 ✓ predict mnist torchscript on cuda batch size 4
 ✓ validate shufflenetv2x050 on mnist onnx
 ✓ validate shufflenetv2x050 on mnist onnx with batch size 4
 ✓ validate shufflenetv2x050 on mnist torchscript
 ✓ validate shufflenetv2x050 on mnist torchscript with batch size 4
 ✓ onnx compatibility check on opset-version 11 retinaface efficientnet

22 tests, 0 failures
```
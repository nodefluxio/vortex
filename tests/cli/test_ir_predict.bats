#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export TEST_IMAGE=tests/images/cat.jpg
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME
    export OUTPUT_DIR=tests/output_predict_test

    if [ ! -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        bash "${BATS_TEST_DIRNAME}"/export_cmd
    else
        :
    fi
}

teardown() {
    : ## do nothing for now
}

predict_test_data_onnx() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}.onnx --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_torchscript() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}.pt --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_onnx_cuda() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}.onnx --image $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_torchscript_cuda() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}.pt --image $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_onnx_bs4() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}_bs4.onnx --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_torchscript_bs4() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}_bs4.pt --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_onnx_cuda_bs4() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}_bs4.onnx --image $TEST_IMAGE $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_torchscript_cuda_bs4() {
    python3.6 vortex/ir_runtime_predict.py --model ${OUTPUT_EXPERIMENT}_bs4.pt --image $TEST_IMAGE $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}


predict_test_data_onnx_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}.onnx --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_torchscript_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}.pt --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_onnx_cuda_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}.onnx --image $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_torchscript_cuda_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}.pt --image $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_onnx_bs4_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}_bs4.onnx --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_torchscript_bs4_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}_bs4.pt --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_onnx_cuda_bs4_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}_bs4.onnx --image $TEST_IMAGE $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

predict_test_data_torchscript_cuda_bs4_cli() {
    vortex ir_runtime_predict --model ${OUTPUT_EXPERIMENT}_bs4.pt --image $TEST_IMAGE $TEST_IMAGE --runtime cuda -o $OUTPUT_DIR
}

@test "predict test_data onnx" {
    run predict_test_data_onnx
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript" {
    run predict_test_data_torchscript
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx on cuda" {
    run predict_test_data_onnx_cuda
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript on cuda" {
    run predict_test_data_torchscript_cuda
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx batch size 4" {
    run predict_test_data_onnx_bs4
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript batch size 4" {
    run predict_test_data_torchscript_bs4
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx on cuda batch size 4" {
    run predict_test_data_onnx_cuda_bs4
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript on cuda batch size 4" {
    run predict_test_data_torchscript_cuda_bs4
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx cli" {
    run predict_test_data_onnx_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript cli" {
    run predict_test_data_torchscript_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx on cuda cli" {
    run predict_test_data_onnx_cuda_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript on cuda cli" {
    run predict_test_data_torchscript_cuda_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx batch size 4 cli" {
    run predict_test_data_onnx_bs4_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript batch size 4 cli" {
    run predict_test_data_torchscript_bs4_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data onnx on cuda batch size 4 cli" {
    run predict_test_data_onnx_cuda_bs4_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data torchscript on cuda batch size 4 cli" {
    run predict_test_data_torchscript_cuda_bs4_cli
    [ "$status" -eq 0 ]
}
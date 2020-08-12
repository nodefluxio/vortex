#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export TEST_IMAGE=tests/images/cat.jpg
    export OUTPUT_DIRECTORY=tests/output_test
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME

    if [ ! -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        bash "${BATS_TEST_DIRNAME}"/export_cmd
    else
        :
    fi
}

teardown() {
    : ## do nothing for now
}

validate_efficientnet_lite0_test_data_onnx() {
    python3 packages/development/vortex/development/ir_runtime_validate.py --model ${OUTPUT_EXPERIMENT}.onnx --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_onnx_bs4() {
    python3 packages/development/vortex/development/ir_runtime_validate.py --model ${OUTPUT_EXPERIMENT}_bs4.onnx --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

validate_efficientnet_lite0_test_data_onnx_cli() {
    vortex ir_runtime_validate --model ${OUTPUT_EXPERIMENT}.onnx --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_onnx_bs4_cli() {
    vortex ir_runtime_validate --model ${OUTPUT_EXPERIMENT}_bs4.onnx --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

@test "validate efficientnet_lite0 on test_data onnx" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_onnx
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data onnx with batch size 4" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_onnx_bs4
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data onnx cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_onnx_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data onnx with batch size 4 cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_onnx_bs4_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_onnx_IR_validation_cpu.md ]
}

validate_efficientnet_lite0_test_data_torchscript() {
    python3 packages/development/vortex/development/ir_runtime_validate.py --model ${OUTPUT_EXPERIMENT}.pt --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_torchscript_bs4() {
    python3 packages/development/vortex/development/ir_runtime_validate.py --model ${OUTPUT_EXPERIMENT}_bs4.pt  --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

validate_efficientnet_lite0_test_data_torchscript_cli() {
    vortex ir_runtime_validate --model ${OUTPUT_EXPERIMENT}.pt --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_torchscript_bs4_cli() {
    vortex ir_runtime_validate --model ${OUTPUT_EXPERIMENT}_bs4.pt  --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

@test "validate efficientnet_lite0 on test_data torchscript" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_torchscript
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data torchscript with batch size 4" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_torchscript_bs4
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data torchscript cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_torchscript_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data torchscript with batch size 4 cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_torchscript_bs4_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_torchscript_IR_validation_cpu.md ]
}
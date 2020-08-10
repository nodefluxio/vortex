#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export TEST_IMAGE=tests/images/cat.jpg
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME
}

teardown() {
    : ## do nothing for now
}

export_test_data() {
    python3.6 vortex/export.py --config $CONFIG_FILE
}

export_test_data_example_input() {
    python3.6 vortex/export.py --config $CONFIG_FILE --example-input $TEST_IMAGE
}

export_test_data_example_input_custom_weights() {
    python3.6 vortex/export.py --config $CONFIG_FILE --example-input $TEST_IMAGE --weights ${OUTPUT_EXPERIMENT}.pth
}

export_test_data_cli() {
    vortex export --config $CONFIG_FILE
}

export_test_data_example_input_cli() {
    vortex export --config $CONFIG_FILE --example-input $TEST_IMAGE
}

export_test_data_example_input_custom_weights_cli() {
    vortex export --config $CONFIG_FILE --example-input $TEST_IMAGE --weights ${OUTPUT_EXPERIMENT}.pth
}

@test "export test_data" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}

@test "export test_data with example input" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data_example_input
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}

@test "export test_data with example input and custom weights" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data_example_input_custom_weights
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}

@test "export test_data cli" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}

@test "export test_data with example input cli" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data_example_input_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}

@test "export test_data with example input and custom weights cli" {
    if [ -f ${OUTPUT_EXPERIMENT}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}.pt
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx
    fi
    if [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]; then
        rm ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt
    fi
    run export_test_data_example_input_custom_weights_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}.pt ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.onnx ]
    [ -f ${OUTPUT_EXPERIMENT}_bs${BATCH_SIZE}.pt ]
}
#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export OUTPUT_DIRECTORY=tests/output_test
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME
}

teardown() {
    : ## do nothing for now
}

validate_efficientnet_lite0_test_data() {
    python3 packages/development/vortex/development/validate.py --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_bs4() {
    python3 packages/development/vortex/development/validate.py --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

validate_efficientnet_lite0_test_data_cli() {
    vortex validate --config $CONFIG_FILE
}

validate_efficientnet_lite0_test_data_cli() {
    vortex validate --config $CONFIG_FILE --batch-size $BATCH_SIZE
}

@test "validate efficientnet_lite0 on test_data" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data with batch size 4" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_bs4
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]
}

@test "validate efficientnet_lite0 on test_data with batch size 4 cli" {
    if [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]; then
        rm ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md
    fi
    run validate_efficientnet_lite0_test_data_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}/reports/${EXPERIMENT_NAME}_validation_cpu.md ]
}
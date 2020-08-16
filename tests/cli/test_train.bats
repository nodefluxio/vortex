#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME
}

teardown() {
    : ## do nothing for now
}

short_train_efficientnet_lite0_test_data() {
    python3 src/development/vortex/development/train.py --config $CONFIG_FILE
}

short_train_efficientnet_lite0_test_data_cli() {
    vortex train --config $CONFIG_FILE
}

@test "test short train efficientnet_lite0 on test data" {
    if [ -f ${OUTPUT_EXPERIMENT}.pth ]; then
        rm ${OUTPUT_EXPERIMENT}.pth
    fi
    run short_train_efficientnet_lite0_test_data
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.pth ]
}


@test "test short train efficientnet_lite0 on test data cli" {
    if [ -f ${OUTPUT_EXPERIMENT}.pth ]; then
        rm ${OUTPUT_EXPERIMENT}.pth
    fi
    run short_train_efficientnet_lite0_test_data_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}.pth ]
}
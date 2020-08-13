#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export TEST_IMAGE=tests/images/cat.jpg
    export PTH_FILE=tests/output_test/$EXPERIMENT_NAME/$EXPERIMENT_NAME.pth
    export OUTPUT_DIR=tests/output_predict_test
}

teardown() {
    : ## do nothing for now
}

predict_test_data() {
    python3 src/development/vortex/development/predict.py --config $CONFIG_FILE --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_batch() {
    python3 src/development/vortex/development/predict.py --config $CONFIG_FILE --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_custom_weights() {
    python3 src/development/vortex/development/predict.py --config $CONFIG_FILE --image $TEST_IMAGE --weights $PTH_FILE -o $OUTPUT_DIR
}

predict_test_data_cli() {
    vortex predict --config $CONFIG_FILE --image $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_batch_cli() {
    vortex predict --config $CONFIG_FILE --image $TEST_IMAGE $TEST_IMAGE -o $OUTPUT_DIR
}

predict_test_data_custom_weights_cli() {
    vortex predict --config $CONFIG_FILE --image $TEST_IMAGE --weights $PTH_FILE -o $OUTPUT_DIR
}

@test "predict test_data" {
    run predict_test_data
    [ "$status" -eq 0 ]
}

@test "predict test_data batch" {
    run predict_test_data_batch
    [ "$status" -eq 0 ]
}

@test "predict test_data custom weight" {
    run predict_test_data_custom_weights
    [ "$status" -eq 0 ]
}

@test "predict test_data cli" {
    run predict_test_data_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data batch cli" {
    run predict_test_data_batch_cli
    [ "$status" -eq 0 ]
}

@test "predict test_data custom weight cli" {
    run predict_test_data_custom_weights_cli
    [ "$status" -eq 0 ]
}
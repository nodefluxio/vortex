#!/usr/bin/env bats

setup() {
    export EXPERIMENT_NAME=test_classification_pipelines
    export CONFIG_FILE=tests/config/test_classification_pipelines_cli.yml
    export BATCH_SIZE=4
    export OUTPUT_EXPERIMENT=tests/output_test/$EXPERIMENT_NAME/hypopt/learning_rate_search
    export OPTCONFIG_FILE=tests/config/test_hypopt_lr_search.yml
}

teardown() {
    : ## do nothing for now
}

hpyopt_classification() {
    python3 src/development/vortex/development/hypopt.py --config $CONFIG_FILE --optconfig $OPTCONFIG_FILE
}

hpyopt_classification_cli() {
    vortex hypopt --config $CONFIG_FILE --optconfig $OPTCONFIG_FILE
}

@test "learning rate search on test classification" {
    if [ -f ${OUTPUT_EXPERIMENT} ]; then
        rm -r ${OUTPUT_EXPERIMENT}
    fi
    run hpyopt_classification
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}/best_params.txt ]
}

@test "learning rate search on test classification cli" {
    if [ -f ${OUTPUT_EXPERIMENT} ]; then
        rm -r ${OUTPUT_EXPERIMENT}
    fi
    run hpyopt_classification_cli
    [ "$status" -eq 0 ]
    [ -f ${OUTPUT_EXPERIMENT}/best_params.txt ]
}
#!/usr/bin/env bats

setup() {
    export TEST_MODEL=RetinaFace
    export TEST_BACKBONE=efficientnet_b0
    mv COMPATIBILITY_REPORT_opset11.md COMPATIBILITY_REPORT_opset11.md.ori
}

teardown() {
    mv COMPATIBILITY_REPORT_opset11.md.ori COMPATIBILITY_REPORT_opset11.md
}

compatibility_check_efficientnetb0() {
    python3 scripts/export_check/compatibility_check.py --opset-version 11 --models $TEST_MODEL --backbones $TEST_BACKBONE
}

@test "onnx compatibility check on opset-version 11 retinaface efficientnet" {
    run compatibility_check_efficientnetb0
    [ "$status" -eq 0 ]
}
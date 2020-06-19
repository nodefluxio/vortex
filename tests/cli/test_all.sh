#!/usr/bin/env bash

bats tests/cli/test_train.bats \
     tests/cli/test_validate.bats \
     tests/cli/test_export.bats \
     tests/cli/test_predict.bats \
     tests/cli/test_ir_predict.bats \
     tests/cli/test_ir_validate.bats \
     tests/cli/test_compatibility_check.bats \
     tests/cli/test_hypopt.bats
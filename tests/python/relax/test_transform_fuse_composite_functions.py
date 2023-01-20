# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import tvm

from tvm import relax
from tvm.script import relax as R
from tvm.relax.dpl.pattern import make_fused_bias_activation_pattern, wildcard, is_op


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.relu(R.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = R.nn.relu(R.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


@tvm.script.ir_module
class Branch:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight)
            relu1 = R.nn.relu(conv1)
            gelu1 = R.nn.gelu(conv1)

            out = relax.op.add(relu1, gelu1)
            R.output(out)

        return out


@tvm.script.ir_module
class BranchMerge:
    @R.function
    def main(
        x1: R.Tensor((10,), "float32"),
        x2: R.Tensor((10,), "float32"),
    ):
        with R.dataflow():
            relu1 = R.nn.relu(x1)
            gelu1 = R.nn.gelu(x2)

            out = relax.op.add(relu1, gelu1)
            R.output(out)

        return out


def test_conv2d_relu_x2():
    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation="relax.nn.relu")

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("dnnl.conv2d_relu", pat)]),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(Conv2dReLUx2).script())


def test_branch():
    conv_pat = make_fused_bias_activation_pattern("relax.nn.conv2d")
    relu_pat = is_op("relax.nn.relu")(wildcard())
    add_pat = is_op("relax.add")(wildcard(), wildcard())

    gelu_pat = is_op("relax.nn.gelu")(wildcard())

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("compiler_A.conv2d", conv_pat),
                                              ("compiler_A.relu", relu_pat),
                                              ("compiler_A.add", add_pat),
                                              ("compiler_B.gelu", gelu_pat),
                                              ]),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(Branch).script())


def test_branch_merge():
    relu_pat = is_op("relax.nn.relu")(wildcard())
    gelu_pat = is_op("relax.nn.gelu")(wildcard())
    add_pat = is_op("relax.add")(wildcard(), wildcard())

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("compiler_A.gelu", gelu_pat),
                                              ("compiler_A.relu", relu_pat),
                                              ("compiler_A.add", add_pat),
                                              ]),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(BranchMerge).script())


if __name__ == "__main__":
    # pytest.main([__file__])
    # test_conv2d_relu_x2()
    test_branch()
    # test_branch_merge()

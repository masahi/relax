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

import numpy as np
import tvm
import tvm.testing

from tvm import relax, relay
from tvm.script import relax as R
from tvm.relax.dpl import *
from tvm.contrib.cutlass.build import finalize_modules_relax


op_name = "cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8"

op_def = """
  using cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8 =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,

    cutlass::epilogue::thread::LinearCombinationRelu<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;
"""


def make_conv_pattern(conv_name, with_bias=False, activation=None):
    data = wildcard()
    weight = wildcard()
    conv = is_op(conv_name)(data, weight)

    if with_bias:
        bias = wildcard()
        conv_out = is_op("relax.add")(conv, bias)
    else:
        conv_out = conv

    if activation:
        return is_op(activation)(conv_out)

    return conv_out


@tvm.script.ir_module
class Conv2dBiasReLU:
    @R.function
    def conv2d(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight: R.Tensor((32, 3, 3, 16), "float16"),
        bias: R.Tensor((1, 1, 1, 32), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(
                relax.op.add(
                    relax.op.nn.conv2d(
                        data, weight, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    ),
                    bias,
                )
            )
            R.output(conv1)

        return conv1


@tvm.script.ir_module
class Conv2dBiasReLUPartitioned:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight: R.Tensor((32, 3, 3, 16), dtype="float16"),
        bias: R.Tensor((1, 1, 1, 32), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 32), dtype="float16"):
        # block 0
        with R.dataflow():
            gv: R.Tensor(
                (16, 32, 32, 32), dtype="float16"
            ) = fused_relax_nn_conv2d_relax_add_relax_nn_relu(data, weight, bias)
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_add_relax_nn_relu(
        data1: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((32, 3, 3, 16), dtype="float16"),
        bias1: R.Tensor((1, 1, 1, 32), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 32), dtype="float16"):
        R.func_attr(
            {"Codegen": "cutlass", "global_symbol": "fused_relax_nn_conv2d_relax_add_relax_nn_relu"}
        )

        @R.function
        def fused_relax_nn_conv2d_relax_add_relax_nn_relu_inner(
            data1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            weight1: R.Tensor((32, 3, 3, 16), dtype="float16"),
            bias1: R.Tensor((1, 1, 1, 32), dtype="float16"),
        ) -> R.Tensor((16, 32, 32, 32), dtype="float16"):
            # function attr dict
            R.func_attr({"Primitive": 1, "Composite": "conv2d_bias_relu"})
            # block 0
            with R.dataflow():
                lv: R.Tensor((16, 32, 32, 32), dtype="float16") = R.nn.conv2d(
                    data1,
                    weight1,
                    strides=[1, 1],
                    padding=[1, 1],
                    dilation=[1, 1],
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="",
                )
                lv1: R.Tensor((16, 32, 32, 32), dtype="float16") = R.add(lv, bias1)
                gv1: R.Tensor((16, 32, 32, 32), dtype="float16") = R.nn.relu(lv1)
                R.output(gv1)
            return gv1

        return fused_relax_nn_conv2d_relax_add_relax_nn_relu_inner(data1, weight1, bias1)


def annotate_attributes(mod):
    # TODO: automate
    f_name = "fused_relax_nn_conv2d_relax_add_relax_nn_relu"
    f = mod[f_name]

    for k, v in {
        "arg0_dtype": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float32",
        "arg0_shape": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float32",
        "op_type": "conv2d_bias_relu",
        "arg0_shape": [16, 32, 32, 16],
        "arg1_shape": [32, 3, 3, 16],
        "ret_shape": [16, 32, 32, 32],
        "strides": [1, 1],
        "padding": [1, 1],
        "dilation": [1, 1],
        "cutlass_op_name": op_name,
        "cutlass_op_def": op_def,
    }.items():
        f = f.with_attr(k, v)

    mod[f_name] = f

    return mod


def test_conv2d_partition():
    mod = Conv2dBiasReLU
    pat = make_conv_pattern("relax.nn.conv2d", True, "relax.nn.relu")
    mod = relax.transform.FuseOpsByPattern(pat)(mod)

    print(mod.script())


def get_relay_conv2d_bias_relu(d_shape, w_shape):
    data = relay.var("data", shape=d_shape)
    weight = relay.var("weight", shape=w_shape)
    bias = relay.var("bias", shape=(1, 1, 1, w_shape[0]))
    return relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
        )
        + bias
    )


def get_ref(data_np, weight_np, bias_np):
    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_bias_relu(data_np.shape, weight_np.shape))

    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"]})]
        )
        relay_mod = seq(relay_mod)

    ref = (
        relay.create_executor("graph", mod=relay_mod, device=tvm.gpu(0), target="cuda")
        .evaluate()(*[data_np, weight_np, bias_np])
        .numpy()
    )

    return ref


def test_conv2d_offload():
    data_np = np.random.randn(16, 32, 32, 16).astype("float16")
    weight_np = np.random.randn(32, 3, 3, 16).astype("float16")
    bias_np = np.random.randn(1, 1, 1, 32).astype("float16")

    seq = tvm.transform.Sequential(
        [
            relax.transform.RunCodegen(),
            relax.transform.RemoveUnusedFunctions(),
        ]
    )

    mod = annotate_attributes(Conv2dBiasReLUPartitioned)
    mod = seq(mod)

    target = tvm.target.Target("cuda")
    ex = relax.vm.build(mod, target)
    ex = finalize_modules_relax(ex)

    dev = tvm.gpu(0)
    vm = relax.VirtualMachine(ex, dev)

    data = tvm.nd.array(data_np, dev)
    weight = tvm.nd.array(weight_np, dev)
    bias = tvm.nd.array(bias_np, dev)
    out = vm["main"](data, weight, bias).numpy()

    ref = get_ref(data_np, weight_np, bias_np)

    print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))


if __name__ == "__main__":
    test_conv2d_offload()

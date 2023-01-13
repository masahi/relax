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

from tvm import relax, topi, relay
from tvm.script import relax as R
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.ir.module import IRModule
from tvm.relax.dpl import *


@mutator
class OperatorLegalizer(PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def _convert_op(self, call):
        if call.op == tvm.ir.Op.get("relax.nn.conv2d"):
            args = call.args
            attrs = call.attrs
            return self.builder_.call_te(
                topi.nn.conv2d,
                input=args[0],
                filter=args[1],
                strides=attrs.strides,
                padding=attrs.padding,
                dilation=attrs.dilation,
                data_layout=attrs.data_layout,
                kernel_layout=attrs.kernel_layout,
                out_dtype=attrs.out_dtype if attrs.out_dtype != "" else None,
            )
        if call.op == tvm.ir.Op.get("relax.nn.relu"):
            return self.builder_.call_te(topi.nn.relu, call.args[0])

        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        return self._convert_op(call)


def make_conv_pattern(conv_name, with_bias=False, activation=None):
    data = wildcard()
    weight = wildcard()
    conv = is_op(conv_name)(data, weight)

    if with_bias:
        bias = wildcard()
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv

    return is_op(activation)(conv_out)


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def conv2d(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2d = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2d)

        return conv2d


def get_relay_conv2d_relu_x2(d_shape, w_shape):
    data = relay.var("data", shape=d_shape)
    weight1 = relay.var("weight1", shape=w_shape)
    weight2 = relay.var("weight2", shape=w_shape)
    conv1 = relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight1,
            kernel_size=w_shape[2:],
            padding=(1, 1),
        )
    )
    return relay.nn.relu(
        relay.nn.conv2d(
            data=conv1,
            weight=weight2,
            kernel_size=w_shape[2:],
            padding=(0, 0),
        )
    )


def test_conv2d_partition():
    mod = Conv2dReLUx2
    pat = make_conv_pattern("relax.nn.conv2d", False, "relax.nn.relu")
    mod = relax.transform.FuseOpsByPattern(pat)(mod)

    print(mod.script())

    mod = OperatorLegalizer(mod).transform()

    target = tvm.target.Target("llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    f = vm["conv2d"]

    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
    weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")
    out = f(tvm.nd.array(data_np), tvm.nd.array(weight1_np), tvm.nd.array(weight2_np)).numpy()

    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_relu_x2(data_np.shape, weight1_np.shape))

    ref = (
        relay.create_executor("graph", mod=relay_mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight1_np, weight2_np])
        .numpy()
    )

    print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))


def partition_for_dnnl_relay(mod, params=None):
    from tvm.relay.build_module import bind_params_by_name
    from tvm.relay.op.contrib import dnnl
    from tvm.relay import transform

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.CanonicalizeOps(),
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.SimplifyExpr(),
            transform.FoldConstant(),
            transform.Legalize(),
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)

    return mod


def test_dnnl_relay():
    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
    weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")

    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_relu_x2(data_np.shape, weight1_np.shape))

    mod = partition_for_dnnl_relay(relay_mod)
    print(mod)
    return
    target = "llvm"
    dev = tvm.device(target, 0)

    with tvm.transform.PassContext(opt_level=3):
        func = relay.create_executor(
            "graph", mod=mod, device=dev, target=target
        ).evaluate()


@tvm.script.ir_module
class Conv2dReLUx2Partitioned:
    @R.function
    def main(data: R.Tensor((1, 64, 56, 56), dtype="float32"), weight1: R.Tensor((64, 64, 3, 3), dtype="float32"), weight2: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = fused_relax_nn_conv2d_relax_nn_relu(data, weight1)
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = fused_relax_nn_conv2d_relax_nn_relu1(lv, weight2)
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(data1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight11: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Codegen": "dnnl", "global_symbol": "fused_relax_nn_conv2d_relax_nn_relu"})

        @R.function
        def fused_relax_nn_conv2d_relax_nn_relu_inner(data1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight11: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            # function attr dict
            R.func_attr({"Primitive": 1, "Composite": "conv2d_relu"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(data1, weight11, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="")
                gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
                R.output(gv1)
            return gv1

        return fused_relax_nn_conv2d_relax_nn_relu_inner(data1, weight11)


    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu1(conv1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight21: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Codegen": "dnnl", "global_symbol": "fused_relax_nn_conv2d_relax_nn_relu1"})
        @R.function
        def fused_relax_nn_conv2d_relax_nn_relu1_inner(conv1: R.Tensor((1, 64, 56, 56), dtype="float32"), weight21: R.Tensor((64, 64, 3, 3), dtype="float32")) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            # function attr dict
            R.func_attr({"Primitive": 1, "Composite": "conv2d_relu"})
            # block 0
            with R.dataflow():
                lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(conv1, weight21, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="")
                gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv2)
                R.output(gv2)
            return gv2

        return fused_relax_nn_conv2d_relax_nn_relu1_inner(conv1, weight21)


def test_dnnl_offload():
    relax.transform.RunCodegen()(Conv2dReLUx2Partitioned)
#    print(


if __name__ == "__main__":
    # test_conv2d_partition()
    test_dnnl_offload()
    # test_dnnl_relay()

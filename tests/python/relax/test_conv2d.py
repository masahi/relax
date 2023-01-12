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


if __name__ == "__main__":
    test_conv2d_partition()

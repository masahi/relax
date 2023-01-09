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

from tvm.relax.analysis import get_var2val
from tvm import relax, topi
from tvm.script import relax as R
from tvm.relax.expr_functor import mutator, PyExprMutator, visitor, PyExprVisitor
from tvm.ir.module import IRModule
from tvm.relax.dpl import *


@tvm.script.ir_module
class Conv2dReLU:
    # T.func_attr({"global_symbol": "main", "tir.noalias": True})
    @R.function
    def conv2d(
        data: R.Tensor((1, 64, 56, 56), "float32"), weight: R.Tensor((64, 64, 3, 3), "float32")
    ):
        return relax.op.nn.relu(relax.op.nn.conv2d(data, weight, padding=(1, 1)))


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


def test_conv2d_run():
    mod = OperatorLegalizer(Conv2dReLU).transform()
    target = tvm.target.Target("llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    f = vm["conv2d"]
    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    weight_np = np.random.randn(64, 64, 3, 3).astype("float32")
    out = f(tvm.nd.array(data_np), tvm.nd.array(weight_np))
    print(out.numpy())


def append_eltwise_ops(op, eltwise):
    if eltwise == "gelu":
        const1 = wildcard()
        const2 = wildcard()
        const3 = wildcard()
        div = is_op("divide")(op, const1)
        erf_val = is_op("erf")(div)
        added_erf_val = is_op("add")(erf_val, const2)
        mul_val = is_op("multiply")(op, added_erf_val)
        op = is_op("multiply")(mul_val, const3)
    elif eltwise == "swish":
        sig_out = is_op("sigmoid")(op)
        op = is_op("multiply")(op, sig_out)
    elif eltwise == "mish":
        const1 = wildcard()
        exp = is_op("exp")(op)
        add = is_op("add")(exp, const1)
        log = is_op("log")(add)
        tanh = is_op("tanh")(log)
        op = is_op("multiply")(op, tanh)
    elif eltwise:
        op = is_op(eltwise)(op)
    return op


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return append_eltwise_ops(conv_out, with_eltwise)


@visitor
class MatchConv2d(PyExprVisitor):
    def __init__(self, binding) -> None:
        super().__init__()
        self.pat = make_conv_pattern("relax.nn.conv2d", False, "relax.nn.relu")
        self.binding = binding

    def visit_call_(self, call):
        for arg in call.args:
            self.visit_expr(arg)
        self.visit_expr(call.op)

        print(self.pat.match(call, self.binding))


def test_conv2d_match():
    mod = Conv2dReLU
    matcher = MatchConv2d(get_var2val(mod["conv2d"]))

    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            matcher.visit_expr(func)


if __name__ == "__main__":
    # test_conv2d_match()
    test_conv2d_run()
